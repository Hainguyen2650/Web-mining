"""Validate a complete cached collection and freeze normalized CSV tables."""

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import platform

from collect import API, cache_name, next_url
from common import ROOT, load_seeds, positive_int, read_json, sha256, write_csv, write_json


def cached_page(raw_dir, filename, expected_url):
    if filename != cache_name(expected_url):
        raise ValueError('Cache filename does not match expected URL')
    page = read_json(raw_dir / filename)
    if page.get('url') != expected_url or page.get('status') not in (200, 204) or not page.get('collected_at'):
        raise ValueError(f'Invalid cached response: {filename}')
    return page


def prepare(seeds_path, raw_dir, output_dir, results_dir):
    seeds_path, raw_dir, output_dir, results_dir = map(Path, (seeds_path, raw_dir, output_dir, results_dir))
    seeds = load_seeds(seeds_path)
    state = read_json(raw_dir / 'collection.json')['repositories']
    developers, repositories, edges = {}, {}, {}
    counts = Counter(contributor_records=0, duplicate_edges=0,
                     anonymous_or_unidentified_excluded=0, bot_records_excluded=0)
    excluded, source_hashes, collected_times = {}, {}, []
    for seed in seeds:
        name = seed['full_name']
        entry = state.get(name, {})
        if entry.get('status') not in ('complete', 'excluded'):
            raise ValueError(f'Incomplete collection for {name}; resume collection first')
        metadata = cached_page(raw_dir, entry['metadata'], f'{API}/repos/{name}')
        source_hashes[entry['metadata']] = sha256(raw_dir / entry['metadata'])
        collected_times.append(metadata['collected_at'])
        repo = metadata['data']
        if any(repo.get(key) is None for key in ('id', 'full_name', 'private', 'fork', 'archived')):
            raise ValueError(f'Incomplete metadata for {name}')
        ineligible = repo['private'] or repo['fork'] or repo['archived']
        if entry['status'] == 'excluded':
            if not ineligible:
                raise ValueError(f'Unjustified exclusion: {name}')
            excluded[name] = entry['reason']
            continue
        if ineligible:
            raise ValueError(f'Ineligible repository marked complete: {name}')
        repo_id = positive_int(repo['id'], 'repo_id')
        if repo_id in repositories:
            raise ValueError('Seed aliases resolve to the same repository')
        repositories[repo_id] = {'repo_id': repo_id, 'full_name': repo['full_name'],
                                'language': repo.get('language') or '',
                                'topics': '|'.join(repo.get('topics') or []),
                                'stars': repo.get('stargazers_count') or 0,
                                'description': repo.get('description') or '',
                                'family': seed['family'], 'collected_at': metadata['collected_at']}
        expected = f'{API}/repos/{name}/contributors?per_page=100&page=1'
        seen_pages = set()
        for filename in entry.get('pages', []):
            if expected is None or expected in seen_pages:
                raise ValueError(f'Duplicate or unexpected contributor page for {name}')
            seen_pages.add(expected)
            page = cached_page(raw_dir, filename, expected)
            source_hashes[filename] = sha256(raw_dir / filename)
            collected_times.append(page['collected_at'])
            if not isinstance(page['data'], list) or (page['status'] == 204 and page['data']):
                raise ValueError('Invalid contributor page')
            for user in page['data']:
                if not isinstance(user, dict):
                    raise ValueError('Invalid contributor record')
                counts['contributor_records'] += 1
                if user.get('type') == 'Anonymous' or not user.get('id') or not user.get('login'):
                    counts['anonymous_or_unidentified_excluded'] += 1
                    continue
                if user.get('type') == 'Bot' or user['login'].lower().endswith(('[bot]', '-bot')):
                    counts['bot_records_excluded'] += 1
                    continue
                if user.get('type') != 'User':
                    raise ValueError(f'Unexpected account type: {user.get("type")}')
                developer_id = positive_int(user['id'], 'developer_id')
                contributions = positive_int(user.get('contributions'), 'contributions')
                key = developer_id, repo_id
                if key in edges:
                    counts['duplicate_edges'] += 1
                    if edges[key] != contributions:
                        raise ValueError('Conflicting duplicate contribution counts; recollect a consistent snapshot')
                    continue
                edges[key] = contributions
                developers[developer_id] = {'developer_id': developer_id, 'login': user['login'], 'account_type': 'User'}
            expected = next_url(page['headers'], expected, repo_id)
        if expected is not None:
            raise ValueError(f'Missing contributor page for {name}')
    if not repositories:
        raise ValueError('No eligible repositories in the collection')
    developer_rows = [developers[key] for key in sorted(developers)]
    repository_rows = [repositories[key] for key in sorted(repositories)]
    edge_rows = [{'developer_id': u, 'repo_id': r, 'contributions': count} for (u, r), count in sorted(edges.items())]
    write_csv(output_dir / 'developers.csv', developer_rows, ['developer_id', 'login', 'account_type'])
    write_csv(output_dir / 'repositories.csv', repository_rows,
              ['repo_id', 'full_name', 'language', 'topics', 'stars', 'description', 'family', 'collected_at'])
    write_csv(output_dir / 'interactions.csv', edge_rows, ['developer_id', 'repo_id', 'contributions'])
    quality = dict(counts)
    quality.update(complete=True, selected_repositories=len(seeds), included_repositories=len(repositories),
                   excluded_repositories=excluded, developers=len(developers), edges=len(edges),
                   missing_pages=0, failed_pages=0, missing_required_fields=0,
                   anonymous_scope='Only identified contributors requested; anonymous records are not enumerated',
                   bot_rule='type Bot or login ending in [bot] or -bot; remaining automation may be missed')
    write_json(results_dir / 'data_quality.json', quality)
    manifest = {'schema_version': 1, 'seed_sha256': sha256(seeds_path),
                'seeds': seeds, 'raw_sha256': source_hashes,
                'collection_start': min(collected_times), 'collection_end': max(collected_times),
                'processed_sha256': {name: sha256(output_dir / name) for name in
                                     ('developers.csv', 'repositories.csv', 'interactions.csv')},
                'python': platform.python_version(),
                'dependencies': {name: version(name) for name in ('networkx', 'numpy', 'scipy', 'matplotlib')}}
    write_json(results_dir / 'manifest.json', manifest)
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=Path, default=ROOT / 'data/seeds.csv')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    try:
        print(prepare(args.seeds, args.raw_dir, args.output_dir, args.results_dir))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Preparation stopped: {error}\n')


if __name__ == '__main__':
    main()
