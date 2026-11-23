"""Serial, resumable collection of public GitHub repository contributor snapshots."""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from common import ROOT, load_seeds, read_json, write_json

API = 'https://api.github.com'
SAFE_HEADERS = {'link', 'etag', 'x-ratelimit-limit', 'x-ratelimit-remaining',
                'x-ratelimit-reset', 'retry-after'}
REPO_FIELDS = ('id', 'full_name', 'private', 'fork', 'archived', 'language',
               'topics', 'stargazers_count', 'description')
USER_FIELDS = ('id', 'login', 'type', 'contributions')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'api.github.com' or parsed.username or parsed.fragment:
        raise ValueError('Only HTTPS api.github.com URLs are permitted')


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def next_url(headers, current, repo_id=None):
    for part in headers.get('link', '').split(','):
        match = re.search(r'<([^>]+)>;\s*rel="next"', part)
        if match:
            url = match.group(1)
            validate_url(url)
            allowed_paths = {urlparse(current).path}
            if repo_id is not None:
                allowed_paths.add(f'/repositories/{repo_id}/contributors')
            if urlparse(url).path not in allowed_paths:
                raise ValueError('Pagination changed the repository endpoint')
            return url
    return None


def cache_name(url):
    return hashlib.sha256(url.encode()).hexdigest() + '.json'


def minimize(payload, contributors):
    if contributors:
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError('Contributor response must be a list of objects')
        return [{key: row[key] for key in USER_FIELDS if key in row} for row in payload]
    if not isinstance(payload, dict) or not {'id', 'full_name', 'private', 'fork', 'archived'} <= payload.keys():
        raise ValueError('Repository metadata is incomplete')
    return {key: payload.get(key) for key in REPO_FIELDS}


class Client:
    def __init__(self, raw_dir, budget=1000, offline=False):
        if budget < 1:
            raise ValueError('Request budget must be positive')
        self.raw_dir = Path(raw_dir)
        self.budget = budget
        self.offline = offline
        self.requests = 0
        self.last_headers = {}
        self.opener = build_opener(SafeRedirect())

    def fetch(self, url, contributors=False):
        validate_url(url)
        path = self.raw_dir / cache_name(url)
        if path.exists():
            cached = read_json(path)
            if cached.get('url') != url or cached.get('status') not in (200, 204):
                raise ValueError(f'Invalid cache entry: {path.name}')
            minimize(cached['data'], contributors)
            return cached
        if self.offline:
            raise RuntimeError(f'Uncached URL in offline mode: {url}')
        headers = {'User-Agent': 'web-mining-course-project', 'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28'}
        token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        for attempt in range(3):
            if self.requests >= self.budget:
                raise RuntimeError('Request budget reached; rerun to resume cached collection')
            if self.last_headers.get('x-ratelimit-remaining') == '0':
                raise RuntimeError('GitHub rate limit reached; resume after reset ' + self.last_headers.get('x-ratelimit-reset', 'unknown'))
            self.requests += 1
            try:
                with self.opener.open(Request(url, headers=headers), timeout=30) as response:
                    status = response.status
                    self.last_headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_HEADERS}
                    payload = [] if status == 204 else json.load(response)
                if status not in (200, 204) or (status == 204 and not contributors):
                    raise ValueError(f'Unexpected HTTP status {status}')
                cached = {'url': url, 'status': status, 'collected_at': utc_now(),
                          'headers': self.last_headers, 'data': minimize(payload, contributors)}
                write_json(path, cached)
                time.sleep(0.15)
                return cached
            except HTTPError as error:
                self.last_headers = {k.lower(): v for k, v in error.headers.items() if k.lower() in SAFE_HEADERS}
                if self.last_headers.get('x-ratelimit-remaining') == '0':
                    raise RuntimeError('GitHub rate limit reached; resume after reset ' + self.last_headers.get('x-ratelimit-reset', 'unknown')) from None
                retry = self.last_headers.get('retry-after')
                if error.code not in (429, 500, 502, 503, 504) and not (error.code == 403 and retry):
                    raise RuntimeError(f'HTTP {error.code} for {url}') from None
                delay = 2 ** attempt
                if retry:
                    try:
                        delay = max(delay, float(retry))
                    except ValueError:
                        delay = max(delay, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                if delay > 30:
                    raise RuntimeError(f'Server requested a {delay:.0f}s pause; rerun later to resume') from None
            except (URLError, TimeoutError):
                delay = 2 ** attempt
            if attempt < 2:
                time.sleep(delay)
        raise RuntimeError(f'Request failed after 3 attempts: {url}')


def collect(seeds_path, raw_dir, budget=1000, offline=False):
    seeds = load_seeds(seeds_path)
    client = Client(raw_dir, budget, offline)
    state = {'started_at': utc_now(), 'seed_file': Path(seeds_path).name, 'repositories': {}}
    for seed in seeds:
        name = seed['full_name']
        entry = {'status': 'incomplete', 'pages': []}
        state['repositories'][name] = entry
        try:
            metadata_url = f'{API}/repos/{name}'
            metadata = client.fetch(metadata_url)
            entry['metadata'] = cache_name(metadata_url)
            repo = metadata['data']
            if repo['private'] or repo['fork'] or repo['archived']:
                entry.update(status='excluded', reason='private, fork, or archived repository')
            else:
                url = f'{API}/repos/{name}/contributors?per_page=100&page=1'
                seen = set()
                while url:
                    if url in seen:
                        raise ValueError('Pagination loop detected')
                    seen.add(url)
                    page = client.fetch(url, contributors=True)
                    entry['pages'].append(cache_name(url))
                    url = next_url(page['headers'], url, repo['id'])
                entry['status'] = 'complete'
            print(f"{name}: {entry['status']} ({len(entry['pages'])} pages)", flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            entry['error'] = str(error)
            print(f'{name}: incomplete: {error}', flush=True)
            state.update(requests_this_run=client.requests, finished_at=utc_now())
            write_json(Path(raw_dir) / 'collection.json', state)
            raise
        state.update(requests_this_run=client.requests, finished_at=utc_now())
        write_json(Path(raw_dir) / 'collection.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=Path, default=ROOT / 'data/seeds.csv')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--offline', action='store_true', help='Rebuild collection state using cache only')
    args = parser.parse_args()
    try:
        collect(args.seeds, args.raw_dir, args.budget, args.offline)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f'Collection stopped: {error}\n')


if __name__ == '__main__':
    main()
