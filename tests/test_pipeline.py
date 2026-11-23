"""Small network and mocked HTTP checks; no GitHub access is needed."""

from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
import re
import xml.etree.ElementTree as ET
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import networkx as nx
from analyze import analyze, build_graph, communities, pair_agreement, rankings, repository_projection
from collect import API, Client, cache_name, collect, next_url, validate_url
from common import read_json, sha256, write_csv, write_json
from prepare import prepare
from figures import CAPTIONS, additional_figures, overlap_matrices


class Response(BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.status = status
        self.headers = headers or {}


def metadata(identifier=11, name='example/a'):
    return {'id': identifier, 'full_name': name, 'private': False, 'fork': False,
            'archived': False, 'language': 'Python', 'topics': ['testing'],
            'stargazers_count': 10, 'description': 'Fixture'}


def user(identifier=1, login='alice', count=3, account_type='User'):
    return {'id': identifier, 'login': login, 'contributions': count, 'type': account_type}


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.raw = self.root / 'raw'
        self.seeds = self.root / 'seeds.csv'
        write_csv(self.seeds, [{'full_name': 'example/a', 'family': 'fixture', 'reason': 'test'}],
                  ['full_name', 'family', 'reason'])

    def tearDown(self):
        self.directory.cleanup()

    def test_cache_minimizes_personal_data_and_reuses_without_network(self):
        client = Client(self.raw)
        url = f'{API}/repos/example/a/contributors?per_page=100&page=1'
        payload = [{**user(), 'email': 'not-for-storage@example.invalid', 'avatar_url': 'unused'}]
        with patch.object(client.opener, 'open', return_value=Response(payload, headers={'Authorization': 'secret'})) as opened, patch('collect.time.sleep'):
            first = client.fetch(url, contributors=True)
            second = client.fetch(url, contributors=True)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(first, second)
        self.assertNotIn('email', first['data'][0])
        self.assertNotIn('authorization', first['headers'])
        self.assertNotIn('secret', (self.raw / cache_name(url)).read_text())

    def test_pagination_accepts_numeric_repo_but_rejects_other_repo_and_host(self):
        url = f'{API}/repos/example/a/contributors?per_page=100&page=1'
        target = f'{API}/repositories/11/contributors?page=2&per_page=100'
        self.assertEqual(next_url({'link': f'<{target}>; rel="next"'}, url, 11), target)
        for target in ('https://evil.invalid/contributors?page=2', f'{API}/repositories/12/contributors?page=2'):
            with self.assertRaises(ValueError):
                next_url({'link': f'<{target}>; rel="next"'}, url, 11)
        with self.assertRaises(ValueError):
            validate_url('https://api.github.com@evil.invalid/repos/a')

    def test_paginated_collection_and_resume_after_interruption(self):
        second_url = f'{API}/repositories/11/contributors?page=2&per_page=100'
        responses = [Response(metadata()), Response([user()], headers={'Link': f'<{second_url}>; rel="next"'}),
                     URLError('interrupted'), URLError('interrupted'), URLError('interrupted')]
        with patch('urllib.request.OpenerDirector.open', side_effect=responses), patch('collect.time.sleep'):
            with self.assertRaises(RuntimeError):
                collect(self.seeds, self.raw)
        state = read_json(self.raw / 'collection.json')
        self.assertEqual(state['repositories']['example/a']['status'], 'incomplete')
        with patch('urllib.request.OpenerDirector.open', return_value=Response([user(2, 'bob')])) as opened, patch('collect.time.sleep'):
            state = collect(self.seeds, self.raw)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(state['repositories']['example/a']['status'], 'complete')
        self.assertEqual(len(state['repositories']['example/a']['pages']), 2)

    def test_empty_repository_is_complete(self):
        with patch('urllib.request.OpenerDirector.open', side_effect=[Response(metadata()), Response(None, status=204)]), patch('collect.time.sleep'):
            state = collect(self.seeds, self.raw)
        page = read_json(self.raw / state['repositories']['example/a']['pages'][0])
        self.assertEqual(page['data'], [])
        self.assertEqual(state['repositories']['example/a']['status'], 'complete')

    def test_budget_and_offline_mode_do_not_request(self):
        client = Client(self.raw, budget=1)
        client.requests = 1
        with patch.object(client.opener, 'open') as opened:
            with self.assertRaises(RuntimeError):
                client.fetch(f'{API}/repos/example/a')
        opened.assert_not_called()
        with self.assertRaises(RuntimeError):
            Client(self.raw, offline=True).fetch(f'{API}/repos/example/a')

    def test_rate_limit_stops_without_long_sleep_or_caching_error(self):
        client = Client(self.raw)
        error = HTTPError(f'{API}/repos/example/a', 403, 'rate limit',
                          {'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '123'}, None)
        with patch.object(client.opener, 'open', side_effect=error), patch('collect.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'reset 123'):
                client.fetch(f'{API}/repos/example/a')
        sleep.assert_not_called()
        self.assertFalse(self.raw.exists())

    def test_transient_failure_retries_but_permanent_failure_does_not(self):
        client = Client(self.raw)
        error = HTTPError(f'{API}/repos/example/a', 503, 'unavailable', {}, None)
        with patch.object(client.opener, 'open', side_effect=[error, Response(metadata())]) as opened, patch('collect.time.sleep'):
            client.fetch(f'{API}/repos/example/a')
        self.assertEqual(opened.call_count, 2)
        client = Client(self.root / 'other')
        error = HTTPError(f'{API}/repos/example/a', 404, 'missing', {}, None)
        with patch.object(client.opener, 'open', side_effect=error) as opened:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 404'):
                client.fetch(f'{API}/repos/example/a')
        self.assertEqual(opened.call_count, 1)

    def test_prepare_filters_bots_and_duplicates_and_freezes_checksums(self):
        payload = [user(), user(), user(2, 'robot[bot]', account_type='Bot'),
                   {'type': 'Anonymous', 'contributions': 2}]
        with patch('urllib.request.OpenerDirector.open', side_effect=[Response(metadata()), Response(payload)]), patch('collect.time.sleep'):
            collect(self.seeds, self.raw)
        quality = prepare(self.seeds, self.raw, self.root / 'processed', self.root / 'results')
        self.assertEqual(quality['edges'], 1)
        self.assertEqual(quality['duplicate_edges'], 1)
        self.assertEqual(quality['bot_records_excluded'], 1)
        self.assertEqual(quality['anonymous_or_unidentified_excluded'], 1)
        manifest = read_json(self.root / 'results/manifest.json')
        self.assertEqual(manifest['processed_sha256']['interactions.csv'], sha256(self.root / 'processed/interactions.csv'))

    def test_prepare_rejects_missing_last_page_even_if_marked_complete(self):
        with patch('urllib.request.OpenerDirector.open', side_effect=[Response(metadata()), Response([user()])]), patch('collect.time.sleep'):
            collect(self.seeds, self.raw)
        state = read_json(self.raw / 'collection.json')
        page_path = self.raw / state['repositories']['example/a']['pages'][0]
        page = read_json(page_path)
        page['headers']['link'] = f'<{API}/repositories/11/contributors?page=2>; rel="next"'
        write_json(page_path, page)
        with self.assertRaisesRegex(ValueError, 'Missing contributor page'):
            prepare(self.seeds, self.raw, self.root / 'processed', self.root / 'results')
        self.assertFalse((self.root / 'processed/interactions.csv').exists())

    def test_analysis_rejects_changed_processed_snapshot(self):
        data = self.root / 'processed'
        results = self.root / 'results'
        data.mkdir()
        path = data / 'interactions.csv'
        path.write_text('original', encoding='utf-8')
        write_json(results / 'manifest.json', {'processed_sha256': {'interactions.csv': sha256(path)}})
        path.write_text('modified', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since preparation'):
            analyze(data, results)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.developers = [{'developer_id': i, 'login': f'user{i}'} for i in (1, 2, 3)]
        self.repositories = [{'repo_id': i, 'full_name': f'example/{i}'} for i in (1, 2)]
        self.edges = [{'developer_id': u, 'repo_id': r, 'contributions': n}
                      for u, r, n in ((1, 1, 3), (1, 2, 5), (2, 1, 2), (3, 2, 1))]
        self.graph = build_graph(self.developers, self.repositories, self.edges)

    def test_partitions_id_collisions_degrees_and_jaccard(self):
        self.assertEqual(len(self.graph), 5)
        self.assertEqual(self.graph.degree('developer:1'), 2)
        self.assertEqual(self.graph.nodes['repository:1']['bipartite'], 1)
        projection = repository_projection(self.graph)
        attrs = projection['repository:1']['repository:2']
        self.assertAlmostEqual(attrs['weight'], 1 / 3)
        self.assertAlmostEqual(attrs['distance'], 3)
        self.assertEqual(attrs['shared_contributors'], 1)

    def test_invalid_edges_and_counts_fail(self):
        for edges in (self.edges + [self.edges[0]], [{'developer_id': 9, 'repo_id': 1, 'contributions': 1}],
                      [{'developer_id': 1, 'repo_id': 1, 'contributions': 0}]):
            with self.assertRaises(ValueError):
                build_graph(self.developers, self.repositories, edges)

    def test_pagerank_and_directed_hits_have_expected_roles(self):
        binary, weighted, hubs, authorities = rankings(self.graph)
        self.assertAlmostEqual(sum(binary.values()), 1)
        self.assertAlmostEqual(sum(weighted.values()), 1)
        self.assertGreater(binary['developer:1'], binary['developer:2'])
        self.assertGreater(hubs['developer:1'], hubs['developer:2'])
        self.assertAlmostEqual(hubs['repository:1'], 0)
        self.assertAlmostEqual(authorities['developer:1'], 0)
        self.assertAlmostEqual(authorities['repository:1'], authorities['repository:2'])

    def test_girvan_newman_recovers_two_triangles(self):
        graph = nx.relabel_nodes(nx.barbell_graph(3, 0), str)
        nx.set_edge_attributes(graph, 1.0, 'weight')
        result = communities(graph)
        gn = next(run for run in result['runs'] if run['method'] == 'girvan_newman')
        self.assertEqual({frozenset(group) for group in gn['communities']},
                         {frozenset(('0', '1', '2')), frozenset(('3', '4', '5'))})
        self.assertGreater(gn['modularity'], 0)
        self.assertEqual(len(result['runs']), 4)

    def test_edgeless_projection_and_isolates(self):
        graph = nx.empty_graph(['a', 'b'])
        result = communities(graph)
        self.assertEqual(result['status'], 'no_projection_edges')
        self.assertEqual(result['isolates'], ['a', 'b'])
        self.assertEqual(result['runs'], [])

    def test_empty_participation_graph_has_uniform_pagerank_and_zero_hits(self):
        graph = build_graph([], self.repositories, [])
        binary, weighted, hubs, authorities = rankings(graph)
        self.assertEqual(binary, {'repository:1': 0.5, 'repository:2': 0.5})
        self.assertEqual(binary, weighted)
        self.assertEqual(sum(hubs.values()), 0)
        self.assertEqual(sum(authorities.values()), 0)

    def test_pair_agreement_is_invariant_to_community_labels(self):
        self.assertEqual(pair_agreement([['a', 'b'], ['c']], [['c'], ['b', 'a']]), 1)
        self.assertAlmostEqual(pair_agreement([['a', 'b'], ['c']], [['a'], ['b', 'c']]), 1 / 3)

    def test_overlap_matrices_match_known_graph_and_keep_isolated_repositories(self):
        self.graph.add_node('repository:3', bipartite=1, label='example/3')
        counts, similarities = overlap_matrices(self.graph, ['repository:1', 'repository:2', 'repository:3'])
        self.assertEqual(counts[0, 1], 1)
        self.assertAlmostEqual(similarities[0, 1], 1 / 3)
        self.assertEqual(counts[0, 2], 0)
        self.assertEqual(similarities[1, 2], 0)
        self.assertNotEqual(counts[0, 0], counts[0, 0])  # The self-overlap diagonal is masked with NaN.

    def test_visualizations_export_and_link_correctly_without_contribution_edges(self):
        graph = build_graph([], self.repositories, [])
        projection = repository_projection(graph)
        tables = {'developers': [], 'repositories': [
            {'node': node, 'label': graph.nodes[node]['label'], 'degree': 0,
             'rank_degree': i, 'rank_pagerank': i, 'rank_hits': i, 'rank_log_pagerank': i}
            for i, node in enumerate(graph, 1)]}
        comparisons = {partition: {name: {'spearman': None, 'top10_overlap': None}
                      for name in ('degree_vs_pagerank', 'pagerank_vs_hits', 'binary_vs_log_pagerank')}
                      for partition in ('developers', 'repositories')}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            additional_figures(graph, projection, tables, communities(projection), comparisons, output)
            for name in CAPTIONS:
                self.assertTrue((output / f'{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(ET.parse(output / f'{name}.svg').getroot().tag, '{http://www.w3.org/2000/svg}svg')
            html = (output / 'index.html').read_text(encoding='utf-8')
            for link in re.findall(r'(?:src|href)="([^"]+)"', html):
                self.assertTrue((output / link).is_file(), link)


if __name__ == '__main__':
    unittest.main()
