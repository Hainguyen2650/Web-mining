"""Reproduce rankings and repository communities from frozen contribution tables."""

import argparse
from collections import Counter
from itertools import combinations
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr

from common import ROOT, positive_int, read_csv, read_json, sha256, write_csv, write_json
from figures import additional_figures

PARAMETERS = {'pagerank_alpha': 0.85, 'pagerank_tol': 1e-12, 'max_iter': 1000,
              'louvain_seeds': [42, 43, 44], 'louvain_resolution': 1,
              'projection': 'Jaccard of binary contributor neighborhoods',
              'girvan_newman': 'unweighted edge betweenness; full hierarchy; best Jaccard-weighted modularity',
              'bridge_distance': 'inverse Jaccard', 'hits_direction': 'developer -> repository',
              'figure_layout_seed': 42}


def build_graph(developers, repositories, interactions):
    graph = nx.Graph()
    for rows, field, prefix, partition, label in (
            (developers, 'developer_id', 'developer', 0, 'login'),
            (repositories, 'repo_id', 'repository', 1, 'full_name')):
        for row in sorted(rows, key=lambda item: positive_int(item[field], field)):
            identifier = positive_int(row[field], field)
            key = f'{prefix}:{identifier}'
            if key in graph or not row.get(label):
                raise ValueError(f'Duplicate node or missing label: {key}')
            graph.add_node(key, bipartite=partition, label=row[label])
    for row in interactions:
        u = f"developer:{positive_int(row['developer_id'], 'developer_id')}"
        r = f"repository:{positive_int(row['repo_id'], 'repo_id')}"
        count = positive_int(row['contributions'], 'contributions')
        if u not in graph or r not in graph or graph.has_edge(u, r):
            raise ValueError('Dangling reference or duplicate edge')
        graph.add_edge(u, r, contributions=count, log_weight=math.log1p(count))
    if not nx.is_bipartite(graph):
        raise ValueError('Invalid bipartite graph')
    return graph


def repository_projection(graph):
    projection = nx.Graph()
    repositories = [node for node, attrs in graph.nodes(data=True) if attrs['bipartite'] == 1]
    projection.add_nodes_from((node, dict(graph.nodes[node])) for node in repositories)
    neighbors = {node: set(graph[node]) for node in repositories}
    for a, b in combinations(repositories, 2):
        shared = len(neighbors[a] & neighbors[b])
        if shared:
            similarity = shared / len(neighbors[a] | neighbors[b])
            projection.add_edge(a, b, weight=similarity, distance=1 / similarity, shared_contributors=shared)
    return projection


def rankings(graph):
    if not graph.number_of_edges():
        zeros = dict.fromkeys(graph, 0.0)
        uniform = dict.fromkeys(graph, 1 / len(graph)) if graph else {}
        return uniform, uniform.copy(), zeros, zeros.copy()
    kwargs = {'alpha': PARAMETERS['pagerank_alpha'], 'tol': PARAMETERS['pagerank_tol'],
              'max_iter': PARAMETERS['max_iter']}
    binary = nx.pagerank(graph, weight=None, **kwargs)
    weighted = nx.pagerank(graph, weight='log_weight', **kwargs)
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes(data=True))
    directed.add_edges_from((u, v) if graph.nodes[u]['bipartite'] == 0 else (v, u)
                            for u, v in graph.edges)
    hubs, authorities = nx.hits(directed, nstart=dict.fromkeys(directed, 1.0),
                                max_iter=1000, tol=1e-12)
    return binary, weighted, hubs, authorities


def canonical_partition(groups):
    return sorted((sorted(group) for group in groups), key=lambda group: (-len(group), group))


def pair_agreement(first, second):
    labels_a = {node: i for i, group in enumerate(first) for node in group}
    labels_b = {node: i for i, group in enumerate(second) for node in group}
    if labels_a.keys() != labels_b.keys():
        raise ValueError('Partitions must cover the same nodes')
    pairs = list(combinations(sorted(labels_a), 2))
    if not pairs:
        return None
    return sum((labels_a[a] == labels_a[b]) == (labels_b[a] == labels_b[b]) for a, b in pairs) / len(pairs)


def communities(projection):
    isolates = sorted(nx.isolates(projection))
    active = projection.subgraph(sorted(set(projection) - set(isolates))).copy()
    if not active.number_of_edges():
        return {'status': 'no_projection_edges', 'isolates': isolates, 'runs': [],
                'girvan_newman_hierarchy': [], 'agreement': []}
    runs = []
    for seed in PARAMETERS['louvain_seeds']:
        groups = canonical_partition(nx.community.louvain_communities(active, weight='weight', seed=seed, resolution=1))
        runs.append({'method': 'louvain', 'seed': seed, 'communities': groups,
                     'modularity': nx.community.modularity(active, groups, weight='weight')})
    groups = canonical_partition(nx.connected_components(active))
    best, best_score = groups, nx.community.modularity(active, groups, weight='weight')
    hierarchy = [{'communities': len(groups), 'modularity': best_score}]
    for partition in nx.community.girvan_newman(active):
        groups = canonical_partition(partition)
        score = nx.community.modularity(active, groups, weight='weight')
        hierarchy.append({'communities': len(groups), 'modularity': score})
        if score > best_score:
            best, best_score = groups, score
    runs.append({'method': 'girvan_newman', 'seed': None, 'communities': best, 'modularity': best_score})
    agreement = [{'first': f"{a['method']}:{a['seed']}", 'second': f"{b['method']}:{b['seed']}",
                  'pair_agreement': pair_agreement(a['communities'], b['communities'])}
                 for a, b in combinations(runs, 2)]
    return {'status': 'complete', 'isolates': isolates, 'runs': runs,
            'girvan_newman_hierarchy': hierarchy, 'agreement': agreement}


def score_ranks(nodes, scores):
    return {node: rank for rank, node in enumerate(sorted(nodes, key=lambda node: (-scores[node], node)), 1)}


def rank_agreement(nodes, a, b):
    if not nodes:
        return {'spearman': None, 'top10_overlap': None}
    k = min(10, len(nodes))
    first, second = score_ranks(nodes, a), score_ranks(nodes, b)
    rho = None
    if len(set(a[node] for node in nodes)) > 1 and len(set(b[node] for node in nodes)) > 1:
        rho = float(spearmanr([a[node] for node in nodes], [b[node] for node in nodes]).statistic)
    return {'spearman': rho, 'top10_overlap': len({n for n in nodes if first[n] <= k}
                                               & {n for n in nodes if second[n] <= k}) / k}


def basic_figures(graph, developer_rows, repository_rows, component_sizes, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for partition, title, axis in zip((0, 1), ('Developers', 'Repositories'), axes):
        degrees = Counter(graph.degree(node) for node in graph if graph.nodes[node]['bipartite'] == partition)
        axis.bar(list(degrees), list(degrees.values()))
        axis.set(title=f'{title} in sampled contribution graph', xlabel='Degree', ylabel='Nodes')
        axis.set_yscale('log')
    fig.tight_layout()
    fig.savefig(output / 'degree_distributions.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(range(1, len(component_sizes) + 1), component_sizes)
    axis.set(title='Connected components of sampled contribution graph', xlabel='Component by size', ylabel='Nodes')
    axis.set_yscale('log')
    fig.tight_layout()
    fig.savefig(output / 'component_sizes.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for rows, axis, title in zip((developer_rows, repository_rows), axes, ('Developers', 'Repositories')):
        top = rows[:10][::-1]
        axis.barh([row['label'] for row in top], [row['pagerank'] for row in top])
        axis.set(title=f'Top {len(top)} {title.lower()} by binary PageRank', xlabel='PageRank')
    fig.tight_layout()
    fig.savefig(output / 'pagerank_top10.png', dpi=160)
    plt.close(fig)


def analyze(data_dir, results_dir):
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    manifest = read_json(results_dir / 'manifest.json')
    for name, checksum in manifest['processed_sha256'].items():
        if sha256(data_dir / name) != checksum:
            raise ValueError(f'Processed input changed since preparation: {name}')
    graph = build_graph(read_csv(data_dir / 'developers.csv'), read_csv(data_dir / 'repositories.csv'),
                        read_csv(data_dir / 'interactions.csv'))
    if not graph.number_of_nodes():
        raise ValueError('Cannot analyze an empty graph')
    projection = repository_projection(graph)
    binary, weighted, hubs, authorities = rankings(graph)
    degree = dict(graph.degree())
    bridge = nx.betweenness_centrality(projection, weight='distance') if projection.number_of_edges() else dict.fromkeys(projection, 0.0)
    partition_results = communities(projection)
    component_sizes = sorted((len(group) for group in nx.connected_components(graph)), reverse=True)
    tables, comparisons = {}, {}
    for partition, name, hits_scores in ((0, 'developers', hubs), (1, 'repositories', authorities)):
        nodes = [node for node in graph if graph.nodes[node]['bipartite'] == partition]
        ordered = sorted(nodes, key=lambda node: (-binary[node], node))
        ranks = {method: score_ranks(nodes, scores) for method, scores in
                 [('degree', degree), ('pagerank', binary), ('hits', hits_scores), ('log_pagerank', weighted)]}
        rows = []
        for node in ordered:
            rows.append({'node': node, 'label': graph.nodes[node]['label'], 'degree': degree[node],
                         'contributions': sum(graph[node][neighbor]['contributions'] for neighbor in graph[node]),
                         'pagerank': binary[node], 'log_pagerank': weighted[node], 'hits': hits_scores[node],
                         'bridge_betweenness': bridge.get(node, 0.0),
                         **{f'rank_{method}': scores[node] for method, scores in ranks.items()}})
        tables[name] = rows
        write_csv(results_dir / f'{name}_rankings.csv', rows,
                  ['node', 'label', 'degree', 'contributions', 'pagerank', 'log_pagerank', 'hits',
                   'bridge_betweenness', 'rank_degree', 'rank_pagerank', 'rank_hits', 'rank_log_pagerank'])
        comparisons[name] = {'degree_vs_pagerank': rank_agreement(nodes, degree, binary),
                             'pagerank_vs_hits': rank_agreement(nodes, binary, hits_scores),
                             'binary_vs_log_pagerank': rank_agreement(nodes, binary, weighted)}
    assignments = []
    for run in partition_results['runs']:
        for index, group in enumerate(run['communities']):
            assignments.extend({'method': run['method'], 'seed': run['seed'], 'community': index,
                                'node': node, 'repository': graph.nodes[node]['label'], 'isolated': False}
                               for node in group)
        assignments.extend({'method': run['method'], 'seed': run['seed'], 'community': '', 'node': node,
                            'repository': graph.nodes[node]['label'], 'isolated': True}
                           for node in partition_results['isolates'])
    if not partition_results['runs']:
        assignments = [{'method': 'none', 'seed': '', 'community': '', 'node': node,
                        'repository': graph.nodes[node]['label'], 'isolated': True} for node in projection]
    write_csv(results_dir / 'community_assignments.csv', assignments,
              ['method', 'seed', 'community', 'node', 'repository', 'isolated'])
    write_json(results_dir / 'communities.json', partition_results)
    write_json(results_dir / 'ranking_comparisons.json', comparisons)
    nu, nr = len(tables['developers']), len(tables['repositories'])
    eligible = sum(degree[node] >= 2 and any(degree[r] >= 2 for r in graph[node])
                   for node in graph if graph.nodes[node]['bipartite'] == 0)
    summary = {'developers': nu, 'repositories': nr, 'edges': graph.number_of_edges(),
               'bipartite_density': graph.number_of_edges() / (nu * nr) if nu and nr else 0,
               'components': len(component_sizes), 'component_sizes': component_sizes,
               'isolates': sorted(nx.isolates(graph)), 'projection_edges': projection.number_of_edges(),
               'projection_isolates': partition_results['isolates'],
               'potential_holdout_developers': eligible,
               'eligibility_note': 'Upper bound before sequential edge holdout; no recommendation experiment run'}
    write_json(results_dir / 'summary.json', summary)
    write_json(results_dir / 'analysis_parameters.json', PARAMETERS)
    write_csv(results_dir / 'repository_projection.csv',
              [{'source': a, 'target': b, **attrs} for a, b, attrs in projection.edges(data=True)],
              ['source', 'target', 'weight', 'distance', 'shared_contributors'])
    nx.write_graphml(graph, results_dir / 'bipartite.graphml')
    nx.write_graphml(projection, results_dir / 'repository_projection.graphml')
    figures = results_dir / 'figures'
    figures.mkdir(exist_ok=True)
    basic_figures(graph, tables['developers'], tables['repositories'], component_sizes, figures)
    additional_figures(graph, projection, tables, partition_results, comparisons, figures)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    try:
        print(analyze(args.data_dir, args.results_dir))
    except (ValueError, OSError, KeyError, nx.NetworkXException) as error:
        parser.exit(1, f'Analysis stopped: {error}\n')


if __name__ == '__main__':
    main()
