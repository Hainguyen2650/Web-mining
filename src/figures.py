"""Explanatory figures for the frozen network analysis; no data collection."""

from html import escape

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np

BLUE, ORANGE, GRAY = '#2563eb', '#c65d12', '#64748b'
CAPTIONS = {
    'repository_communities': ('Repository communities', 'All repositories; colors show Louvain seed 42 communities. Gray nodes have no shared contributors. Node area reflects contributor count; edge width reflects Jaccard similarity.'),
    'contributor_overlap': ('Shared contributor heatmaps', 'All repository pairs, ordered by community. The diagonal is intentionally masked. Left: Jaccard similarity with a square-root color scale. Right: shared contributor counts.'),
    'bipartite_contributors': ('Developer and repository relationships', 'Only the 12 developers with the broadest participation are selected; all repositories remain visible. Lines represent contribution presence, not commit volume.'),
    'ranking_comparison': ('Ranking methods compared', 'Top 10 nodes by binary PageRank in each partition. Cells give actual ranks; darker means a better rank. Each partition uses its own logarithmic color scale.'),
    'ranking_sensitivity': ('Ranking agreement and sensitivity', 'Spearman correlation and top-10 membership overlap compare degree, HITS, and contribution-weighted PageRank with binary PageRank. Undefined comparisons are marked N/A.'),
    'community_diagnostics': ('Community detection diagnostics', 'Girvan–Newman hierarchy scores and agreement across methods and seeds. Projection isolates are excluded. Pair agreement is unadjusted, not ground-truth accuracy.'),
}


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)


def repository_order(projection, partition_results):
    groups = partition_results['runs'][0]['communities'] if partition_results['runs'] else []
    ordered = [node for group in groups for node in sorted(group, key=lambda n: projection.nodes[n]['label'].lower())]
    return ordered + sorted(partition_results['isolates'], key=lambda n: projection.nodes[n]['label'].lower())


def overlap_matrices(graph, nodes):
    neighbors = [set(graph[node]) for node in nodes]
    counts = np.zeros((len(nodes), len(nodes)))
    similarities = np.zeros_like(counts)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            shared = len(neighbors[i] & neighbors[j])
            union = len(neighbors[i] | neighbors[j])
            counts[i, j] = shared
            similarities[i, j] = shared / union if union else 0
    np.fill_diagonal(counts, np.nan)
    np.fill_diagonal(similarities, np.nan)
    return counts, similarities


def community_map(graph, projection, partitions, output):
    nodes = repository_order(projection, partitions)
    groups = partitions['runs'][0]['communities'] if partitions['runs'] else []
    colors = [BLUE, ORANGE, '#0f766e', '#9333ea', '#be185d', '#854d0e']
    membership = {node: i for i, group in enumerate(groups) for node in group}
    active = projection.copy()
    active.remove_nodes_from(partitions['isolates'])
    positions = nx.spring_layout(active, seed=42, weight='weight', k=0.7, iterations=150)
    for i, node in enumerate(partitions['isolates']):
        positions[node] = np.array([(i - (len(partitions['isolates']) - 1) / 2) * 0.5, -1.5])
    fig, (axis, key) = plt.subplots(1, 2, figsize=(15, max(8, len(nodes) * 0.35)), width_ratios=[1.1, 1])
    fig.suptitle('Repository communities in the sampled ecosystem', fontsize=17, fontweight='bold')
    maximum = max((graph.degree(node) for node in nodes), default=1) or 1
    nx.draw_networkx_edges(projection, positions, ax=axis, edge_color=GRAY, alpha=0.32,
                           width=[0.5 + 5 * np.sqrt(attrs['weight']) for _, _, attrs in projection.edges(data=True)])
    nx.draw_networkx_nodes(projection, positions, nodelist=nodes, ax=axis,
                           node_size=[250 + 650 * graph.degree(node) / maximum for node in nodes],
                           node_color=[colors[membership[n] % len(colors)] if n in membership else GRAY for n in nodes],
                           edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(projection, positions, labels={n: str(i + 1) for i, n in enumerate(nodes)},
                            ax=axis, font_color='white', font_size=9, font_weight='bold')
    axis.set(xlim=(-1.35, 1.35), ylim=(-1.8, 1.3))
    axis.axis('off')
    key.axis('off')
    key.set_title(f'All {len(nodes)} repositories | {projection.number_of_edges()} shared-contributor links', loc='left', fontsize=11)
    for i, node in enumerate(nodes):
        color = colors[membership[node] % len(colors)] if node in membership else GRAY
        group = f'C{membership[node] + 1}' if node in membership else 'isolated'
        key.text(0, 0.98 - i * 0.9 / max(len(nodes), 1),
                 f'{i + 1:>2}. {projection.nodes[node]["label"]}  [{group}]',
                 color=color, va='top', fontsize=9, transform=key.transAxes)
    fig.text(0.05, 0.025, 'Louvain seed 42 | node area: identified contributors | edge width: Jaccard similarity\nGray isolates are retained; map distances are a layout, not measured collaboration distances.', fontsize=10, color=GRAY)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, output, 'repository_communities')


def overlap_heatmap(graph, projection, partitions, output):
    nodes = repository_order(projection, partitions)
    counts, similarities = overlap_matrices(graph, nodes)
    labels = [f'{i + 1}. {projection.nodes[node]["label"].split("/")[-1]}' for i, node in enumerate(nodes)]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(nodes) * 0.4)), layout='constrained')
    fig.suptitle('Which repositories share contributors?', fontsize=17, fontweight='bold')
    cmap = plt.get_cmap('Blues').with_extremes(bad='#e2e8f0')
    for axis, matrix, title in zip(axes, (similarities, counts), ('Jaccard similarity', 'Shared contributors')):
        values = matrix[np.isfinite(matrix)]
        maximum = max(float(values.max()) if values.size else 0, 0.001)
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=maximum) if title == 'Jaccard similarity' else None
        image = axis.imshow(matrix, cmap=cmap, norm=norm, **({} if norm else {'vmin': 0, 'vmax': max(1, maximum)}))
        axis.set_xticks(range(len(nodes)), range(1, len(nodes) + 1), fontsize=8)
        axis.set_yticks(range(len(nodes)), labels, fontsize=8)
        axis.set(title=title, xlabel='Repository number; same order as rows')
        fig.colorbar(image, ax=axis, shrink=0.7, label=title)
        if title == 'Shared contributors':
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    value = matrix[i, j]
                    if np.isfinite(value) and value > 0:
                        axis.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=7,
                                  color='white' if value > maximum / 2 else '#0f172a')
    save(fig, output, 'contributor_overlap')


def bipartite_view(graph, tables, output):
    developers = sorted(tables['developers'], key=lambda row: row['rank_degree'])[:12]
    repositories = sorted(tables['repositories'], key=lambda row: row['label'].lower())
    positions = {row['node']: (x, y) for x, rows in ((0, developers), (1, repositories))
                 for row, y in zip(rows, np.linspace(1, 0, len(rows)))}
    selected = graph.subgraph(positions)
    fig, axis = plt.subplots(figsize=(15, 10))
    fig.suptitle('Developers connecting multiple sampled repositories', fontsize=17, fontweight='bold')
    nx.draw_networkx_edges(selected, positions, ax=axis, edge_color=GRAY, alpha=0.35, width=1.2,
                           edgelist=sorted(tuple(sorted(edge)) for edge in selected.edges))
    for x, rows, color in ((0, developers, BLUE), (1, repositories, ORANGE)):
        nx.draw_networkx_nodes(selected, positions, nodelist=[row['node'] for row in rows],
                               node_color=color, node_size=80, ax=axis)
        for row in rows:
            y = positions[row['node']][1]
            label = row['label'] + (f" ({row['degree']} repos)" if x == 0 else '')
            axis.text(x + (-0.03 if x == 0 else 0.03), y, label,
                      ha='right' if x == 0 else 'left', va='center', fontsize=10)
    axis.text(0, 1.07, f'Top {len(developers)} developers by repository breadth', ha='center', color=BLUE)
    axis.text(1, 1.07, f'All {len(repositories)} sampled repositories', ha='center', color=ORANGE)
    axis.set(xlim=(-0.65, 2), ylim=(-0.07, 1.12))
    axis.axis('off')
    fig.text(0.07, 0.025, f"Selected view: {len(developers)} of {len(tables['developers'])} developers; "
             f'{selected.number_of_edges()} of {graph.number_of_edges()} contribution links. Lines show presence, not contribution volume.', color=GRAY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, output, 'bipartite_contributors')


def ranking_heatmap(tables, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
    fig.suptitle('How do ranking methods change the leading nodes?\nTop 10 by binary PageRank; numbers are within-partition ranks', fontsize=15, fontweight='bold')
    fig.supxlabel('Color uses a separate logarithmic rank scale in each partition.', fontsize=10, color=GRAY)
    columns = ['rank_degree', 'rank_pagerank', 'rank_hits', 'rank_log_pagerank']
    for name, axis in zip(('developers', 'repositories'), axes):
        rows = tables[name][:10]
        if not rows:
            axis.text(0.5, 0.5, 'No nodes in this partition', ha='center')
            axis.axis('off')
            continue
        matrix = np.array([[row[key] for key in columns] for row in rows])
        norm = LogNorm(vmin=1, vmax=max(2, len(tables[name])))
        axis.imshow(matrix, cmap='Blues_r', norm=norm, aspect='auto')
        axis.set_xticks(range(4), ['Degree', 'PageRank', 'HITS', 'Log-weight\nPageRank'])
        axis.set_yticks(range(len(rows)), [row['label'] for row in rows], fontsize=9)
        axis.set_title(name.title() + ' | darker = better rank')
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                axis.text(j, i, str(value), ha='center', va='center',
                          color='white' if norm(value) < 0.45 else '#0f172a', fontweight='bold')
    save(fig, output, 'ranking_comparison')


def sensitivity_plot(comparisons, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    fig.suptitle('Ranking agreement across methods and edge weights', fontsize=16, fontweight='bold')
    keys = ['degree_vs_pagerank', 'pagerank_vs_hits', 'binary_vs_log_pagerank']
    labels = ['Degree vs\nPageRank', 'PageRank vs\nHITS', 'Binary vs log-weight\nPageRank']
    for axis, metric, title in zip(axes, ('spearman', 'top10_overlap'), ('Spearman rank correlation', 'Top-10 membership overlap')):
        for offset, name, color in ((-0.18, 'developers', BLUE), (0.18, 'repositories', ORANGE)):
            values = [comparisons[name][key][metric] for key in keys]
            bars = axis.bar(np.arange(3) + offset, [value or 0 for value in values], width=0.34, color=color, label=name.title())
            axis.bar_label(bars, labels=['N/A' if value is None else format(value, '.4f' if metric == 'spearman' else '.2f')
                                        for value in values], padding=4, fontsize=9)
        axis.set_xticks(range(3), labels, fontsize=9)
        axis.set(title=title, ylim=(-1.12 if metric == 'spearman' else 0, 1.17))
        axis.axhline(0, color=GRAY, linewidth=0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
    save(fig, output, 'ranking_sensitivity')


def community_diagnostics(partitions, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    fig.suptitle('Community detection: partition quality and agreement', fontsize=16, fontweight='bold')
    runs, hierarchy = partitions['runs'], partitions['girvan_newman_hierarchy']
    if not runs:
        for axis in axes:
            axis.text(0.5, 0.5, 'No shared-contributor edges\nCommunity comparison is undefined', ha='center', va='center')
            axis.axis('off')
    else:
        axes[0].plot([row['communities'] for row in hierarchy], [row['modularity'] for row in hierarchy], marker='o', color=BLUE)
        best = max(hierarchy, key=lambda row: row['modularity'])
        axes[0].scatter(best['communities'], best['modularity'], color=ORANGE, s=95, zorder=3)
        axes[0].set(title=f"Girvan–Newman | best Q = {best['modularity']:.3f}",
                    xlabel='Communities in the hierarchy', ylabel='Jaccard-weighted modularity')
        axes[0].grid(alpha=0.2)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        names = [f"{run['method']}:{run['seed']}" for run in runs]
        agreement = np.eye(len(runs))
        for row in partitions['agreement']:
            i, j = names.index(row['first']), names.index(row['second'])
            agreement[i, j] = agreement[j, i] = row['pair_agreement'] if row['pair_agreement'] is not None else np.nan
        labels = [f"Louvain {run['seed']}" if run['method'] == 'louvain' else 'Girvan–Newman' for run in runs]
        axes[1].imshow(agreement, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_xticks(range(len(runs)), labels, rotation=25, ha='right', fontsize=9)
        axes[1].set_yticks(range(len(runs)), labels, fontsize=9)
        axes[1].set_title('Unadjusted node-pair agreement | isolates excluded')
        for i in range(len(runs)):
            for j in range(len(runs)):
                value = agreement[i, j]
                axes[1].text(j, i, f'{value:.2f}' if np.isfinite(value) else 'N/A', ha='center', va='center',
                             color='white' if value > 0.5 else '#0f172a')
    save(fig, output, 'community_diagnostics')


def additional_figures(graph, projection, tables, partitions, comparisons, output):
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'svg.hashsalt': 'web-mining', 'axes.spines.top': False, 'axes.spines.right': False}):
        community_map(graph, projection, partitions, output)
        overlap_heatmap(graph, projection, partitions, output)
        bipartite_view(graph, tables, output)
        ranking_heatmap(tables, output)
        sensitivity_plot(comparisons, output)
        community_diagnostics(partitions, output)
    cards = []
    all_figures = {**CAPTIONS,
                   'degree_distributions': ('Degree distributions', 'Complete sampled bipartite graph; both node partitions are shown.'),
                   'component_sizes': ('Connected components', 'Component sizes in the complete sampled bipartite graph.'),
                   'pagerank_top10': ('PageRank leaders', 'The ten highest PageRank nodes in each partition.')}
    for name, (title, caption) in all_figures.items():
        if not (output / f'{name}.png').exists():
            continue
        svg = f' · <a href="{name}.svg">SVG</a>' if name in CAPTIONS else ''
        cards.append(f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{escape(caption)}" loading="lazy"></a>'
                     f'<p><a href="{name}.png">PNG</a>{svg}</p></section>')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>GitHub network figures</title>'
        '<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#172033;background:#f4f7fb}'
        'section{background:white;padding:24px;margin:24px 0;border:1px solid #dbe3ef;border-radius:10px}img{width:100%;height:auto}'
        'a{color:#174ea6}h1,h2{line-height:1.2}</style><h1>GitHub contribution network figures</h1>'
        '<p>Frozen sample analysis. Open a PNG at full resolution or use SVG for slides. '
        'Selected subgraphs and internal community metrics are labeled in each figure.</p>'
        + ''.join(cards) + '</html>\n', encoding='utf-8')
