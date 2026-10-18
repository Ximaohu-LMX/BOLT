#!/usr/bin/env python3
"""Render Figures 3–5 at their final print sizes from paper_v3 results.

Usage: python3 paper_v3/scripts/render_layout_figures.py [--preview-dir DIR]
No training, statistics, historical assets, or Figures 1–2 are changed.
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/paperv3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATASETS = ['AUTHOR', 'DBLP', 'IMDB', 'WIKI']
BUDGETS = ['A100', 'A75', 'A50', 'A25', 'A10', 'A1', 'A0']
COLORS = {
    'CLIQUE': '#264653', 'UB_LB': '#6c757d', 'UB_ORDER': '#2a9d8f',
    'BOLT': '#e76f51', 'LB_ORDER': '#b07aa1', 'NO_BOUND': '#aeb7bf', 'BOLT_L0': '#6c757d',
}
def display_method(method):
    from repro.names import display_method as paper_label
    return r'BOLT ($\lambda=0$)' if method == 'BOLT_L0' else paper_label(method)


def display_ablation(method):
    from repro.names import display_method as paper_label
    return r'BOLT ($\lambda=0$)' if method == 'BOLT_L0' else paper_label(method, ablation=True)


WIDE_SIZE = (7.0, 2.25)
COLUMN_SIZE = (3.33, 2.95)
# Shared final-size typography and colors for all three empirical figures.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .6, 'lines.linewidth': 1.0, 'lines.markersize': 3.3,
    'axes.grid': True, 'grid.alpha': .3, 'grid.linestyle': '--',
    'grid.linewidth': .5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 120, 'savefig.bbox': None,
})


def read(name):
    with (PAPER / 'results' / (name + '.csv')).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    aliases = {'L40': 'A25', 'L60': 'A50', 'L80': 'A75'}
    for row in rows:
        row['budget'] = aliases.get(row.get('budget'), row.get('budget'))
    return rows


def select(rows, dataset, budget, method):
    matches = [r for r in rows if (r['dataset'], r['budget'], r['method'], r['seed'])
               == (dataset, budget, method, '0')]
    assert len(matches) == 1, (dataset, budget, method, len(matches))
    return matches[0]


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='both', pad=2, length=2.5)


def wide_panels():
    fig, axes = plt.subplots(1, 4, figsize=WIDE_SIZE)
    # Identical canvas, plot height and legend position for Figures 3 and 4.
    fig.subplots_adjust(left=.075, right=.99, bottom=.22, top=.765, wspace=.40)
    for ax, dataset in zip(axes, DATASETS):
        clean(ax)
        ax.set_title(dataset, pad=5)
    return fig, axes



def compact_xlabel(fig, axes, label):
    """Keep the shared title four points below the lowest x tick label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_bottom = min(tick.get_window_extent(renderer).y0
                      for ax in axes for tick in ax.get_xticklabels()
                      if tick.get_visible())
    title_top = (tick_bottom - 4 * fig.dpi / 72) / fig.bbox.height
    fig.supxlabel(label, y=title_top, va='top', fontsize=8)


def legend(fig, handles, labels=None, columns=5, columnspacing=1.5):
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    return fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995),
                      ncol=columns, frameon=False, columnspacing=columnspacing,
                      handlelength=1.5, handletextpad=.5, borderaxespad=0)


def save(fig, name, preview_dir):
    fig.savefig(PAPER / 'figs' / (name + '.pdf'),
                metadata={'CreationDate': None, 'ModDate': None})
    if preview_dir:
        fig.savefig(preview_dir / (name + '.png'), dpi=200)
    # Refresh the already-versioned F4 preview alongside its vector asset.
    if name == 'F4_absolute_ablations':
        fig.savefig(PAPER / 'figs' / (name + '.png'), dpi=200)
    plt.close(fig)


def accuracy(core, supplement, preview_dir):
    fig, axes = wide_panels()
    for ax, dataset in zip(axes, DATASETS):
        ax.set_yscale('log')
        ax.set_ylim(1, 10000)
        ax.set_yticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1,000', '10,000'])
        ax.minorticks_off()
        ax.set_xlim(-.55, len(BUDGETS)-.45)
        ax.set_xticks(range(len(BUDGETS)), BUDGETS, rotation=45, ha='right')
        for mi, method in enumerate(['CLIQUE', 'BOLT']):
            color = COLORS[method]
            fill = tuple(.62 + .38*c for c in to_rgb(color))
            for bi, budget in enumerate(BUDGETS):
                rows = supplement if budget in ['A25', 'A50', 'A75'] else core
                row = select(rows, dataset, budget, method)
                mean, q50, q99, q100 = [float(row[k]) for k in ['q_avg', 'q50', 'q99', 'q100']]
                assert 1 <= q50 <= q99 <= q100 <= 10000
                assert 1 <= mean <= q100
                x, width = bi+(mi-.5)*.34, .25
                ax.add_patch(Rectangle((x-width/2, q50), width, q99-q50,
                                       facecolor=fill, edgecolor=color, linewidth=.65, zorder=2))
                ax.vlines(x, q99, q100, color=color, lw=.8, zorder=3)
                ax.hlines(q100, x-width*.4, x+width*.4, color=color, lw=.9, zorder=3)
                ax.plot(x, mean, color=color, linestyle='none', marker='o',
                        markersize=3.0, markeredgewidth=0, zorder=5)
        # Read the reference directly from the existing full-label result.
        reference = float(select(core, dataset, 'A100', 'CLIQUE')['q_avg'])
        ax.axhline(reference, color='#777777', linestyle=(0, (3, 2)),
                   linewidth=.8, zorder=4)
    axes[0].set_ylabel('Q-error (log scale)')
    handles = [Patch(facecolor=tuple(.62+.38*c for c in to_rgb(COLORS[m])),
                     edgecolor=COLORS[m], label=display_method(m)) for m in ['CLIQUE', 'BOLT']]
    handles += [Line2D([], [], color='#444444', marker='o', linestyle='none',
                       markeredgewidth=0, label='Mean (dot)'),
                Patch(facecolor='#cccccc', edgecolor='#555555', label='q50–q99 (box)'),
                Line2D([], [], color='#555555', marker='_', markersize=7,
                       linestyle='none', label='q100 (whisker + cap)')]
    handles.append(Line2D([], [], color='#777777', linestyle=(0, (3, 2)),
                          linewidth=.8, label='CLIQUE–A100 mean'))
    key = legend(fig, handles, columns=6, columnspacing=.75)
    fig.canvas.draw()
    assert fig.bbox.contains(*key.get_window_extent().p0)
    assert fig.bbox.contains(*key.get_window_extent().p1)
    compact_xlabel(fig, axes, 'Label-revelation setting (decreasing labels)')
    save(fig, 'F3A_quantile_ranges', preview_dir)


def ablations(core, preview_dir):
    fig, axes = wide_panels()
    methods = ['BOLT', 'CLIQUE', 'UB_LB', 'UB_ORDER', 'LB_ORDER', 'NO_BOUND']
    budgets = ['A0', 'A10', 'A100']
    centers, width = np.arange(3), .123
    for ax, dataset in zip(axes, DATASETS):
        for mi, method in enumerate(methods):
            values = [float(select(core, dataset, b, method)['q_avg']) for b in budgets]
            ax.bar(centers+(mi-2.5)*width, values, width=width*.92,
                   color=COLORS[method], edgecolor=COLORS['CLIQUE'], linewidth=.35,
                   hatch='///' if method == 'NO_BOUND' else None, label=display_ablation(method))
        ax.set_xticks(centers, budgets)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].set_ylabel('Mean Q-error')
    handles, labels = axes[0].get_legend_handles_labels()
    key = legend(fig, handles, labels, columns=6)
    compact_xlabel(fig, axes, 'Label-revelation setting')
    save(fig, 'F4_absolute_ablations', preview_dir)


def matched_loss(core, preview_dir):
    pairs = {}
    for dataset in DATASETS:
        off, on = [select(core, dataset, 'A0', m) for m in ['BOLT_L0', 'BOLT']]
        for field in ['seed', 'label_mask_id', 'test_sha256', 'revealed_unique',
                      'output_convention', 'checkpoint_selection', 'chain_ratio',
                      'protocol', 'lower_bound_mode']:
            assert off[field] == on[field], (dataset, field)
        assert float(off['lambda_order']) == 0 and float(on['lambda_order']) > 0
        pairs[dataset] = [off, on]
    fig, axes = plt.subplots(2, 1, figsize=COLUMN_SIZE)
    # Preserve the panel heights while reducing inter-panel and outer whitespace.
    fig.subplots_adjust(left=.16, right=.99, bottom=.08, top=.87, hspace=.34)
    centers = np.arange(len(DATASETS))
    for ax, metric, title, limit in zip(axes, ['q_avg', 'q99'],
                                       ['Mean Q-error', 'q99 Q-error'], [10.5, 165]):
        clean(ax)
        for mi, method in enumerate(['BOLT_L0', 'BOLT']):
            values = [float(pairs[d][mi][metric]) for d in DATASETS]
            bars = ax.bar(centers+(mi-.5)*.36, values, width=.32,
                          color=COLORS[method], edgecolor='none',
                          label=display_ablation(method) if mi == 0 else 'BOLT: loss on')
            # Stagger close labels vertically; every value retains two decimals.
            for di, (bar, value) in enumerate(zip(bars, values)):
                other = float(pairs[DATASETS[di]][1-mi][metric])
                padding = 10 if mi == 1 and abs(value-other) < limit*.12 else 3
                ax.annotate(f'{value:.2f}', (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, padding), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7)
        ax.set_title(title, pad=5)
        ax.set_ylabel('Q-error')
        ax.set_xticks(centers, DATASETS)
        ax.set_ylim(0, limit)
        ax.set_yticks([0, 2, 4, 6, 8, 10] if metric == 'q_avg' else [0, 30, 60, 90, 120, 150])
    handles, labels = axes[0].get_legend_handles_labels()
    key = legend(fig, handles, labels, columns=2)
    save(fig, 'F5_matched_loss_bars', preview_dir)


def main(preview_dir=None):
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
    core = [r for r in read('run_metrics') if r['family'] in ['core', 'control']]
    accuracy(core, read('supplement_metrics'), preview_dir)
    ablations(core, preview_dir)
    matched_loss(core, preview_dir)
    print(f'Rendered F3/F4: {WIDE_SIZE} in; F5: {COLUMN_SIZE} in; verified BOLT metric CSVs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    main(parser.parse_args().preview_dir)
