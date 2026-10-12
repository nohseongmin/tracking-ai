"""Plot the published 150-model results for the English README."""

import argparse
import logging
import math
from pathlib import Path

from build_model_graphs import ARCHITECTURES, COLORS, MARKERS, load_results

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('csv', 'export', 'planned', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rows, best = load_results(args)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
    figure, axis = plt.subplots(figsize=(15, 8), facecolor='white')
    figure.subplots_adjust(left=0.085, right=0.975, top=0.77, bottom=0.26)
    legend = []
    for architecture in ARCHITECTURES:
        group = [row for row in rows if row['architecture'] == architecture]
        axis.scatter([row['condition'] for row in group],
                     [row['accuracy_percent'] for row in group], s=65,
                     marker=MARKERS[architecture], color=COLORS[architecture],
                     alpha=0.8, zorder=3)
        selected = best[architecture]
        axis.scatter(selected['condition'], selected['accuracy_percent'], s=380,
                     marker='*', color=COLORS[architecture], edgecolor='white',
                     linewidth=1, zorder=4)
        legend.append(Line2D([], [], linestyle='none', marker=MARKERS[architecture],
                             color=COLORS[architecture], markersize=9,
                             label=f"{architecture}: selected {selected['accuracy_percent']:.2f}%"))
    axis.set(xlim=(0, 51),
             ylim=(math.floor(min(row['accuracy_percent'] for row in rows)) - 0.5,
                   math.ceil(max(row['accuracy_percent'] for row in rows)) + 0.5),
             ylabel='Test accuracy (%)')
    axis.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    axis.set_xlabel('Configuration number (paired across GRU, LSTM, and MLP)', labelpad=14)
    axis.grid(axis='y', color='#E3E6EB', linewidth=0.8, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color('#B8BEC6')
    figure.suptitle('150 models, one held-out test split', x=0.085, y=0.955,
                    ha='left', fontsize=25, fontweight='bold')
    figure.text(0.085, 0.897, 'Three-second position forecasting from six movement observations',
                fontsize=15, color='#434B57')
    figure.legend(handles=legend, loc='upper left', bbox_to_anchor=(0.076, 0.867),
                  ncol=3, frameon=False, fontsize=13)
    figure.text(0.085, 0.135, 'Point = trained model  |  Star = lowest validation mean distance within architecture',
                fontsize=12)
    figure.text(0.085, 0.084, 'Success: 3D position error <= 250 CS:GO units  |  Test samples: 6,107,209',
                fontsize=12, color='#434B57')
    figure.text(0.085, 0.035, 'Source: ESTA (github.com/pnxenopoulos/esta), CC BY-SA 4.0; recorded AWS experiment results',
                fontsize=11, color='#6B7280')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor='white')
    plt.close(figure)
    logging.info('Validated all 150 results and saved %s', args.output)


if __name__ == '__main__':
    main()
