import argparse
import collections
import csv
import hashlib
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import font_manager


ARCHITECTURES = ('GRU', 'LSTM', 'MLP')
PARAMETERS = ('hidden_size', 'layers', 'batch_size', 'learning_rate', 'dropout', 'weight_decay')
INTEGER_FIELDS = ('hidden_size', 'layers', 'batch_size', 'epochs_completed', 'best_epoch',
                  'test_samples', 'successful_predictions', 'parameter_count')
LABELS = ('은닉 크기', '층 수', '배치 크기', '학습률', '드롭아웃', '가중치 감소')
COLORS = dict(zip(ARCHITECTURES, ('#2867C8', '#C81E3A', '#18856B')))
MARKERS = dict(zip(ARCHITECTURES, ('o', '^', 's')))
SOURCE = '데이터: ESTA | github.com/pnxenopoulos/esta   /   결과: AWS 전체 데이터 실험 150건'
EXPECTED_TEST_SAMPLES = 6107209


def load_results(paths):
    with paths.csv.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    planned = {r['name']: r for r in json.loads(paths.planned.read_text(encoding='utf-8'))}
    pack = json.loads(paths.export.read_text(encoding='utf-8'))
    if pack['status']['status'] != 'complete' or len(rows) != 150:
        raise ValueError('Expected the completed 150-model experiment')
    if len({r['name'] for r in rows}) != 150:
        raise ValueError('Duplicate model names')
    for row in rows:
        for key in INTEGER_FIELDS:
            row[key] = int(row[key])
        for key in (*PARAMETERS[3:], 'accuracy_percent', 'validation_mean_distance', 'elapsed_seconds'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Non-finite result: ' + row['name'])
        row['condition'] = int(row['name'].rsplit('_', 1)[1])
        if row['architecture'] not in ARCHITECTURES or not 1 <= row['condition'] <= 50:
            raise ValueError('Unexpected architecture or condition')
        if any(row[k] != planned[row['name']][k] for k in (*PARAMETERS, 'architecture')):
            raise ValueError('Actual configuration differs from the frozen plan')
        if row['test_samples'] != EXPECTED_TEST_SAMPLES or not 0 <= row['successful_predictions'] <= row['test_samples']:
            raise ValueError('Unexpected evaluation sample count')
        if not math.isclose(row['accuracy_percent'], 100 * row['successful_predictions'] / row['test_samples'], abs_tol=1e-9):
            raise ValueError('Accuracy does not match the successful prediction count')
    if collections.Counter(r['architecture'] for r in rows) != {a: 50 for a in ARCHITECTURES}:
        raise ValueError('Expected 50 models per architecture')
    for number in range(1, 51):
        group = [r for r in rows if r['condition'] == number]
        if len(group) != 3 or any(tuple(r[k] for k in PARAMETERS) != tuple(group[0][k] for k in PARAMETERS) for r in group):
            raise ValueError('The paired configuration is inconsistent')
    best = {a: min((r for r in rows if r['architecture'] == a), key=lambda r: r['validation_mean_distance']) for a in ARCHITECTURES}
    if {a: r['name'] for a, r in best.items()} != pack['selection']['best_per_architecture']:
        raise ValueError('Validation selection mismatch')
    if not pack['selection']['test_not_used_for_selection']:
        raise ValueError('The test set was used for selection')
    return rows, best


def style_axis(axis, limits):
    axis.set_ylim(limits)
    axis.set_ylabel('테스트 정확도 (%)', labelpad=9)
    axis.grid(axis='y', color='#E3E6EB', linewidth=0.7, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color('#B8BEC6')
    axis.tick_params(colors='#434B57')


def save_figure(figure, stem, paths):
    for extension in ('png', 'svg'):
        figure.savefig(paths.output / (stem + '.' + extension), dpi=180, facecolor='white')


def main():
    parser = argparse.ArgumentParser()
    for name in ('csv', 'export', 'planned', 'template', 'fragment', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rows, best = load_results(args)
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Malgun Gothic', 'Noto Sans CJK KR', 'Arial Unicode MS') if f in fonts), None)
    if font is None:
        raise RuntimeError('A Korean font is required for readable exported figures')
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 11,
                         'axes.labelsize': 12, 'axes.titlesize': 15, 'svg.fonttype': 'none'})
    limits = (math.floor(min(r['accuracy_percent'] for r in rows)) - 0.5,
              math.ceil(max(r['accuracy_percent'] for r in rows)) + 0.5)
    legend = [Line2D([], [], marker=MARKERS[a], linestyle='none', color=COLORS[a], markersize=7,
                    label=f"{a}  (검증 선정: {best[a]['condition']:02d}번, {best[a]['accuracy_percent']:.2f}%)") for a in ARCHITECTURES]
    overview, ax = plt.subplots(figsize=(13.8, 6.7))
    overview.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.22)
    for architecture in ARCHITECTURES:
        group = [r for r in rows if r['architecture'] == architecture]
        ax.scatter([r['condition'] for r in group], [r['accuracy_percent'] for r in group],
                   s=42, marker=MARKERS[architecture], color=COLORS[architecture], alpha=0.85, zorder=3)
        b = best[architecture]
        ax.scatter([b['condition']], [b['accuracy_percent']], marker='*', s=210,
                   color=COLORS[architecture], edgecolor='white', linewidth=0.9, zorder=4)
    style_axis(ax, limits)
    ax.set_xlim(0, 51)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel('설정 번호 (01~50)  /  같은 번호의 세 모델은 변수 설정이 동일함', labelpad=12)
    overview.suptitle('150개 모델의 정확도 비교', x=0.075, y=0.96, ha='left', fontsize=22)
    overview.legend(handles=legend, loc='upper left', bbox_to_anchor=(0.07, 0.91), ncol=3, frameon=False, fontsize=10)
    overview.text(0.075, 0.075, '점 하나 = 모델 한 개   /   별표 = 구조별 검증 평균 거리 오차로 선정한 모델', fontsize=10)
    overview.text(0.075, 0.03, SOURCE + '\n정확도: 3초 뒤 실제 위치와 3차원 거리 오차가 250단위 이내인 비율', fontsize=9, color='#6B7280')
    save_figure(overview, '01_all_150_accuracy', args)

    facets, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), sharey=True)
    facets.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.15, hspace=0.40, wspace=0.24)
    facets.suptitle('변수 설정과 정확도의 관계', x=0.07, y=0.97, ha='left', fontsize=22)
    facets.legend(handles=[Line2D([], [], marker=MARKERS[a], linestyle='none', color=COLORS[a], label=a) for a in ARCHITECTURES],
                  loc='upper left', bbox_to_anchor=(0.065, 0.925), ncol=3, frameon=False)
    for ax, key, label in zip(axes.flat, PARAMETERS, LABELS):
        values = sorted({r[key] for r in rows})
        positions = {v: i for i, v in enumerate(values)}
        for number, architecture in enumerate(ARCHITECTURES):
            group = [r for r in rows if r['architecture'] == architecture]
            # Horizontal offsets reveal coincident values; the vertical accuracy is unchanged.
            x = [positions[r[key]] + (number - 1) * 0.17 + ((r['condition'] * 17) % 11 - 5) * 0.009 for r in group]
            ax.scatter(x, [r['accuracy_percent'] for r in group], s=24, marker=MARKERS[architecture],
                       color=COLORS[architecture], alpha=0.58, edgecolors='none', zorder=3)
            b = best[architecture]
            ax.scatter([positions[b[key]] + (number - 1) * 0.17], [b['accuracy_percent']], marker='*',
                       s=140, color=COLORS[architecture], edgecolor='white', linewidth=0.5, zorder=4)
        ax.set_title(label, loc='left', pad=12)
        ax.set_xticks(range(len(values)), [f'{v:g}' for v in values])
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_xlabel(label)
        style_axis(ax, limits)
    facets.text(0.07, 0.073, '모든 그래프의 세로축은 동일함. 각 점은 실제 실험 1건이며, 다른 변수도 함께 달라질 수 있음.', fontsize=10)
    facets.text(0.07, 0.025, SOURCE + '\n정확도: 3초 뒤 실제 위치와 3차원 거리 오차가 250단위 이내인 비율', fontsize=9, color='#6B7280')
    save_figure(facets, '02_six_parameters', args)
    with PdfPages(args.output / 'model_comparison_150.pdf') as pdf:
        pdf.savefig(overview)
        pdf.savefig(facets)
    plt.close('all')

    payload = {'models': rows, 'best': {a: r['name'] for a, r in best.items()},
               'selected': json.loads(args.export.read_text(encoding='utf-8'))['selection']['selected_experiment']}
    template = args.template.read_text(encoding='utf-8')
    if template.count('__MODEL_DATA_JSON__') != 1:
        raise ValueError('Missing or duplicated inline data placeholder')
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(template.replace('__MODEL_DATA_JSON__', json.dumps(payload, ensure_ascii=False, separators=(',', ':'))), encoding='utf-8')
    summary = {'models': len(rows), 'best_by_validation': best,
               'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               'paired_settings_verified': 50, 'test_accuracy_verified_from_counts': True}
    (args.output / 'validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    logging.info('Verified 150 real results, 50 paired settings and all accuracy calculations; exported 2 figures and PDF.')


if __name__ == '__main__':
    main()
