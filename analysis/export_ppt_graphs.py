import argparse
import logging
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image
from matplotlib import font_manager
from matplotlib.lines import Line2D

from build_model_graphs import (
    ARCHITECTURES, COLORS, LABELS, MARKERS, PARAMETERS, SOURCE,
    load_results, plt, style_axis,
)


FIGURE_SIZE = (16, 9)
EXPORT_DPI = 200
IMAGE_SIZE = (3200, 1800)
POINT_SIZE = 110
STAR_SIZE = 500
ARCHITECTURE_OFFSET = 0.18
JITTER_STEP = 0.017
METRIC = '정확도: 3초 뒤 실제 위치와의 3차원 거리 오차가 250단위 이내인 비율'


def export_chart(rows, best, settings):
    key, title, output = settings
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    fig.subplots_adjust(left=0.12, right=0.975, top=0.80, bottom=0.32)
    fig.suptitle(title, x=0.06, y=0.965, ha='left', fontsize=34, fontweight='bold')
    legend = [Line2D([], [], marker=MARKERS[a], linestyle='none', color=COLORS[a],
                     markersize=12, label=a) for a in ARCHITECTURES]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(0.052, 0.90),
               ncol=3, frameon=False, fontsize=22, columnspacing=2)
    limits = (math.floor(min(r['accuracy_percent'] for r in rows)) - 0.5,
              math.ceil(max(r['accuracy_percent'] for r in rows)) + 0.5)
    style_axis(ax, limits)
    ax.set_yticks(range(math.ceil(limits[0]), math.floor(limits[1]) + 1, 2))
    ax.tick_params(axis='both', labelsize=22, pad=10)
    ax.yaxis.label.set_size(24)
    ax.yaxis.labelpad = 20
    values = sorted({r[key] for r in rows}) if key else []
    positions = {v: i for i, v in enumerate(values)}
    plotted_count = 0
    for index, architecture in enumerate(ARCHITECTURES):
        group = [r for r in rows if r['architecture'] == architecture]
        x = ([positions[r[key]] + (index - 1) * ARCHITECTURE_OFFSET
              + ((r['condition'] * 17) % 11 - 5) * JITTER_STEP for r in group]
             if key else [r['condition'] for r in group])
        ax.scatter(x, [r['accuracy_percent'] for r in group], s=POINT_SIZE,
                   marker=MARKERS[architecture], color=COLORS[architecture],
                   alpha=0.72, edgecolors='none', zorder=3)
        selected = best[architecture]
        best_x = (positions[selected[key]] + (index - 1) * ARCHITECTURE_OFFSET
                  if key else selected['condition'])
        ax.scatter([best_x], [selected['accuracy_percent']], marker='*', s=STAR_SIZE,
                   color=COLORS[architecture], edgecolor='white', linewidth=1.2, zorder=4)
        plotted_count += len(group)
    if plotted_count != 150:
        raise ValueError('Each chart must show all 150 actual experiments')
    if key:
        ax.set_xticks(range(len(values)), [f'{v:g}' for v in values])
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_xlabel(LABELS[PARAMETERS.index(key)], fontsize=24, labelpad=18)
        note = '다른 변수도 함께 변경된 실험이며, 이 변수 하나의 효과만을 뜻하지 않음'
        if key == 'hidden_size':
            note = '은닉 크기: GRU/LSTM은 각 층의 크기, MLP는 첫 층의 크기 / 다른 변수도 함께 변경됨'
    else:
        ax.set_xlim(0, 51)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax.set_xlabel('설정 번호 (01~50)', fontsize=24, labelpad=18)
        note = '같은 설정 번호의 GRU, LSTM, MLP는 여섯 가지 변수 값이 동일함'
    fig.text(0.06, 0.155, '점 하나 = 모델 한 개    /    별표 = 구조별 검증 오차로 선정한 모델',
             fontsize=17, color='#303640')
    fig.text(0.06, 0.112, note, fontsize=15, color='#434B57')
    fig.text(0.06, 0.064, METRIC, fontsize=15, color='#434B57')
    fig.text(0.06, 0.022, SOURCE, fontsize=14, color='#59616D')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check labels and footnotes against the actual export canvas before saving.
    for artist in (*fig.texts, ax.xaxis.label, ax.yaxis.label,
                   *ax.get_xticklabels(), *ax.get_yticklabels(), *fig.legends):
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > IMAGE_SIZE[0] or box.y1 > IMAGE_SIZE[1]:
            raise ValueError(f'Clipped chart text: {output.name}')
    if any(ax.xaxis.label.get_window_extent(renderer).overlaps(artist.get_window_extent(renderer))
           for artist in fig.texts):
        raise ValueError(f'Axis label overlaps footer: {output.name}')
    fig.savefig(output, dpi=EXPORT_DPI, facecolor='white', transparent=False)
    plt.close(fig)
    with Image.open(output) as image:
        if image.size != IMAGE_SIZE:
            raise ValueError(f'Wrong image dimensions: {output.name}')
        if any(image.convert('RGB').getpixel(point) != (255, 255, 255)
               for point in ((0, 0), (3199, 0), (0, 1799), (3199, 1799))):
            raise ValueError(f'Expected opaque white background: {output.name}')
    logging.info('Saved %s', output.name)


def main():
    parser = argparse.ArgumentParser()
    for name in ('csv', 'export', 'planned', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rows, best = load_results(args)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ('Malgun Gothic', 'Noto Sans CJK KR', 'Arial Unicode MS')
                        if name in fonts), None)
    if korean_font is None:
        raise RuntimeError('Korean font required')
    plt.rcParams.update({'font.family': korean_font, 'axes.unicode_minus': False,
                         'font.size': 22, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [(None, '150개 모델의 정확도 비교', '01_all_150_accuracy.png')]
    specs.extend((key, label + '에 따른 정확도', f'{index:02d}_{key}.png')
                 for index, (key, label) in enumerate(zip(PARAMETERS, LABELS), start=2))
    outputs = []
    for key, title, filename in specs:
        output = args.output / filename
        export_chart(rows, best, (key, title, output))
        outputs.append(output)
    with ZipFile(args.output / 'PPT_graphs_7.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for output in outputs:
            archive.write(output, arcname=output.name)
    logging.info('Verified 7 separate 3200x1800 charts; source results unchanged')


if __name__ == '__main__':
    main()
