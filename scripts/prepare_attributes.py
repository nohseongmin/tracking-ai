import argparse
import concurrent.futures
import json
import logging
import lzma
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INPUT_STEPS = 6
MIN_MOVEMENT = 1e-3
ATTRIBUTES = ('isDucking', 'isAirborne')


def extract_match(job):
    item, dataset, prepared, output, horizon = job
    with lzma.open(dataset / item['source'], 'rt', encoding='utf-8') as stream:
        match = json.load(stream)
    rate = float(match['tickRate'])
    parse = int(match['parserParameters']['parseRate'])
    length = INPUT_STEPS + max(1, round(horizon * rate / parse))
    extras, inputs, targets = [], [], []
    counts = Counter()
    fields = set()
    for round_data in match.get('gameRounds', []):
        tracks = defaultdict(list)
        for frame in round_data.get('frames') or []:
            for side in ('t', 'ct'):
                for player in (frame.get(side) or {}).get('players') or []:
                    fields.update(player)
                    if not player.get('isAlive'):
                        continue
                    xyz = [player.get(key) for key in ('x', 'y', 'z')]
                    if any(value is None for value in xyz):
                        continue
                    if any(not isinstance(player.get(key), bool) for key in ATTRIBUTES):
                        raise ValueError('Missing/non-boolean attribute in ' + item['source'])
                    row = (int(frame['tick']), *map(float, xyz), float(bool(player.get('isWalking'))),
                           *(float(player[key]) for key in ATTRIBUTES))
                    tracks[str(player['steamID'])].append(row)
                    counts['alive_records'] += 1
                    for key in ATTRIBUTES:
                        counts[key + '_true'] += int(player[key])
        for track in tracks.values():
            rows = np.asarray(sorted({row[0]: row for row in track}.values()), dtype=np.float64)
            if len(rows) < length:
                continue
            indices = np.arange(len(rows) - length + 1)[:, None] + np.arange(length)
            windows = rows[indices]
            valid = (np.diff(windows[:, :, 0], axis=1) == parse).all(axis=1)
            valid &= np.isfinite(windows[:, :, :5]).all(axis=(1, 2))
            valid &= (np.linalg.norm(np.diff(windows[:, :INPUT_STEPS, 1:4], axis=1), axis=2)
                      >= MIN_MOVEMENT).all(axis=1)
            windows = windows[valid]
            if not len(windows):
                continue
            origin = windows[:, INPUT_STEPS - 1, 1:4]
            x = np.concatenate((windows[:, :INPUT_STEPS, 1:4] - origin[:, None, :],
                                windows[:, :INPUT_STEPS, 4:5]), axis=2)
            inputs.append(x.astype(np.float32))
            targets.append((windows[:, -1, 1:4] - origin).astype(np.float32))
            extras.append(windows[:, :INPUT_STEPS, 5:7].astype(np.float32))
    x, y, extra = np.concatenate(inputs), np.concatenate(targets), np.concatenate(extras)
    with np.load(prepared / 'shards' / item['shard'], allow_pickle=False) as previous:
        if not np.array_equal(previous['x'], x) or not np.array_equal(previous['y'], y):
            raise ValueError('Attribute rows do not align with previous preprocessing: ' + item['shard'])
    if len(extra) != item['samples']:
        raise ValueError('Attribute count mismatch')
    with (output / item['shard']).open('xb') as stream:
        np.savez(stream, extra=extra)
    return dict(counts), sorted(fields), len(extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    summary = json.loads((args.prepared_run / 'dataset_summary.json').read_text())
    horizon = float(summary['prediction_seconds_requested'])
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Invalid prediction horizon')
    args.output.mkdir(parents=True, exist_ok=False)
    counts, fields, sample_count = Counter(), set(), 0
    started = time.monotonic()
    jobs = [(item, args.dataset, args.prepared_run, args.output, horizon)
            for item in summary['manifest']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (record_counts, names, samples) in enumerate(pool.map(extract_match, jobs), 1):
            counts.update(record_counts)
            fields.update(names)
            sample_count += samples
            if index % 25 == 0 or index == len(jobs):
                logging.info('ATTRIBUTES %d/%d | aligned samples %d | %.1fs',
                             index, len(jobs), sample_count, time.monotonic() - started)
    expected = sum(item['samples'] for item in summary['manifest'])
    if sample_count != expected:
        raise ValueError('Total attribute count mismatch')
    result = {'attributes': list(ATTRIBUTES), 'input_only': True, 'aligned_samples': sample_count,
              'matches_with_samples': len(jobs), 'counts': dict(counts), 'player_columns': sorted(fields),
              'alignment_check': 'All original x/y arrays exactly equal after float32 conversion',
              'elapsed_seconds': time.monotonic() - started,
              'source': 'https://github.com/pnxenopoulos/esta'}
    (args.output / 'attributes_summary.json').write_text(json.dumps(result, indent=2))
    logging.info('COMPLETE %s', json.dumps(result))
