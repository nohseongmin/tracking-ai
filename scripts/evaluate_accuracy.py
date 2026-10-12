import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

BATCH_SIZE = 1024
MEAN_ERROR_TOLERANCE = 0.001


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(4, 64, batch_first=True)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1])


def count_hits(distances, radius):
    if not np.isfinite(distances).all():
        raise ValueError('Prediction distances must be finite')
    return int(np.count_nonzero(distances <= radius))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--running-speed', type=float, required=True)
    parser.add_argument('--tolerance-seconds', type=float, required=True)
    parser.add_argument('--speed-source', required=True)
    args = parser.parse_args()
    for value in (args.running_speed, args.tolerance_seconds):
        if not math.isfinite(value) or value <= 0:
            parser.error('Speed and tolerance time must be positive finite numbers')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to reproduce the original GPU evaluation')
    radius = args.running_speed * args.tolerance_seconds
    boundary = np.linalg.norm(np.array([[0., 0., 0.], [150., 200., 0.],
                                       [0., 0., 250.], [250.001, 0., 0.]]), axis=1)
    assert count_hits(boundary, 250.) == 3
    summary = json.loads((args.run / 'dataset_summary.json').read_text())
    previous = json.loads((args.run / 'test_metrics.json').read_text())
    checkpoint = torch.load(args.run / 'best_model.pt', map_location='cpu', weights_only=True)
    if checkpoint['prediction_seconds_requested'] != summary['prediction_seconds_requested']:
        raise ValueError('Checkpoint and dataset prediction horizons differ')
    model = Model().cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    scale = float(checkpoint['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Checkpoint scale must be positive and finite')
    started = time.monotonic()
    total = hits = 0
    distance_sum = 0.
    test_files = [item for item in summary['manifest'] if item['split'] == 'test']
    with torch.inference_mode():
        for index, item in enumerate(test_files, 1):
            with np.load(args.run / 'shards' / item['shard'], allow_pickle=False) as shard:
                x = shard['x'].copy()
                y = shard['y']
            if x.shape != (len(y), 6, 4) or y.shape != (item['samples'], 3):
                raise ValueError('Unexpected shard shape: ' + item['shard'])
            x[:, :, :3] /= scale
            for start in range(0, len(x), BATCH_SIZE):
                batch = torch.from_numpy(x[start:start + BATCH_SIZE]).cuda()
                prediction = model(batch).cpu().numpy() * scale
                distances = np.linalg.norm(prediction - y[start:start + BATCH_SIZE], axis=1)
                hits += count_hits(distances, radius)
                total += len(distances)
                distance_sum += float(distances.sum(dtype=np.float64))
            if index % 50 == 0:
                logging.info('EVALUATED %s/%s test files; %s samples', index, len(test_files), total)
    if total == 0 or total != previous['test_samples']:
        raise ValueError('Evaluation count differs from the original test set')
    mean_error = distance_sum / total
    if not math.isclose(mean_error, previous['mean_distance_units'], abs_tol=MEAN_ERROR_TOLERANCE):
        raise ValueError('Reproduced mean error differs from the original evaluation')
    result = {
        'test_samples': total, 'successful_predictions': hits,
        'failed_predictions': total - hits, 'accuracy_percent': 100 * hits / total,
        'radius_units': radius, 'running_speed_units_per_second': args.running_speed,
        'tolerance_seconds': args.tolerance_seconds, 'speed_source': args.speed_source,
        'speed_definition': 'Maximum normal ground running speed; excludes jumping and falling',
        'criterion': 'Euclidean xyz distance <= radius_units; spherical success region',
        'prediction_seconds_requested': summary['prediction_seconds_requested'],
        'prediction_gap_seconds_min': min(item['prediction_gap_seconds'] for item in test_files),
        'prediction_gap_seconds_max': max(item['prediction_gap_seconds'] for item in test_files),
        'best_epoch': checkpoint['epoch'], 'mean_distance_units': mean_error,
        'evaluation_elapsed_seconds': time.monotonic() - started,
        'test_scope': 'Preprocessed held-out test windows; nearly stationary input windows were excluded',
    }
    with (args.run / 'test_accuracy.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    logging.info(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
