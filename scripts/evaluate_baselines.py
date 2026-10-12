import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np

NOISE_VARIANCES = (1.0, 100.0, 10000.0)


def kalman(positions, noise, prediction_steps):
    q, r = noise
    # One time step is one stored frame, including during future extrapolation.
    transition = np.array([[1., 1.], [0., 1.]])
    covariance = np.diag([r, q])
    process_noise = q * np.array([[.25, .5], [.5, 1.]])
    state = np.stack((positions[:, 0], positions[:, 1] - positions[:, 0]), axis=-1).astype(np.float64)
    for index in range(1, positions.shape[1]):
        state = state @ transition.T
        covariance = transition @ covariance @ transition.T + process_noise
        gain = covariance[:, 0] / (covariance[0, 0] + r)
        state += (positions[:, index] - state[:, :, 0])[:, :, None] * gain
        covariance = (np.eye(2) - np.outer(gain, [1., 0.])) @ covariance
    return state[:, :, 0] + prediction_steps * state[:, :, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    summary = json.loads((args.run / 'dataset_summary.json').read_text())
    gru = json.loads((args.run / 'test_accuracy.json').read_text())
    if summary['prediction_seconds_requested'] != gru['prediction_seconds_requested']:
        raise ValueError('Dataset and GRU evaluation horizons differ')
    radius = float(gru['radius_units'])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Success radius must be positive and finite')
    noise_grid = list(itertools.product(NOISE_VARIANCES, repeat=2))
    stationary = np.zeros((2, 6, 3))
    linear = np.arange(6.)[None, :, None] * np.array([1., 2., 3.])[None, None, :]
    for noise in noise_grid:
        assert np.allclose(kalman(stationary, noise, 6), 0.)
        assert np.allclose(kalman(linear, noise, 6), [[11., 22., 33.]])
    boundary = np.linalg.norm(np.array([[150., 200., 0.], [0., 0., 250.], [250.001, 0., 0.]]), axis=1)
    assert np.count_nonzero(boundary <= 250.) == 2
    started = time.monotonic()

    def records(split):
        for item in summary['manifest']:
            if item['split'] != split:
                continue
            with np.load(args.run / 'shards' / item['shard'], allow_pickle=False) as shard:
                x, y = shard['x'], shard['y']
            if x.shape != (item['samples'], 6, 4) or y.shape != (item['samples'], 3):
                raise ValueError('Unexpected shape: ' + item['shard'])
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Non-finite positions: ' + item['shard'])
            if item['prediction_steps'] <= 0:
                raise ValueError('Prediction steps must be positive')
            yield x[:, :, :3], y, item['prediction_steps']

    validation = []
    for noise in noise_grid:
        total = hits = 0
        distance_sum = 0.
        for positions, y, steps in records('validation'):
            distances = np.linalg.norm(kalman(positions, noise, steps) - y, axis=1)
            if not np.isfinite(distances).all():
                raise ValueError('Non-finite Kalman predictions')
            total += len(distances)
            hits += int(np.count_nonzero(distances <= radius))
            distance_sum += float(distances.sum(dtype=np.float64))
        if total == 0 or total != summary['totals']['validation_samples']:
            raise ValueError('Validation count differs from the original split')
        row = {'q': noise[0], 'r': noise[1], 'samples': total, 'hits': hits,
               'accuracy_percent': 100 * hits / total, 'mean_distance_units': distance_sum / total}
        validation.append(row)
        logging.info('VALIDATION %s', json.dumps(row))
    chosen = max(validation, key=lambda row: (row['accuracy_percent'], -row['mean_distance_units']))
    selected_noise = (chosen['q'], chosen['r'])
    metrics = {name: {'samples': 0, 'hits': 0, 'distance_sum': 0.}
               for name in ('last_position', 'kalman')}
    for positions, y, steps in records('test'):
        predictions = {'last_position': positions[:, -1],
                       'kalman': kalman(positions, selected_noise, steps)}
        for name, prediction in predictions.items():
            distances = np.linalg.norm(prediction - y, axis=1)
            if not np.isfinite(distances).all():
                raise ValueError('Non-finite test predictions: ' + name)
            metrics[name]['samples'] += len(distances)
            metrics[name]['hits'] += int(np.count_nonzero(distances <= radius))
            metrics[name]['distance_sum'] += float(distances.sum(dtype=np.float64))
    for name, row in metrics.items():
        if row['samples'] != gru['test_samples']:
            raise ValueError('Test count differs from GRU evaluation: ' + name)
        row['failures'] = row['samples'] - row['hits']
        row['accuracy_percent'] = 100 * row['hits'] / row['samples']
        row['mean_distance_units'] = row.pop('distance_sum') / row['samples']
    metrics['GRU'] = {'samples': gru['test_samples'], 'hits': gru['successful_predictions'],
                      'failures': gru['failed_predictions'], 'accuracy_percent': gru['accuracy_percent'],
                      'mean_distance_units': gru['mean_distance_units']}
    result = {
        'radius_units': radius, 'criterion': gru['criterion'],
        'prediction_seconds_requested': gru['prediction_seconds_requested'],
        'prediction_gap_seconds_min': gru['prediction_gap_seconds_min'],
        'prediction_gap_seconds_max': gru['prediction_gap_seconds_max'],
        'kalman_model': 'Constant-velocity Kalman filter; xyz observations only; six past input positions',
        'kalman_time_unit': 'One stored frame; extrapolation uses each shard prediction_steps',
        'kalman_q': chosen['q'], 'kalman_r': chosen['r'],
        'kalman_selection': 'Best validation accuracy among 9 candidates; mean distance breaks ties',
        'kalman_validation': validation, 'metrics': metrics,
        'evaluation_elapsed_seconds': time.monotonic() - started, 'test_scope': gru['test_scope'],
    }
    with (args.run / 'baseline_accuracy.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    logging.info('COMPARISON_COMPLETE %s', json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
