import argparse
import csv
import os
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_STEPS = 6
BASE_FEATURES = 4
PROGRESS_SECONDS = 10.



class Model(nn.Module):
    def __init__(self, spec, features):
        super().__init__()
        self.architecture = spec['architecture']
        if self.architecture == 'MLP':
            units = [spec['hidden_size'] // (2 ** level) for level in range(spec['layers'])]
            modules = [nn.Flatten(start_dim=1)]
            previous = INPUT_STEPS * features
            for width in units:
                modules.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(spec['dropout'])])
                previous = width
            modules.append(nn.Linear(previous, 3))
            self.network = nn.Sequential(*modules)
        else:
            recurrent = {'GRU': nn.GRU, 'LSTM': nn.LSTM}[self.architecture]
            self.rnn = recurrent(features, spec['hidden_size'], num_layers=spec['layers'],
                                 dropout=spec['dropout'], batch_first=True)
            self.head = nn.Linear(spec['hidden_size'], 3)

    def forward(self, x):
        if self.architecture == 'MLP':
            return self.network(x)
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1])



def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_experiments(experiments):
    if not experiments or len({s['name'] for s in experiments}) != len(experiments):
        raise ValueError('Experiment names must be nonempty and unique')
    for spec in experiments:
        if not spec['name'] or not all(c.isalnum() or c == '_' for c in spec['name']):
            raise ValueError('Unsafe experiment name')
        if spec['architecture'] not in {'GRU', 'LSTM', 'MLP'}:
            raise ValueError('Unsupported architecture')
        for key in ['hidden_size', 'layers', 'batch_size']:
            if type(spec[key]) is not int or spec[key] < 1:
                raise ValueError('Positive integer required: ' + key)
        if spec['layers'] > 3 or spec['hidden_size'] < 8:
            raise ValueError('Unsupported depth/width')
        for key in ['learning_rate', 'dropout', 'weight_decay']:
            if not math.isfinite(spec[key]):
                raise ValueError('Finite configuration required: ' + key)
        if spec['learning_rate'] <= 0 or spec['weight_decay'] < 0 or not 0 <= spec['dropout'] < 1:
            raise ValueError('Invalid learning rate/regularization')
        if spec['architecture'] != 'MLP' and spec['layers'] == 1 and spec['dropout']:
            raise ValueError('Single recurrent layer has no inter-layer dropout')


def save_comparison(path, results):
    fields = ['name', 'architecture', 'hidden_size', 'layers', 'batch_size', 'learning_rate',
              'dropout', 'weight_decay', 'accuracy_percent', 'validation_mean_distance',
              'epochs_completed', 'best_epoch', 'test_samples', 'successful_predictions',
              'parameter_count', 'elapsed_seconds', 'stop_reason']
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiments', type=Path, required=True)
    parser.add_argument('--after-run', type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--radius', type=float, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or not math.isfinite(args.radius) or args.radius <= 0:
        parser.error('Epochs, patience and radius must be positive')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    experiments = json.loads(args.experiments.read_text())
    validate_experiments(experiments)
    args.output.mkdir(parents=True, exist_ok=True)
    planned = args.output / 'planned_experiments.json'
    if planned.exists() and json.loads(planned.read_text()) != experiments:
        raise ValueError('Existing planned configurations differ; use a new output directory')
    save_json(planned, experiments)
    if args.after_run:
        if not args.wait_pid or args.wait_pid <= 0:
            parser.error('--after-run requires a positive --wait-pid')
        save_json(args.output / 'status.json', {'status': 'queued', 'total': len(experiments),
                  'completed': 0, 'waiting_for': str(args.after_run)})
        logging.info('QUEUED %d FULL-DATA EXPERIMENTS | waiting for existing 5-model run PID %d',
                     len(experiments), args.wait_pid)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
        previous = json.loads((args.after_run / 'experiment_results.json').read_text())
        selection = json.loads((args.after_run / 'selection.json').read_text())
        if len(previous) != 5 or not selection['test_not_used_for_selection']:
            raise RuntimeError('The preceding five-model experiment did not complete')
    logging.info('STARTING %d FULL-DATA EXPERIMENTS', len(experiments))
    summary = json.loads((args.prepared_run / 'dataset_summary.json').read_text())
    if summary['input_shape'] != [6, 4] or summary['prediction_seconds_requested'] != 3.:
        raise ValueError('Expected the existing 6x4 input, 3-second dataset')
    old_checkpoint = torch.load(args.prepared_run / 'best_model.pt', map_location='cpu', weights_only=True)
    scale = float(old_checkpoint['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scale')
    results = []
    cache = []
    for number, item in enumerate(summary['manifest'], 1):
        if Path(item['shard']).name != item['shard']:
            raise ValueError('Unsafe shard name')
        with np.load(args.prepared_run / 'shards' / item['shard'], allow_pickle=False) as data:
            x, y = data['x'], data['y']
        if x.shape != (item['samples'], 6, 4) or y.shape != (item['samples'], 3):
            raise ValueError('Wrong shard shape')
        x[:, :, :3] /= scale
        y /= scale
        cache.append({'meta': item, 'x': torch.from_numpy(x).cuda(), 'y': torch.from_numpy(y).cuda()})
        if number % 250 == 0:
            logging.info('GPU CACHE %d/%d', number, len(summary['manifest']))
    logging.info('GPU CACHE READY | %.2f GiB allocated', torch.cuda.memory_allocated() / 1024**3)
    for experiment_number, spec in enumerate(experiments, 1):
        name, architecture, batch_size = spec['name'], spec['architecture'], spec['batch_size']
        run = args.output / name
        if (run / 'results.json').exists():
            result = json.loads((run / 'results.json').read_text())
            expected = {**spec, 'prepared_run': str(args.prepared_run), 'max_epochs': args.epochs,
                        'patience': args.patience, 'radius_units': args.radius}
            if any(result[key] != value for key, value in expected.items()):
                raise ValueError('Completed experiment configuration changed')
            results.append(result)
            logging.info('RESUME: existing completed result %s', name)
            continue
        run.mkdir(exist_ok=True)
        save_json(args.output / 'status.json', {'status': 'running', 'total': len(experiments),
                  'completed': len(results), 'current_number': experiment_number, 'current': name})
        features = BASE_FEATURES
        seed = int(summary['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.shuffle(list(range(summary['totals']['matches'])))
        model = Model(spec, features).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        loss_fn = nn.MSELoss()
        config = {'name': name, 'architecture': architecture, 'layers': spec['layers'], 'hidden_size': spec['hidden_size'],
                  'input_columns': ['x', 'y', 'z', 'isWalking'],
                  'input_shape': [6, features], 'prediction_seconds': 3., 'batch_size': batch_size,
                  'learning_rate': spec['learning_rate'], 'optimizer': 'Adam', 'loss': 'MSE', 'seed': seed,
                  'max_epochs': args.epochs, 'patience': args.patience, 'dropout': spec['dropout'], 'weight_decay': spec['weight_decay'],
                  'radius_units': args.radius, 'parameter_count': sum(p.numel() for p in model.parameters()),
                  'dataset_source': 'https://github.com/pnxenopoulos/esta',
                  'prepared_run': str(args.prepared_run), 'oracle_attributes': False,
                  'selection_metric': 'validation_mean_distance', 'gpu_cache': True}
        if architecture == 'MLP':
            config['hidden_units'] = [spec['hidden_size'] // (2 ** level) for level in range(spec['layers'])]
        save_json(run / 'config.json', config)
        logging.info('EXPERIMENT %d/%d START %s | %s | batch %d | %d parameters',
                     experiment_number, len(experiments), name, architecture, batch_size, config['parameter_count'])
        started = time.monotonic()
        best, stale, best_epoch = float('inf'), 0, 0
        history = []
        stop_reason = 'maximum_epochs'
        for epoch in range(1, args.epochs + 1):
            epoch_started = progress_at = time.monotonic()
            train = [entry for entry in cache if entry['meta']['split'] == 'train']
            random.shuffle(train)
            model.train()
            total_loss = torch.zeros((), device='cuda', dtype=torch.float64)
            seen = 0
            logging.info('%s EPOCH %d/%d TRAIN 0.0%%', name, epoch, args.epochs)
            for entry in train:
                x, y = entry['x'], entry['y']
                ids = torch.from_numpy(np.random.permutation(len(x))).cuda()
                for start in range(0, len(x), batch_size):
                    indices = ids[start:start + batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(model(x[indices]), y[indices])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().double() * len(indices)
                    seen += len(indices)
                    now = time.monotonic()
                    if now - progress_at >= PROGRESS_SECONDS:
                        logging.info('%s EPOCH %d/%d TRAIN %.1f%% | MSE %.6f', name, epoch, args.epochs,
                                     100 * seen / summary['totals']['train_samples'], total_loss.item() / seen)
                        progress_at = now
            model.eval()
            distance_sum = torch.zeros((), device='cuda', dtype=torch.float64)
            val_hits, val_n = 0, 0
            with torch.no_grad():
                for entry in cache:
                    if entry['meta']['split'] != 'validation':
                        continue
                    x, y = entry['x'], entry['y']
                    for start in range(0, len(x), batch_size):
                        distances = torch.linalg.vector_norm((model(x[start:start + batch_size])
                                                              - y[start:start + batch_size]) * scale, dim=1)
                        distance_sum += distances.double().sum()
                        val_hits += int((distances <= args.radius).sum().item())
                        val_n += len(distances)
            val = distance_sum.item() / val_n
            mse = total_loss.item() / seen
            if seen != summary['totals']['train_samples'] or val_n != summary['totals']['validation_samples']:
                raise ValueError('Training/validation count mismatch')
            if not math.isfinite(val) or not math.isfinite(mse):
                raise FloatingPointError('Nonfinite training result')
            row = {'epoch': epoch, 'training_mse_normalized': mse, 'validation_mean_distance': val,
                   'validation_accuracy_percent': 100 * val_hits / val_n,
                   'elapsed_seconds': time.monotonic() - epoch_started}
            history.append(row)
            save_json(run / 'training_history.json', history)
            if val < best:
                best, stale, best_epoch = val, 0, epoch
                torch.save({'state_dict': model.state_dict(), 'scale': scale, 'epoch': epoch,
                            'config': config}, run / 'best_model.pt')
            else:
                stale += 1
            logging.info('%s EPOCH %d/%d DONE | MSE %.6f | val distance %.3f | val accuracy %.2f%% | stale %d/%d | %.1fs',
                         name, epoch, args.epochs, mse, val, row['validation_accuracy_percent'],
                         stale, args.patience, row['elapsed_seconds'])
            if stale >= args.patience:
                stop_reason = 'early_stopping'
                break
        checkpoint = torch.load(run / 'best_model.pt', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        errors = []
        with torch.no_grad():
            for entry in cache:
                if entry['meta']['split'] != 'test':
                    continue
                x, y = entry['x'], entry['y']
                for start in range(0, len(x), batch_size):
                    distances = torch.linalg.vector_norm((model(x[start:start + batch_size])
                                                          - y[start:start + batch_size]) * scale, dim=1)
                    errors.append(distances.cpu().numpy())
        distances = np.concatenate(errors)
        if len(distances) != summary['totals']['test_samples'] or not np.isfinite(distances).all():
            raise ValueError('Test result count/values invalid')
        hits = int(np.count_nonzero(distances <= args.radius))
        result = {**config, 'epochs_completed': len(history), 'best_epoch': best_epoch,
                  'stop_reason': stop_reason, 'validation_mean_distance': best,
                  'validation_accuracy_percent': history[best_epoch - 1]['validation_accuracy_percent'],
                  'test_samples': len(distances), 'successful_predictions': hits,
                  'accuracy_percent': 100 * hits / len(distances),
                  'mean_distance_units': float(distances.mean(dtype=np.float64)),
                  'median_distance_units': float(np.median(distances)),
                  'p90_distance_units': float(np.percentile(distances, 90)),
                  'elapsed_seconds': time.monotonic() - started}
        save_json(run / 'results.json', result)
        results.append(result)
        save_json(args.output / 'experiment_results.json', results)
        save_comparison(args.output / 'comparison.csv', results)
        save_json(args.output / 'status.json', {'status': 'running', 'total': len(experiments),
                  'completed': len(results), 'last_completed': name})
        logging.info('RESULT %s | test accuracy %.2f%% | best epoch %d | completed %d | %.1fs',
                     name, result['accuracy_percent'], best_epoch, len(history), result['elapsed_seconds'])
        del model, optimizer
    eligible = [result for result in results if not result['oracle_attributes']]
    selected = min(eligible, key=lambda result: result['validation_mean_distance'])
    save_json(args.output / 'selection.json', {'selected_experiment': selected['name'],
              'criterion': 'Smallest validation mean distance among predeclared full-data candidates',
              'test_not_used_for_selection': True, 'oracle_excluded': True,
              'best_per_architecture': {architecture: min((r for r in eligible if r['architecture'] == architecture),
                                        key=lambda r: r['validation_mean_distance'])['name']
                                       for architecture in sorted({r['architecture'] for r in eligible})}})
    save_comparison(args.output / 'comparison.csv', results)
    save_json(args.output / 'status.json', {'status': 'complete', 'total': len(experiments),
              'completed': len(results), 'selected_experiment': selected['name']})
    logging.info('ALL EXPERIMENTS COMPLETE | selected using validation: %s', selected['name'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('EXPERIMENT SWEEP FAILED: see traceback; completed results remain saved')
        raise
