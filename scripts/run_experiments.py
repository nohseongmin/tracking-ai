import argparse
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
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
PROGRESS_SECONDS = 10.
EXPERIMENTS = [
    ('gru_1024', 'GRU', 1024, False),
    ('lstm_1024', 'LSTM', 1024, False),
    ('gru_4096', 'GRU', 4096, False),
    ('gru_posture_1024', 'GRU', 1024, True),
]


class Model(nn.Module):
    def __init__(self, architecture, features):
        super().__init__()
        recurrent = {'GRU': nn.GRU, 'LSTM': nn.LSTM}[architecture]
        self.rnn = recurrent(features, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 3)

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1])


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attributes', type=Path, required=True)
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
    summary = json.loads((args.prepared_run / 'dataset_summary.json').read_text())
    if summary['input_shape'] != [6, 4] or summary['prediction_seconds_requested'] != 3.:
        raise ValueError('Expected the existing 6x4 input, 3-second dataset')
    old_checkpoint = torch.load(args.prepared_run / 'best_model.pt', map_location='cpu', weights_only=True)
    scale = float(old_checkpoint['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scale')
    args.output.mkdir(parents=True, exist_ok=False)
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
    for name, architecture, batch_size, attributes in EXPERIMENTS:
        run = args.output / name
        run.mkdir()
        features = BASE_FEATURES + (2 if attributes else 0)
        if attributes:
            logging.info('WAITING FOR ATTRIBUTE PREPROCESSING')
            while not (args.attributes / 'attributes_summary.json').exists():
                error = args.attributes / 'FAILED.txt'
                if error.exists():
                    raise RuntimeError(error.read_text())
                time.sleep(5)
            for cached in cache:
                item = cached['meta']
                with np.load(args.attributes / item['shard'], allow_pickle=False) as data:
                    extra = data['extra']
                if extra.shape != (item['samples'], 6, 2) or not np.isfinite(extra).all():
                    raise ValueError('Invalid attribute shape')
                cached['x'] = torch.cat((cached['x'], torch.from_numpy(extra).cuda()), dim=2)
        seed = int(summary['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.shuffle(list(range(summary['totals']['matches'])))
        model = Model(architecture, features).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        config = {'name': name, 'architecture': architecture, 'layers': 1, 'hidden_size': HIDDEN_SIZE,
                  'input_columns': ['x', 'y', 'z', 'isWalking'] + (['isDucking', 'isAirborne'] if attributes else []),
                  'input_shape': [6, features], 'prediction_seconds': 3., 'batch_size': batch_size,
                  'learning_rate': LEARNING_RATE, 'optimizer': 'Adam', 'loss': 'MSE', 'seed': seed,
                  'max_epochs': args.epochs, 'patience': args.patience, 'dropout': 0., 'weight_decay': 0.,
                  'radius_units': args.radius, 'parameter_count': sum(p.numel() for p in model.parameters()),
                  'dataset_source': 'https://github.com/pnxenopoulos/esta',
                  'prepared_run': str(args.prepared_run), 'oracle_attributes': attributes,
                  'selection_metric': 'validation_mean_distance', 'gpu_cache': True}
        save_json(run / 'config.json', config)
        logging.info('EXPERIMENT START %s | %s | batch %d | %d parameters', name, architecture,
                     batch_size, config['parameter_count'])
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
        logging.info('RESULT %s | test accuracy %.2f%% | best epoch %d | completed %d | %.1fs',
                     name, result['accuracy_percent'], best_epoch, len(history), result['elapsed_seconds'])
        del model, optimizer
    eligible = [result for result in results if not result['oracle_attributes']]
    selected = min(eligible, key=lambda result: result['validation_mean_distance'])
    save_json(args.output / 'selection.json', {'selected_experiment': selected['name'],
              'criterion': 'Smallest validation mean distance among base-input experiments',
              'test_not_used_for_selection': True, 'oracle_excluded': True})
    logging.info('ALL EXPERIMENTS COMPLETE | selected using validation: %s', selected['name'])


if __name__ == '__main__':
    main()
