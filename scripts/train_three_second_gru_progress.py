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

INPUT_SHAPE = (6, 4)
OUTPUT_SIZE = 3
HIDDEN_SIZE = 64
BATCH_SIZE = 1024
LEARNING_RATE = 0.001


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SHAPE[1], HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1])


parser = argparse.ArgumentParser()
parser.add_argument('--prepared-run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--progress-seconds', type=float, default=5.)
parser.add_argument('--success-radius', type=float, required=True)
args = parser.parse_args()
if args.epochs <= 0 or args.patience <= 0:
    parser.error('Epochs and patience must be positive')
if any(not math.isfinite(value) or value <= 0
       for value in (args.progress_seconds, args.success_radius)):
    parser.error('Progress interval and success radius must be positive and finite')
logging.basicConfig(level=logging.INFO, format='%(message)s')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required')
prepared = args.prepared_run.resolve(strict=True)
summary = json.loads((prepared / 'dataset_summary.json').read_text())
if summary['input_shape'] != list(INPUT_SHAPE) or summary['output_shape'] != [OUTPUT_SIZE]:
    raise ValueError('Prepared dataset has unexpected input/output shapes')
manifest = summary['manifest']
totals = summary['totals']
for split in ('train', 'validation', 'test'):
    count = sum(item['samples'] for item in manifest if item['split'] == split)
    if count <= 0 or count != totals[split + '_samples']:
        raise ValueError('Prepared sample count does not match: ' + split)
for item in manifest:
    if Path(item['shard']).name != item['shard']:
        raise ValueError('Shard name must be a filename')
    if not (prepared / 'shards' / item['shard']).is_file():
        raise FileNotFoundError(item['shard'])
checkpoint = torch.load(prepared / 'best_model.pt', map_location='cpu', weights_only=True)
prediction_seconds = float(summary['prediction_seconds_requested'])
if not math.isfinite(prediction_seconds) or prediction_seconds <= 0:
    raise ValueError('Prepared prediction horizon must be positive and finite')
if checkpoint['prediction_seconds_requested'] != prediction_seconds:
    raise ValueError('Prepared checkpoint and dataset horizons differ')
scale = float(checkpoint['scale'])
if not math.isfinite(scale) or scale <= 0:
    raise ValueError('Prepared coordinate scale must be positive and finite')
seed = int(summary['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# Replay the original file shuffle so training retains the original RNG sequence.
random.shuffle(list(range(totals['matches'])))
torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=False)
shards = args.output / 'shards'
shards.symlink_to(prepared / 'shards', target_is_directory=True)
write_json(args.output / 'dataset_summary.json', summary)
config = {
    'prepared_run': str(prepared), 'prepared_data_reused': True,
    'fresh_model_initialization': True, 'max_epochs': args.epochs,
    'early_stopping_patience': args.patience, 'progress_seconds': args.progress_seconds,
    'success_radius_units': args.success_radius, 'seed': seed,
    'input_shape': list(INPUT_SHAPE), 'output_shape': [OUTPUT_SIZE],
    'prediction_seconds_requested': prediction_seconds, 'hidden_size': HIDDEN_SIZE,
    'gru_layers': 1, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
    'optimizer': 'Adam', 'loss': 'MSE', 'dropout': 0., 'weight_decay': 0.,
    'coordinate_scale': scale, 'selection_metric': 'validation_mean_distance',
}
write_json(args.output / 'run_config.json', config)
logging.info('START | max_epochs=%d | patience=%d | prediction=%.3fs | radius=%.1f',
             args.epochs, args.patience, prediction_seconds, args.success_radius)
logging.info('DATA | train=%d | validation=%d | test=%d | preprocessing=reused',
             totals['train_samples'], totals['validation_samples'], totals['test_samples'])
model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
history = []
best = float('inf')
stale = 0
started = time.monotonic()
stop_reason = 'maximum_epochs'
for epoch in range(1, args.epochs + 1):
    epoch_started = time.monotonic()
    progress_at = epoch_started
    train = [item for item in manifest if item['split'] == 'train']
    random.shuffle(train)
    model.train()
    train_loss = 0.
    seen = 0
    logging.info('EPOCH %d/%d | TRAIN 0.0%%', epoch, args.epochs)
    for item in train:
        with np.load(shards / item['shard'], allow_pickle=False) as data:
            x, y = data['x'], data['y']
        x[:, :, :3] /= scale
        y /= scale
        ids = np.random.permutation(len(x))
        for start in range(0, len(x), BATCH_SIZE):
            indices = ids[start:start + BATCH_SIZE]
            xx = torch.from_numpy(x[indices]).cuda()
            yy = torch.from_numpy(y[indices]).cuda()
            optimizer.zero_grad()
            loss = loss_fn(model(xx), yy)
            current_loss = loss.item()
            if not math.isfinite(current_loss):
                raise FloatingPointError('Training loss is not finite')
            loss.backward()
            optimizer.step()
            train_loss += current_loss * len(indices)
            seen += len(indices)
            now = time.monotonic()
            if now - progress_at >= args.progress_seconds:
                remaining = (now - epoch_started) * (totals['train_samples'] - seen) / seen
                logging.info('EPOCH %d/%d | TRAIN %.1f%% | %d/%d | MSE %.6f | phase ETA %.0fs',
                             epoch, args.epochs, 100 * seen / totals['train_samples'],
                             seen, totals['train_samples'], train_loss / seen, remaining)
                progress_at = now
    logging.info('EPOCH %d/%d | TRAIN 100.0%% | VALIDATION 0.0%%', epoch, args.epochs)
    model.eval()
    val_sum = 0.
    val_n = 0
    validation_started = progress_at = time.monotonic()
    with torch.no_grad():
        for item in manifest:
            if item['split'] != 'validation':
                continue
            with np.load(shards / item['shard'], allow_pickle=False) as data:
                x, y = data['x'], data['y']
            x[:, :, :3] /= scale
            for start in range(0, len(x), BATCH_SIZE):
                prediction = model(torch.from_numpy(x[start:start + BATCH_SIZE]).cuda())
                prediction = prediction.cpu().numpy() * scale
                distances = np.linalg.norm(prediction - y[start:start + BATCH_SIZE], axis=1)
                val_sum += distances.sum()
                val_n += len(distances)
                now = time.monotonic()
                if now - progress_at >= args.progress_seconds:
                    remaining = (now - validation_started) * (totals['validation_samples'] - val_n) / val_n
                    logging.info('EPOCH %d/%d | VALIDATION %.1f%% | phase ETA %.0fs',
                                 epoch, args.epochs, 100 * val_n / totals['validation_samples'], remaining)
                    progress_at = now
    if seen != totals['train_samples'] or val_n != totals['validation_samples']:
        raise ValueError('Processed sample count differs from prepared dataset')
    val = float(val_sum / val_n)
    if not math.isfinite(val):
        raise FloatingPointError('Validation distance is not finite')
    row = {'epoch': epoch, 'training_mse_normalized': train_loss / seen,
           'validation_mean_distance': val, 'epoch_elapsed_seconds': time.monotonic() - epoch_started}
    history.append(row)
    write_json(args.output / 'training_history.json', history)
    if val < best:
        best = val
        stale = 0
        torch.save({'state_dict': model.state_dict(), 'scale': scale, 'epoch': epoch,
                    'input_shape': list(INPUT_SHAPE), 'prediction_seconds_requested': prediction_seconds},
                   args.output / 'best_model.pt')
    else:
        stale += 1
    logging.info('EPOCH %d/%d DONE | MSE %.6f | val distance %.3f | best %.3f | no improvement %d/%d | %.1fs',
                 epoch, args.epochs, row['training_mse_normalized'], val, best,
                 stale, args.patience, row['epoch_elapsed_seconds'])
    if stale >= args.patience:
        stop_reason = 'early_stopping'
        logging.info('EARLY STOP | epoch=%d | consecutive non-improvements=%d', epoch, stale)
        break
checkpoint = torch.load(args.output / 'best_model.pt', weights_only=True)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
logging.info('TEST | evaluating best epoch %d | %d samples', checkpoint['epoch'], totals['test_samples'])
all_errors = []
all_states = []
with torch.no_grad():
    for item in manifest:
        if item['split'] != 'test':
            continue
        with np.load(shards / item['shard'], allow_pickle=False) as data:
            x, y, states = data['x'], data['y'], data['walking']
        x[:, :, :3] /= scale
        for start in range(0, len(x), BATCH_SIZE):
            prediction = model(torch.from_numpy(x[start:start + BATCH_SIZE]).cuda())
            prediction = prediction.cpu().numpy() * scale
            all_errors.append(np.linalg.norm(prediction - y[start:start + BATCH_SIZE], axis=1))
            all_states.append(states[start:start + BATCH_SIZE])
errors = np.concatenate(all_errors)
states = np.concatenate(all_states)
if len(errors) != totals['test_samples'] or not np.isfinite(errors).all():
    raise ValueError('Test results are incomplete or nonfinite')
hits = int(np.count_nonzero(errors <= args.success_radius))
metrics = {
    'test_samples': len(errors), 'successful_predictions': hits,
    'failed_predictions': len(errors) - hits, 'accuracy_percent': 100 * hits / len(errors),
    'radius_units': args.success_radius, 'criterion': 'Euclidean xyz distance <= radius_units',
    'mean_distance_units': float(errors.mean()), 'median_distance_units': float(np.median(errors)),
    'p90_distance_units': float(np.percentile(errors, 90)),
    'walking_mean_distance_units': float(errors[states == 1].mean()) if (states == 1).any() else None,
    'running_proxy_mean_distance_units': float(errors[states == 0].mean()) if (states == 0).any() else None,
    'best_epoch': checkpoint['epoch'], 'epochs_completed': len(history),
    'max_epochs': args.epochs, 'stop_reason': stop_reason, 'early_stopping_patience': args.patience,
    'prediction_seconds_requested': prediction_seconds, 'coordinate_units': 'CSGO game units',
    'total_elapsed_seconds': time.monotonic() - started,
}
write_json(args.output / 'test_metrics.json', metrics)
write_json(args.output / 'test_accuracy.json', metrics)
logging.info('RESULT | accuracy %.2f%% | best epoch %d | epochs completed %d/%d | mean distance %.3f | %s',
             metrics['accuracy_percent'], metrics['best_epoch'], len(history), args.epochs,
             metrics['mean_distance_units'], stop_reason)
logging.info('COMPLETE %s', json.dumps(metrics, allow_nan=False))
