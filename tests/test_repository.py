"""Small CPU checks; no dataset download or full training is required."""

import copy
import csv
import itertools
import json
import lzma
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_accuracy import count_hits
from scripts.evaluate_baselines import NOISE_VARIANCES, kalman
from scripts.prepare_attributes import extract_match
from scripts.run_sweep_150 import Model, validate_experiments

ROOT = Path(__file__).resolve().parents[1]
INPUT_STEPS = 6
PREDICTION_STEPS = 6
SUCCESS_RADIUS = 250.


class PublishedResultsTests(unittest.TestCase):
    def test_all_results_match_plan_and_success_counts(self):
        with (ROOT / 'results/sweep_150/comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        planned = {spec['name']: spec for spec in json.loads((ROOT / 'configs/sweep_150.json').read_text())}
        selection = json.loads((ROOT / 'results/sweep_150/export.json').read_text())['selection']
        self.assertEqual(len(rows), 150)
        self.assertEqual({row['name'] for row in rows}, set(planned))
        for row in rows:
            with self.subTest(name=row['name']):
                self.assertEqual(row['architecture'], planned[row['name']]['architecture'])
                for key in ('hidden_size', 'layers', 'batch_size', 'learning_rate', 'dropout', 'weight_decay'):
                    self.assertEqual(float(row[key]), planned[row['name']][key])
                self.assertEqual(int(row['test_samples']), 6107209)
                self.assertAlmostEqual(float(row['accuracy_percent']),
                                       100 * int(row['successful_predictions']) / int(row['test_samples']))
        best = {architecture: min((row for row in rows if row['architecture'] == architecture),
                                  key=lambda row: float(row['validation_mean_distance']))
                for architecture in ('GRU', 'LSTM', 'MLP')}
        self.assertTrue(selection['test_not_used_for_selection'])
        self.assertEqual({architecture: row['name'] for architecture, row in best.items()},
                         selection['best_per_architecture'])
        self.assertEqual(best['LSTM']['name'], 'lstm_45')
        self.assertEqual(int(best['LSTM']['successful_predictions']), 4497928)
        summary = json.loads((ROOT / 'results/baseline/dataset_summary.json').read_text())
        total = sum(summary['totals'][split + '_samples'] for split in ('train', 'validation', 'test'))
        self.assertEqual(total, 40196193)

    def test_original_and_classical_result_accuracy(self):
        results = json.loads((ROOT / 'results/baseline/baselines.json').read_text())
        for name, result in results['metrics'].items():
            with self.subTest(name=name):
                self.assertEqual(result['samples'], result['hits'] + result['failures'])
                self.assertAlmostEqual(result['accuracy_percent'], 100 * result['hits'] / result['samples'])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.specs = json.loads((ROOT / 'configs/sweep_150.json').read_text())

    def test_all_150_model_shapes_and_backward_passes(self):
        validate_experiments(self.specs)
        torch.manual_seed(42)
        for spec in self.specs:
            with self.subTest(name=spec['name']):
                model = Model(spec, 4)
                prediction = model(torch.randn(2, INPUT_STEPS, 4))
                self.assertEqual(tuple(prediction.shape), (2, 3))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))

    def test_unsafe_experiment_name_is_rejected(self):
        spec = copy.deepcopy(self.specs[0])
        spec['name'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            validate_experiments([spec])

    def test_single_recurrent_layer_dropout_is_rejected(self):
        spec = copy.deepcopy(self.specs[0])
        spec.update(layers=1, dropout=0.2)
        with self.assertRaisesRegex(ValueError, 'Single recurrent layer'):
            validate_experiments([spec])


class EvaluationTests(unittest.TestCase):
    def test_success_boundary_and_nonfinite_distance(self):
        distances = np.array([0., SUCCESS_RADIUS, SUCCESS_RADIUS + 0.001])
        self.assertEqual(count_hits(distances, SUCCESS_RADIUS), 2)
        with self.assertRaisesRegex(ValueError, 'finite'):
            count_hits(np.array([np.nan]), SUCCESS_RADIUS)

    def test_kalman_stationary_and_constant_velocity_horizon(self):
        velocity = np.array([1., 2., 3.])
        positions = np.arange(INPUT_STEPS)[None, :, None] * velocity[None, None, :]
        for noise in itertools.product(NOISE_VARIANCES, repeat=2):
            with self.subTest(noise=noise):
                np.testing.assert_allclose(kalman(np.zeros((1, INPUT_STEPS, 3)), noise, PREDICTION_STEPS), 0.)
                expected = (INPUT_STEPS - 1 + PREDICTION_STEPS) * velocity
                np.testing.assert_allclose(kalman(positions, noise, PREDICTION_STEPS), expected[None, :])


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / 'prepared'
        (self.prepared / 'shards').mkdir(parents=True)
        self.output = self.root / 'extras'
        self.output.mkdir()
        self.velocity = np.array([10., 20., 5.])
        self.frames = []
        window_length = INPUT_STEPS + PREDICTION_STEPS
        # Two valid windows separated by a missing tick, followed by a short new round.
        for index in range(window_length * 2):
            xyz = index * self.velocity
            player = dict(zip(('x', 'y', 'z'), xyz.tolist()))
            player.update(steamID='synthetic-player', isAlive=True, isWalking=index % 2 == 0,
                          isDucking=index % 2 == 1, isAirborne=index % 3 == 0)
            tick = (index + int(index >= window_length)) * 64
            self.frames.append({'tick': tick, 't': {'players': [player]}, 'ct': {'players': []}})
        self.match = {'tickRate': 128, 'parserParameters': {'parseRate': 64},
                      'gameRounds': [{'frames': self.frames}, {'frames': self.frames[:INPUT_STEPS]}]}
        x, y = [], []
        for start in (0, window_length):
            indices = np.arange(start, start + INPUT_STEPS)
            coordinates = indices[:, None] * self.velocity
            walking = (indices % 2 == 0).astype(np.float32)[:, None]
            x.append(np.concatenate((coordinates - coordinates[-1], walking), axis=1))
            y.append(PREDICTION_STEPS * self.velocity)
        self.x, self.y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
        self.item = {'source': 'synthetic.json.xz', 'shard': 'synthetic.npz', 'samples': len(x)}

    def extract(self):
        with lzma.open(self.root / self.item['source'], 'wt', encoding='utf-8') as stream:
            json.dump(self.match, stream)
        with (self.prepared / 'shards' / self.item['shard']).open('wb') as stream:
            np.savez(stream, x=self.x, y=self.y)
        return extract_match((self.item, self.root, self.prepared, self.output, 3.))

    def test_target_horizon_tick_gap_round_boundary_and_alignment(self):
        _, columns, samples = self.extract()
        self.assertEqual(samples, 2)
        self.assertIn('isWalking', columns)
        with np.load(self.output / self.item['shard'], allow_pickle=False) as result:
            self.assertEqual(result['extra'].shape, (2, INPUT_STEPS, 2))
            expected = np.array([[index % 2 == 1, index % 3 == 0] for index in range(INPUT_STEPS)])
            np.testing.assert_array_equal(result['extra'][0], expected)

    def test_stationary_input_window_is_excluded(self):
        previous = self.frames[2]['t']['players'][0]
        self.frames[3]['t']['players'][0].update({key: previous[key] for key in ('x', 'y', 'z')})
        self.x, self.y = self.x[1:], self.y[1:]
        self.item['samples'] = 1
        self.assertEqual(self.extract()[2], 1)

    def test_misaligned_target_is_rejected(self):
        self.y[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'do not align'):
            self.extract()

    def test_nonboolean_extra_attribute_is_rejected(self):
        self.frames[0]['t']['players'][0]['isDucking'] = 'false'
        with self.assertRaisesRegex(ValueError, 'non-boolean'):
            self.extract()


if __name__ == '__main__':
    unittest.main()
