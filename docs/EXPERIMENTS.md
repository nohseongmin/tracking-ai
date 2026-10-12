# Experiment record

## Recorded preprocessing counts

Source: the actual AWS `dataset_summary.json` report, summarized in [dataset_summary.json](../results/baseline/dataset_summary.json). The upstream ESTA README describes its own dataset totals; the counts below describe the local parser/preprocessing run and should not be substituted for those upstream totals.

| Item | Count |
| --- | ---: |
| Match files read | 1,558 |
| Matches producing retained samples | 1,525 |
| Rounds observed | 41,076 |
| Stored frames observed | 7,708,638 |
| Player records observed | 77,077,619 |
| Alive-player records observed | 58,503,985 |
| Candidate windows | 53,986,310 |
| Windows rejected for tick gaps | 2,987,542 |
| Windows rejected for near-stationary input intervals | 10,802,575 |
| Retained training windows | 27,990,778 |
| Retained validation windows | 6,098,206 |
| Retained test windows | 6,107,209 |
| Total retained windows | 40,196,193 |

Split percentages apply to **match files**, not exact sample counts. Windows overlap within a player track, but whole match files remain in one split. A track does not continue across rounds. The coordinate scale is the training-only RMS of relative input coordinates, with a lower bound of 1; the recorded scale was 116.31114467987345.

## Timing and targets

For a match, `prediction_steps = max(1, round(3 * tickRate / parseRate))`. With `tickRate=128` and `parseRate=64`, each stored observation is 0.5 seconds apart and the future horizon is six intervals.

| Stored record | 1 | 2 | 3 | 4 | 5 | 6 | 7-11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time from first record, seconds | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3-5 | 5.5 |
| Role | Input | Input | Input | Input | Input | Input | Continuity only | Target |

The history spans 2.5 seconds between its endpoints and the prediction horizon is 3 seconds after the last input. At 127 Hz, the six-interval prediction gap is 3.023622 seconds. Server ticks are not the same unit as stored frames.

## Training protocol

| Setting | Value |
| --- | --- |
| Input / target shape | `(6, 4)` / `(3,)` |
| Target | Normalized relative XYZ displacement |
| Loss | Mean squared error (MSE) |
| Optimizer | Adam |
| Maximum epochs | 100 per sweep model |
| Early stopping | Three consecutive epochs without improved validation mean 3D distance |
| Checkpoint | Lowest validation mean distance |
| Seed | 42 |
| Test metric | Percentage of 3D errors at or below 250 CS:GO units |
| Model selection | Lowest validation mean distance, not highest test accuracy |
| Hardware | EC2 g5.xlarge, A10G 24 GB, Ubuntu, 200 GB storage |

The CSV contains all 150 actual completed experiments. Reported epochs are the number executed before early stopping or the maximum; `best_epoch` identifies the restored checkpoint. The best LSTM completed 26 epochs and restored epoch 23.

## Search space and pairing

| Parameter | Values used |
| --- | --- |
| `hidden_size` | 64, 96, 128, 192, 256 |
| `layers` | 1, 2, 3 |
| `batch_size` | 1024, 2048, 4096, 8192 |
| `learning_rate` | 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003 |
| `dropout` | 0, 0.1, 0.2, 0.3 |
| `weight_decay` | 0, 0.000001, 0.00001, 0.0001, 0.001 |

The frozen [configuration file](../configs/sweep_150.json) contains **50 unique baseline-centered settings**, each applied once to GRU, LSTM, and MLP. It is not a full factorial search. The raw Cartesian product would have 7,200 settings per architecture, or 21,600 architecture/configuration combinations; some combinations have redundant or inapplicable dropout settings for one-layer recurrent models.

For GRU/LSTM, `hidden_size` is the state dimension in every recurrent layer. For MLP, it is the first hidden-layer width; subsequent widths are halved. Dropout applies between recurrent layers in GRU/LSTM and after hidden-layer ReLU in MLP. One-layer GRU/LSTM configurations use zero dropout.

Parameter values have unequal experiment counts. For example, hidden size 128 appears in 38 of the 50 settings and batch size 4096 in 41. Parameter-effect analysis must compare settings where the remaining variables match. Equal numeric hyperparameters do not imply equal parameter counts or compute budgets across architectures.

## Additional-column diagnostic

[Pilot results](../results/pilot/results.json) include `isDucking` and `isAirborne` as two additional historical inputs. Their arrays were aligned against the original prepared XYZ/walking arrays before training.

The posture-augmented GRU recorded 69.94% test accuracy, compared with 69.68% for the corresponding base GRU. These posture states are ground-truth game attributes, so the run is marked `oracle_attributes=true` and was excluded from the sound-observation candidate selection. They cannot be assumed available to a creature using sound alone without a separate observation system.

## Reproducibility and limitations

Historical training and evaluation scripts are retained with their original computations. [provenance.json](../results/provenance.json) records their source filenames and SHA-256 hashes. Presentation-generation code, raw player-level exports, AWS screenshots, credentials, prepared arrays, and trained weights are outside this publication.

Publication checks use Python 3.12, PyTorch 2.12.0, NumPy 2.4.6, Matplotlib 3.10.9, and Pillow 12.2.0. The historical full run was executed in the school's AWS GPU environment; its complete package lockfile was not captured. These publication checks are not a new full training run. Library versions and GPU nondeterminism can change reproduced scores.

Only one seed was evaluated. Results concern filtered, continuously moving input windows and a single future endpoint. The model receives no map geometry and does not guarantee wall-respecting or floor-correct predictions. Real audio localization, unseen-map transfer, Unity latency, creature search decisions, and player enjoyment remain future evaluations.

The planned next experiment is hard-example oversampling from **training data only**, keeping evaluation splits intact and controlling total sampling volume. No hard-example training result is claimed in this release.

Dataset attribution: [ESTA](https://github.com/pnxenopoulos/esta), Xenopoulos et al., [paper](https://arxiv.org/abs/2209.09861), CC BY-SA 4.0.
