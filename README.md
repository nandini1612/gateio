# GateIO — Yaw-Rate-Gated Learned Dead Reckoning for UAV GPS-Outage Bridging

**Preprint:** [10.5281/zenodo.22853978](https://doi.org/10.5281/zenodo.22853978)

GateIO predicts GPS velocity from inertial data during a GPS outage and integrates
those predictions to estimate position. The contribution is a **formulation**, not a
particular network: a **yaw-rate-gated velocity-persistence residual head**. The output
head predicts a *change* in velocity over the last known value, so its default is to hold
velocity; while the aircraft flies straight a gate holds the model close to the last known
velocity, and the gate switches off during turns so the drift loss teaches turning
dynamics. We evaluate this formulation with two interchangeable backbones — a recurrent
network (**GateIO-LSTM**) and a temporal-convolution-plus-attention network
(**GateIO-TCN**).

**Result, stated honestly.** Under **leave-one-flight-out cross-validation** across all
five flights (no data leakage; model selection on a held-out validation flight), the
recurrent backbone **GateIO-LSTM reaches 11.2 m mean endpoint drift** over a 10 s outage,
pooled over 1,390 held-out sequences — about **7.5× better** than a tuned EKF (99.5 m) and
a constant-velocity baseline (85.0 m), and roughly **10× better on turns**. The
convolution-plus-attention backbone, GateIO-TCN, is worse (23.4 m): on this small
five-flight dataset the recurrent model generalizes better and avoids the large tail
errors the TCN incurs on straight flight. Both learned models beat the classical baselines
by wide margins; **the recurrent backbone is the one to use.** A common within-flight split
makes both models look better — about 3× for the convolutional backbone — and even flips
which one appears best; we report the leakage-free figures as the honest measure. Cross-flight bridging on this five-flight dataset is improved by learning, but it
is not solved (55% of outages stay under 5 m for GateIO-LSTM). See [Limitations](#limitations).

---

## Results

Endpoint drift in metres after a 10 s outage. `Const-v` holds the last known GPS
velocity through the outage. Lower is better.

### Leave-one-flight-out cross-validation — headline, no leakage

Pooled over all five held-out flights (1,390 sequences). A sequence is *Turn* if its
mean yaw rate during the outage exceeds 0.10 rad/s, else *Straight*. Both learned models
use the same yaw-rate-gated residual head; they differ only in the backbone.

| Group | **GateIO-LSTM** | GateIO-TCN | EKF | Const-v | LSTM % < 5 m |
|---|---:|---:|---:|---:|---:|
| Straight (1200) | **7.2** | 20.8 | 55.9 | 41.4 | 62% |
| Turn (190) | **36.8** | 40.0 | 374.8 | 360.6 | 6% |
| **All (1390)** | **11.2** | 23.4 | 99.5 | 85.0 | 55% |

Per test flight, mean drift over all sequences (GateIO-LSTM / GateIO-TCN):
gnss01 4.1 / 27.3, gnss02 11.1 / 13.1, gnss03 20.3 / 17.6, island_gnss02 16.5 / 23.9,
island_gnss03 30.1 / 35.0. GateIO-LSTM wins on four of the five flights; the TCN backbone
wins only on gnss03.

![Leave-one-flight-out result](results/figures/fig_lofo.png)

GateIO-LSTM beats every baseline on every group. The gain over classical methods is largest
on turns, where constant-velocity and EKF diverge. On straight flight it is roughly six
times more accurate than holding velocity, because even low-yaw flight has real speed
changes across flights that a learned model captures and pure persistence does not. The
TCN backbone captures the same turning behaviour but carries a heavy tail on straight
flight — its median drift is competitive (10.8 m) but individual sequences blow up past
60 m — which is what pushes its mean to 23.4 m. Reproduce with `eval/evaluate_lofo.py`;
see [`results/lofo_summary.csv`](results/lofo_summary.csv).

**Two held-out outages on gnss03.** On a typical straight segment (left, the common case)
GateIO-LSTM tracks the true path within 1.4 m over 10 s. On a turn (right) constant
velocity shoots off along the pre-outage heading while the learned model follows the
curve; the error is larger but stays bounded. Turns are about one outage in seven, and
they are where the learned model earns its margin over the classical baselines.

![Held-out trajectories](results/figures/fig_trajectory.png)

### Within-flight split — shown for contrast (inflated by leakage)

The numbers below use a within-flight 80/10/10 split, where validation and test windows
sit next to training windows from the *same* flight. They are **not** the headline: the
split leaks. Its validation figures (about 7 m for both backbones) sit far below the
leakage-free ones — roughly 1.4× for GateIO-LSTM and 3.5× for GateIO-TCN — and the split
even reverses which backbone looks best. We keep them to show the size of the effect.

| Metric | Split | GateIO-LSTM | GateIO-TCN | EKF | Const-v |
|---|---|---:|---:|---:|---:|
| Mean (m) | val | 7.78 | 6.72 | 74.80 | 30.64 |
|          | test | 42.50 | 34.11 | 58.16 | 62.97 |
| % under 5 m | val | 79.7% | 76.3% | 13.6% | 74.6% |
|             | test | 0.0% | 0.0% | 11.9% | 84.7% |

The within-flight *validation* numbers look strong; the sealed within-flight *test*
numbers already hint at the problem, and leave-one-flight-out confirms it. Note that the
within-flight split even reverses the backbone ranking — the TCN looks slightly better
in-distribution but generalizes worse — another reason to trust only the leakage-free
result. Per-sequence tables: [`results/val_all_results.csv`](results/val_all_results.csv),
[`results/test_all_results.csv`](results/test_all_results.csv).

---

## Method

GateIO encodes each 1 s IMU window (200 samples × 14 channels) into a single token,
adds a positional encoding of *elapsed outage steps*, mixes tokens with a backbone, and
decodes velocity with a dual-branch head that fuses the last known GPS velocity during
outages. The head is the same for both backbones; only the token-mixing network changes.

- **GateIO-LSTM** (recommended) — a 2-layer causal LSTM backbone. Best under
  leave-one-flight-out (11.2 m). 265,334 parameters.
- **GateIO-TCN** — a causal temporal-convolution backbone (2 × 5 dilated causal blocks,
  ≈ 25 s receptive field) with ALiBi causal attention. 186,390 parameters. Higher capacity
  but generalizes worse on this small dataset (23.4 m).

![GateIO architecture](results/figures/architecture.svg)

The diagram shows the GateIO-TCN backbone; GateIO-LSTM replaces the TCN + attention block
with the 2-layer causal LSTM and is otherwise identical.

### The yaw-rate gate

On outage windows with `|ω_z| < 0.10 rad/s` (straight cruise) a prior pushes the
predicted velocity *change* toward zero — hold the last known velocity. Above the
threshold (turns) the gate is off and the drift loss shapes the prediction. An earlier
ungated version of this prior fought the drift loss on turns and lost; gating is what
lets it carry weight.

![The yaw-rate gate](results/figures/gate.svg)

### Residual output (v2)

The dead-reckoning head predicts the change in velocity over the last known value,
scaled by the natural size of that change. A zero output is exactly "hold velocity", so
on an unseen flight the model degrades to persistence instead of to a biased guess. This
is enabled with `--v2` in training and is the change that makes both backbones generalize
across flights. The original absolute-velocity head is kept for reference (default,
reproduces the within-flight numbers).

| Loss term | Where it applies | Weight |
|---|---|---|
| `L_data` — Huber on GPS velocity | GPS-aided windows | 1.0 |
| `L_dr` — Huber on the velocity increment | all outage windows | 0.90 |
| `L_cvprior` — push the increment → 0 (hold velocity) | outage windows with `|ω_z|<0.10 rad/s` | 0.50 |
| `L_smooth` — jerk penalty | whole sequence | 0.001 |
| `L_drift` — cumulative position error | outage, warmed in after epoch 30 | 0.002 |

---

## Evaluation protocol

**Task.** During a simulated GPS outage the model predicts GPS velocity from IMU data
and the last known GPS-aided velocity. Velocity is integrated to a position estimate and
scored by horizontal endpoint drift.

**Windowing.** Each input window is 200 IMU samples (1.0 s at 200 Hz), encoded to one
token; predictions are made at 10 Hz. A sequence is 300 windows (30 s).

**Outage.** GPS channels are zeroed for 100 windows (10 s), from window 100 to 199, with
a 3-window ramp on the outage flag at onset. Position is integrated from outage onset.

**Split — leave-one-flight-out.** Whole flights are assigned to train / validation /
test. We run 5 folds; each flight is the test flight once, with 3 flights for training
and 1 held-out flight for validation and model selection. **No window shares a flight
across splits**, so there is no train/test leakage.

**Normalisation.** Per-channel median and IQR are computed on the training flights only
(target IQR floored at 0.1 m/s; quaternion channels passed through unscaled).

**Baselines.** Constant-velocity (hold the last GPS velocity); a 6-state EKF with
per-timestep quaternion gravity compensation, with process and measurement noise tuned
by Nelder-Mead on each fold's validation flight only — never on the test flight.

**Metric.** Endpoint drift = horizontal (xy) position error at the last outage window,
integrated from outage onset, in metres, averaged over all held-out test sequences; we
also report the fraction of outages under 5 m, and the median and IQR across sequences.

---

## Repository layout

```
gateio/
├── models/gateio.py               GateIO backbones (LSTM + TCN) and residual head (--v2)
├── train/train_gateio.py          training loop (--v2 residual, --model lstm|gateio)
├── eval/
│   ├── ekf_baseline.py            quaternion gravity-compensated EKF
│   ├── evaluate.py                within-flight eval + persistence-check diagnostic
│   ├── evaluate_lofo.py           leave-one-flight-out eval (both backbones via --lstm)
│   └── plot_lofo_trajectory.py    held-out trajectory figure
├── data/
│   ├── data_loader.py             MARSDataset (windowing, normalization, outage sim)
│   └── preprocess/
│       ├── combine_and_norm_v2.py within-flight split
│       └── combine_baglevel.py    leave-one-flight-out folds
├── notebooks/                     cleaned training / evaluation notebooks (05 = LOFO extras)
├── results/                       per-sequence CSVs, LOFO summary, figures
└── paper/                         preprint (docx/PDF) + gateio_v2_preprint.md
```

## Installation

```bash
git clone https://github.com/nandini1612/gateio.git
cd gateio
pip install -r requirements.txt
```

PyTorch ≥ 2.6 is supported; checkpoints must be loaded with `weights_only=False`.

## Reproducing the leave-one-flight-out result

Build the 5 folds (needs the per-flight `transformer_ds_<flight>.npz` files), train both
backbones on each, then pool:

```bash
for k in 0 1 2 3 4; do
  python data/preprocess/combine_baglevel.py --fold $k \
      --processed-dir path/to/processed --out folds/fold$k.npz
  python train/train_gateio.py --data folds/fold$k.npz --model lstm   --v2 --ckpt-dir checkpoints_lofo/fold$k
  python train/train_gateio.py --data folds/fold$k.npz --model gateio --v2 --ckpt-dir checkpoints_lofo/fold$k
done
python eval/evaluate_lofo.py --folds-dir folds --ckpt-dir checkpoints_lofo --lstm \
    --out results/lofo_summary.csv
```

`notebooks/05_lofo_extras.ipynb` runs the same steps on Colab. Quick architecture
self-test (no data) — prints `186,390` parameters for the TCN backbone:

```bash
python models/gateio.py
```

The within-flight numbers (for contrast) come from `eval/evaluate.py` on a dataset built
with `combine_and_norm_v2.py`; add `--persistence-check` to see the diagnostic that
motivated the residual head.

## Dataset

Derived from the [MARS-LVIG dataset](https://mars.hku.hk/dataset.html) (Li et al., 2024):
five UAV flights from a DJI M300 RTK platform, three from an airport site and two from an
island site. Two flights (gnss03, island_gnss03) carry most of the turning. Sequences are
30 s at 10 Hz with a simulated 10 s outage. The leave-one-flight-out folds are built by
`combine_baglevel.py`; the within-flight split by `combine_and_norm_v2.py`.

## Limitations

- **Small dataset.** Five flights, two sites, one platform. Leave-one-flight-out gives
  five folds with a single validation and test flight each, so per-flight estimates have
  low statistical power. This is the main limit on generalization, and it is likely why
  the higher-capacity TCN backbone generalizes worse than the LSTM.
- **Modest absolute accuracy.** 11 m mean drift and 55% of outages under 5 m is a
  research result, not a fielded navigation solution.
- **Backbone conclusion is dataset-bound.** The recurrent backbone wins here; on a larger,
  more varied dataset the ordering could change. We report both and recommend the LSTM on
  this data rather than claiming a general architectural result.
- **No uncertainty and no filter coupling.** GateIO gives a point estimate; predicting
  uncertainty and fusing it with a filter is known to help and is not done here.
- **No published learned baseline.** We compare to constant-velocity, a tuned EKF, and two
  backbones of our own, not to published learned-odometry methods run on this data.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{saxena2026gateio,
  title     = {GateIO: Yaw-Rate-Gated Learned Dead Reckoning for UAV GPS-Outage Bridging},
  author    = {Saxena, Nandini},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22853978},
  url       = {https://doi.org/10.5281/zenodo.22853978}
}
```
