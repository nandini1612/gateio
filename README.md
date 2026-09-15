# GateIO — Yaw-Rate-Gated Learned Dead Reckoning for UAV GPS-Outage Bridging

GateIO predicts GPS velocity from inertial data during a GPS outage and integrates
those predictions to estimate position. Its distinguishing component is a
**yaw-rate-gated velocity-persistence prior**: while the aircraft flies straight the
model is held close to the last known velocity, and the gate switches off during turns
so the drift loss teaches turning dynamics. The output head predicts a *change* in
velocity over the last known value, so its default is to hold velocity.

**Result, stated honestly.** Under **leave-one-flight-out cross-validation** across all
five flights (no data leakage; model selection on a held-out validation flight), GateIO
reaches **23.4 m mean endpoint drift** over a 10 s outage, pooled over 1,390 held-out
sequences — **3.6× better** than a tuned EKF (99.5 m) and a constant-velocity baseline
(85.0 m), and **9× better on turns**. A common within-flight split inflates this to
about 7 m; we report the leakage-free number as the honest measure. Cross-flight
bridging on this five-flight dataset is improved by learning, but it is not solved
(42% of outages stay under 5 m). See [Limitations](#limitations).

---

## Results

Endpoint drift in metres after a 10 s outage. `Const-v` holds the last known GPS
velocity through the outage. Lower is better.

### Leave-one-flight-out cross-validation — headline, no leakage

Pooled over all five held-out flights (1,390 sequences). A sequence is *Turn* if its
mean yaw rate during the outage exceeds 0.10 rad/s, else *Straight*.

| Group | GateIO | EKF | Const-v | GateIO % < 5 m |
|---|---:|---:|---:|---:|
| Straight (1200) | **20.8** | 55.9 | 41.4 | 48% |
| Turn (190) | **40.0** | 374.8 | 360.6 | 6% |
| **All (1390)** | **23.4** | 99.5 | 85.0 | 42% |

Per test flight (all sequences): gnss01 27.3, gnss02 13.1, gnss03 17.6,
island_gnss02 23.9, island_gnss03 35.0.

![Leave-one-flight-out result](results/figures/fig_lofo.png)

GateIO beats both classical baselines on every group. The gain is largest on turns,
where constant-velocity and EKF diverge. On straight flight GateIO is roughly twice as
accurate as holding velocity — because even low-yaw flight has real speed changes across
flights, which a learned model captures and pure persistence does not. Reproduce with
`eval/evaluate_lofo.py`; see [`results/lofo_summary.csv`](results/lofo_summary.csv).

### Within-flight split — shown for contrast (inflated by leakage)

The numbers below use a within-flight 80/10/10 split, where validation and test windows
sit next to training windows from the *same* flight. They are **not** the headline: the
split leaks, so it overstates accuracy roughly 3× (6.72 m here vs 23.4 m under
leave-one-flight-out). We keep them to show the size of the effect.

| Metric | Split | GateIO | LSTM | EKF | Const-v |
|---|---|---:|---:|---:|---:|
| Mean (m) | val | 6.72 | 7.78 | 74.80 | 30.64 |
|          | test | 34.11 | 42.50 | 58.16 | 62.97 |
| % under 5 m | val | 76.3% | 79.7% | 13.6% | 74.6% |
|             | test | 0.0% | 0.0% | 11.9% | 84.7% |

The within-flight *validation* number (6.72 m) looks strong; the sealed within-flight
*test* number (34 m) already hints at the problem, and leave-one-flight-out confirms it.
Per-sequence tables: [`results/val_all_results.csv`](results/val_all_results.csv),
[`results/test_all_results.csv`](results/test_all_results.csv).

**Per-group mean drift, validation vs test.** The learned models keep turns bounded where
the classical baselines diverge; on held-out flights they pay a large fixed cost on
straight segments.

![Per-group mean drift, within-flight split](results/figures/fig2_group_bars.png)

**Cumulative distribution of drift.** In-distribution the learned models dominate;
out-of-distribution the constant-velocity baseline (grey) reaches the 5 m mark far sooner
than GateIO.

![Drift CDF, within-flight split](results/figures/fig3_drift_cdf.png)

---

## Method

GateIO encodes each 1 s IMU window (200 samples × 14 channels) into a single token,
adds a positional encoding of *elapsed outage steps*, mixes tokens with a causal TCN
and ALiBi attention, and decodes velocity with a dual-branch head that fuses the last
known GPS velocity during outages.

![GateIO architecture](results/figures/architecture.svg)

The recurrent baseline, **GateIO-LSTM**, is identical except that the TCN + attention
backbone is replaced by a 2-layer causal LSTM.

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
is enabled with `--v2` in training and is the change that makes GateIO generalize across
flights. The original absolute-velocity head is kept for reference (default, reproduces
the within-flight numbers).

| Loss term | Where it applies | Weight |
|---|---|---|
| `L_data` — Huber on GPS velocity | GPS-aided windows | 1.0 |
| `L_dr` — Huber on the velocity increment | all outage windows | 0.90 |
| `L_cvprior` — push the increment → 0 (hold velocity) | outage windows with `|ω_z|<0.10 rad/s` | 0.50 |
| `L_smooth` — jerk penalty | whole sequence | 0.001 |
| `L_drift` — cumulative position error | outage, warmed in after epoch 30 | 0.002 |

GateIO has **186,390 parameters**; the TCN receptive field spans ≈ 25 s of context.

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
also report the fraction of outages under 5 m.

---

## Repository layout

```
gateio/
├── models/gateio.py               GateIO + GateIO-LSTM (residual head via --v2)
├── train/train_gateio.py          training loop (--v2 residual, --lam-cvprior knob)
├── eval/
│   ├── ekf_baseline.py            quaternion gravity-compensated EKF
│   ├── evaluate.py                within-flight eval + persistence-check diagnostic
│   └── evaluate_lofo.py           leave-one-flight-out pooled evaluation
├── data/
│   ├── data_loader.py             MARSDataset (windowing, normalization, outage sim)
│   └── preprocess/
│       ├── combine_and_norm_v2.py within-flight split
│       └── combine_baglevel.py    leave-one-flight-out folds
├── notebooks/                     cleaned training / evaluation notebooks
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

Build the 5 folds (needs the per-flight `transformer_ds_<flight>.npz` files), train v2
on each, then pool:

```bash
for k in 0 1 2 3 4; do
  python data/preprocess/combine_baglevel.py --fold $k \
      --processed-dir path/to/processed --out folds/fold$k.npz
  python train/train_gateio.py --data folds/fold$k.npz --model gateio --v2 \
      --ckpt-dir checkpoints_lofo/fold$k
done
python eval/evaluate_lofo.py --folds-dir folds --ckpt-dir checkpoints_lofo
```

Quick architecture self-test (no data) — prints `186,390` parameters:

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
  low statistical power. This is the main limit on generalization.
- **Modest absolute accuracy.** 23 m mean drift and 42% of outages under 5 m is a
  research result, not a fielded navigation solution.
- **No uncertainty and no filter coupling.** GateIO gives a point estimate; predicting
  uncertainty and fusing it with a filter is known to help and is not done here.
- **No published learned baseline.** We compare to constant-velocity, a tuned EKF, and
  our own LSTM, not to published learned-odometry methods run on this data.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{gateio2026,
  title  = {GateIO: Yaw-Rate-Gated Learned Dead Reckoning for UAV GPS-Outage Bridging},
  author = {Saxena, Nandini},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/nandini1612/gateio}
}
```
