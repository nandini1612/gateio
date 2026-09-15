# GateIO: Learned Dead Reckoning for UAV GPS-Outage Bridging, Tested Across Held-Out Flights

**Nandini Saxena**
Independent research (work begun during a research internship at DIAT)
nandinisaxenawork@gmail.com

Preprint, 2026. Code and models: https://github.com/nandini1612/gateio

---

## Abstract

A UAV that loses GPS for a few seconds to tens of seconds must estimate its own motion from inertial data alone. Constant-velocity extrapolation drifts badly when the aircraft turns, because horizontal velocity changes with yaw rate. We present GateIO, a small causal network that predicts GPS velocity during an outage from inertial measurements and the last known GPS-aided velocity. GateIO uses one physical idea: while the aircraft flies straight (low yaw rate), the predicted velocity is held close to the last known value; the constraint is switched off during turns so the network still learns turning dynamics.

The main lesson of this work is about evaluation, not architecture. A within-flight train/validation/test split — common in this area — leaks information, because validation and test windows sit next to training windows from the same flight. Under that split GateIO looks very accurate (about 7 m mean drift over a 10-second outage). Under leave-one-flight-out cross-validation, where every test flight is fully held out, the honest number is much larger: 23.4 m mean drift over 1,390 held-out outages. Even so, GateIO reduces drift 3.6 times relative to a tuned extended Kalman filter and a constant-velocity baseline (99 m and 85 m), with the largest gain on turns (9 times). We report the leakage-free result as the true measure of cross-flight performance, and treat the gap between the two protocols as a caution for the field. Cross-flight inertial bridging on this platform is improved by learning, but it is not solved.

---

## 1. Introduction

GPS-denied flight is a normal failure mode for small UAVs. Signals drop near buildings, under bridges, and in jamming. When the gap lasts several seconds and the aircraft is manoeuvring, position error grows fast.

The standard fix is a Kalman filter with a constant-velocity or constant-acceleration model. A well-tuned filter helps on straight flight, but it has no way to know that a turn is happening, so it keeps extrapolating in a straight line and the error explodes.

Learned inertial odometry offers another route. Networks trained on windowed IMU data can estimate motion directly, and they can pick up patterns a fixed motion model misses. Most of this work targets pedestrians and hand-held or ground platforms. Aerial data is different: speeds are higher, the vehicle rolls into turns, and the useful signal is spread across accelerometer, gyroscope, and attitude channels.

GateIO predicts GPS velocity during an outage and integrates it to a position estimate. It adds one prior tied to flight physics: when yaw rate is low, horizontal velocity is close to constant, so the model is told to hold the last known velocity; when yaw rate is high, the prior is switched off so the drift term teaches the turn.

Our contributions are:

1. **GateIO**, a small (186k-parameter) causal model for UAV GPS-outage bridging, with a yaw-rate-gated velocity-persistence prior and a residual output head that predicts the change in velocity over the last known value.
2. A **leakage-free evaluation**. We show that a within-flight split overstates accuracy by a large margin, and we report leave-one-flight-out cross-validation as the honest measure.
3. An **error analysis** that explains why an earlier version of the model failed to generalise, and a fix (the residual head plus a corrected prior) that recovers most of the loss on unseen straight flight.

We are direct about the ceiling: under the honest protocol, mean drift is 23.4 m and 42% of outages stay under 5 m. GateIO clearly beats the classical baselines, but cross-flight accuracy on this five-flight dataset is modest.

---

## 2. Related Work

**Learned inertial odometry.** IONet framed pedestrian inertial odometry as regressing a displacement increment (a polar vector) from a window of IMU data. RoNIN set strong velocity-regression baselines and a benchmark. TLIO regressed 3D displacement and its uncertainty and fused it into a filter, which gives a principled way to fall back on a motion model when the network is unsure. AirIO brought this to the aerial setting, predicting body-frame velocity with uncertainty; its gains come mostly from the uncertainty term. GateIO predicts a velocity increment, like IONet and TLIO, but produces a point estimate without uncertainty, which we note as a limitation.

**Classical and hybrid dead reckoning.** AI-IMU learns the noise covariances of an invariant EKF rather than the motion directly, letting the filter do the integration. Attention models have been applied to GPS-outage bridging in ground vehicles. GateIO replaces the filter during the outage instead of tuning it.

**Attention with length extrapolation.** GateIO uses ALiBi attention, which biases attention by distance and extrapolates to sequence lengths not seen in training. This matters because outage length varies.

**Dataset.** We use MARS-LVIG, a multi-sensor aerial dataset collected with a DJI M300 RTK platform. To our knowledge it has not been used for learned GPS-outage bridging before.

---

## 3. Data and Preprocessing

We use five flight sequences from MARS-LVIG: three from an airport site (gnss01, gnss02, gnss03) and two from an island site (island_gnss02, island_gnss03). RTK GPS gives ground-truth velocity.

Raw IMU data is cut into windows of 200 samples (1.0 s at 200 Hz) with a stride that gives a prediction every 0.1 s (10 Hz). Each window has 14 channels: accelerometer (3), gyroscope (3), attitude quaternion in body-to-navigation frame (4), GPS velocity (3, zeroed during an outage), and an outage flag (1).

A sequence is 300 windows (30 s). An outage is simulated by zeroing the GPS channels for 100 windows (10 s), starting at window 100. Targets are GPS velocity. Input channels and targets are scaled with median and inter-quartile range computed on the training set only; quaternion channels pass through unscaled so they stay valid rotations.

Two of the five flights (gnss03, island_gnss03) carry most of the turning. This matters for the evaluation, as we explain next.

---

## 4. Method

### 4.1 Architecture

GateIO has four parts:

1. **Window encoder.** Three 1D convolutions (kernels 7, 5, 3) with group normalisation, followed by attention pooling with a learned query, map each 200-sample window to a 48-dimensional token. This runs the rest of the model at 10 Hz instead of 200 Hz.
2. **Outage-step encoding.** A sinusoidal encoding of the number of steps since the outage began, reset on each GPS-aided step. The model conditions on how long it has been dead-reckoning, which is the quantity that drives drift.
3. **Temporal backbone.** A stack of dilated causal convolutions with a receptive field of about 25 s, followed by one layer of causal attention with ALiBi bias.
4. **Velocity head.** Two branches: one for GPS-aided steps and one for outage steps. The outage branch takes the token and the last known velocity.

The model has 186,390 parameters. A recurrent baseline, GateIO-LSTM, replaces the convolution-and-attention backbone with a two-layer LSTM and keeps everything else.

### 4.2 The yaw-rate gate

During an outage, on windows where the yaw rate is below 0.10 rad/s (straight cruise), a loss term pushes the predicted velocity change toward zero — that is, hold the last known velocity. Above the threshold (turns), the gate is off and the drift loss shapes the prediction. An earlier ungated version of this prior fought the drift loss on turns and lost. Gating is what lets the prior carry real weight.

### 4.3 Residual output (the fix that made it generalise)

An early version of GateIO predicted absolute velocity. It worked well within a flight but failed on held-out flights: on straight flight it drifted about 30 m where simply holding velocity drifted about 2 m. We traced this to two causes. First, the model produced a roughly constant velocity error of a few m/s on the forward axis and did not fall back to the last known velocity, which was almost exact. Second, the prior, as written, pulled the output toward the training-set median speed rather than toward the last known speed.

The fix is to predict a residual. The outage head outputs the *change* in velocity over the last known value, scaled by the natural size of that change. The default output (a zero residual) is exactly "hold the last velocity", so when the input shifts on an unseen flight the model degrades to persistence instead of to a biased guess. This is the same idea as displacement-increment prediction in the learned-odometry literature. With the residual head and a corrected prior, straight-flight error on held-out flights drops back toward the persistence floor.

### 4.4 Training

AdamW, learning rate 8e-4, one-cycle schedule, batch size 16, up to 200 epochs with early stopping. Turn windows are up-weighted so the model spends enough capacity on the harder dynamics. Loss terms: a data term on GPS-aided steps, a drift term on outage steps, the gated velocity-persistence prior, and small smoothness and cumulative-position terms.

---

## 5. Evaluation Protocol

This section is the core of the paper.

### 5.1 Two splits

**Within-flight split.** Each flight is cut 80/10/10 into train/validation/test in time order. Validation and test windows are the last parts of flights whose earlier parts are in training. Because a 30 s sequence and a 25 s receptive field are long relative to the split boundaries, and because flight dynamics change slowly, the validation and test sets are close to the training distribution. This is the split used in most early versions of this work and in a good deal of related work.

**Leave-one-flight-out (LOFO) cross-validation.** Whole flights are assigned to train, validation, and test. We run five folds; each flight is the test flight once, with three flights for training and one held-out flight for validation and model selection. No test window shares a flight with any training window.

### 5.2 Why the difference matters

Under the within-flight split, GateIO reaches about 7 m mean drift and selects its best checkpoint on validation drift. But we found that lower validation drift there could mean *worse* held-out-flight performance: the split rewards fitting flight-specific patterns. In other words, the within-flight split corrupts not only the reported number but also model selection. LOFO removes both problems, at the cost of a much smaller effective dataset (five folds, one flight each for validation and test).

---

## 6. Results

All numbers are endpoint drift in metres after a 10 s outage. Sequences are labelled STRAIGHT or TURN by their yaw activity during the outage. GateIO here uses the residual head.

### 6.1 Leave-one-flight-out (the honest result)

Pooled over 1,390 held-out sequences across all five flights:

| Group | GateIO | EKF | Const-v | GateIO % < 5 m |
|---|---:|---:|---:|---:|
| Straight (1200) | **20.8** | 55.9 | 41.4 | 48% |
| Turn (190) | **40.0** | 374.8 | 360.6 | 6% |
| **All (1390)** | **23.4** | 99.5 | 85.0 | 42% |

Per test flight (all sequences): gnss01 27.3, gnss02 13.1, gnss03 17.6, island_gnss02 23.9, island_gnss03 35.0.

GateIO cuts drift 3.6 times relative to the constant-velocity baseline and 4.2 times relative to the tuned EKF. The gain is 9 times on turns and 2 times on straight flight. It is worth noting that the constant-velocity baseline is far from perfect on straight flight here (41 m): even low-yaw sequences have real speed changes across flights, which is why a learned velocity model helps and pure persistence does not.

### 6.2 Within-flight (for contrast, not as the headline)

Under the within-flight split the same model reaches about 7 m mean drift, with straight-flight sequences near 1–2 m. We report this only to show how much the split inflates the number. The 3x gap between 7 m and 23 m is the size of the leakage.

### 6.3 Diagnosis

A substitution test isolates the cause. On held-out flights, if we replace the model's output with the last known velocity on gated straight windows only, straight-flight drift falls from about 30 m to under 1 m. So the loss was entirely in the learned straight-flight prediction, not the architecture, and not the turns. This is the observation that motivated the residual head in Section 4.3.

---

## 7. Discussion

GateIO beats classical dead reckoning on unseen flights, and it beats it most where the classical methods break: turns. That is a real and useful result. The learned model captures velocity changes on straight flight that a constant-velocity model misses, and it avoids the runaway turn error of a straight-line filter.

The honest cross-flight number is also a warning. The within-flight split, which is easy to set up and common, made the same model look three times better than it is. Anyone comparing GPS-outage methods on this kind of data should hold out whole flights.

We think the largest remaining error source is that the model has no way to say "I am unsure, hold the last velocity." The residual head gives it a safe default, but a learned uncertainty output, fused with a motion model, is the direction that has helped the most in related aerial work.

---

## 8. Limitations

- **Small dataset.** Five flights, two sites, one platform (DJI M300). Leave-one-flight-out gives five folds, each with a single validation and test flight, so per-flight estimates have low statistical power. This is the main limit on how far the results generalise.
- **Modest absolute accuracy.** 23 m mean drift and 42% of outages under 5 m is a research result, not a fielded navigation solution.
- **No uncertainty and no filter coupling.** GateIO gives a point estimate. Predicting uncertainty and fusing it with a filter is known to help and is not done here.
- **Weak learned baseline.** We compare to a constant-velocity model, a tuned EKF, and our own LSTM. We do not compare to published learned-odometry methods run on this data.
- **Single manoeuvre regime.** The turn/straight split is coarse and the turn set is small (190 sequences).

---

## 9. Conclusion

GateIO is a small causal model that bridges UAV GPS outages by predicting velocity from inertial data, held to the last known value on straight flight and free to react in turns. Tested the easy way, it looks very accurate; tested honestly, with whole flights held out, it reaches 23.4 m mean drift — 3.6 times better than the best classical baseline, but far from solved. We report the honest number, show why the common split inflates results, and point to more data and learned uncertainty as the next steps.

---

## References

- Brossard, M., Barrau, A., Bonnabel, S. (2020). AI-IMU dead-reckoning. *IEEE T-IV* 5(4).
- Chen, C., Lu, X., Markham, A., Trigoni, N. (2018). IONet: learning to cure the curse of drift in inertial odometry. *AAAI*.
- Cioffi, G., Bauersfeld, L., Kaufmann, E., Scaramuzza, D. (2023). Learned inertial odometry for autonomous drone racing. *IEEE RA-L* 8(5).
- Herath, S., Yan, H., Furukawa, Y. (2020). RoNIN: Robust neural inertial navigation in the wild. *ICRA*.
- Li, H., Zou, Y., Chen, N., Lin, J., et al. (2024). MARS-LVIG dataset. *IJRR* 43(8).
- Liu, W., Caruso, D., Ilg, E., et al. (2020). TLIO: Tight learned inertial odometry. *IEEE RA-L* 5(4).
- Press, O., Smith, N., Lewis, M. (2022). Train short, test long: ALiBi. *ICLR*.
- Qiu, Y., Xu, C., Chen, Y., et al. (2025). AirIO: learning inertial odometry with enhanced IMU feature observability. *arXiv:2501.15659*.
- Rao, B., Kazemi, E., Ding, Y., et al. (2022). CTIN: robust contextual transformer network for inertial navigation. *AAAI*.
