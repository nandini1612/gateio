# Results

Per-sequence evaluation outputs and the figures used in the README and preprint.
All numbers come from the trained checkpoints on the **v2** dataset; the headline
tables are in the top-level [README](../README.md).

| File | Description |
|---|---|
| `val_all_results.csv` | Per-sequence val drift (GateIO, LSTM, EKF, Const-v), 59 seqs |
| `test_all_results.csv` | Per-sequence test drift, 59 seqs |
| `paths/{val,test}_path_seq{41,47,53}.npy` | Trajectory arrays (pred/true) for the turn, false-alarm, and long-outage examples |
| `figures/fig1_trajectories_test.png` | Dead-reckoned test trajectories |
| `figures/fig2_group_bars.png` | Per-group mean drift, val vs test |
| `figures/fig3_drift_cdf.png` | Cumulative distribution of drift, val vs test |
| `figures/architecture.svg` | GateIO architecture diagram |
| `figures/gate.svg` | The yaw-rate gate |

## Regenerate

```bash
python eval/evaluate.py --data path/to/MARS_Master_Dataset.npz \
    --gateio-ckpt path/to/marsnet_r20_final.pt \
    --lstm-ckpt   path/to/marsnet_lstm_best.pt \
    --split val --out-dir results
```

Swap `--split test` for the test CSV. The figures are built from these CSVs and the
`paths/*.npy` arrays.
