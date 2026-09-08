"""
GateIO — model architecture for UAV GPS-outage bridging.

GateIO predicts GPS velocity from IMU data during GPS outages and integrates the
predictions to estimate position. The distinguishing contribution is a
yaw-rate-gated velocity-persistence prior (``L_cvprior``, defined in the training
script, not here): during straight-flight windows (|omega_z| < 0.10 rad/s) the model is
penalised for predicting velocity *changes*, while the gate switches off during
turns so the drift/prediction loss can teach turning dynamics without interference.

This module contains only the network. Two variants share every component except
the temporal backbone:

    GateIO      — TCN backbone + ALiBi causal attention (the main model)
    GateIOLSTM  — 2-layer causal LSTM backbone (the recurrent baseline)

Shared components:
    OutageStepPE          sinusoidal encoding of *elapsed outage steps*
    WindowEncoder         per-window 1-D conv stack + attention pooling -> token
    TCNBlock/TCNBackbone  dilated causal temporal conv network
    ALiBiCausalAttention  causal MHA with linear (ALiBi) position bias
    VelocityHead          dual-branch (GPS-aided vs dead-reckoning) head

Checkpoint compatibility
------------------------
The published R20 checkpoint (``marsnet_r20_final.pt``) was trained when the model
class was named ``MARSNet``. State-dict keys depend only on *attribute* names, not
the class name, so the attribute layout here is byte-for-byte compatible:

    window_enc.*        pos_enc.pe        tcn.net.*        attn.*        head.*

The class is aliased as ``MARSNet = GateIO`` (and ``MARSNetLSTM = GateIOLSTM``) so
old code and pickles that reference the former name still resolve.

Load a checkpoint with (PyTorch >= 2.6 requires weights_only=False because these
checkpoints bundle numpy arrays alongside the weights)::

    from models.gateio import GateIO
    model = GateIO()
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants (frozen R14–R20; do not change without retraining)
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL: int = 48          # token / hidden width throughout the network
N_CHAN: int = 14           # channels the WindowEncoder consumes at inference
N_IMU_CHAN: int = 10       # IMU-only channels (accel, gyro, quat) = X[..., 0:10]
SEQ_LEN: int = 300         # windows per sequence (30 s at 10 Hz)
WIN_LEN: int = 200         # IMU samples per window (1 s at 200 Hz)
DT: float = 0.1            # seconds between predictions (10 Hz)

N_HEADS: int = 4           # attention heads (ALiBi and pooling)
N_TCN_STACKS: int = 2      # dilation stacks in the TCN backbone
TCN_DILATIONS = (1, 2, 4, 8, 16)   # per-stack dilation schedule
DROPOUT: float = 0.15

# LSTM-baseline backbone
LSTM_HIDDEN: int = 128
LSTM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.15

# v2 residual head: floor on the per-axis velocity-increment scale (DV_IQR_TRUE).
# The vertical increment scale is ~8e-4 m/s; without a floor, dividing by it amplifies
# vertical residual errors ~1000x. Applied in both the head and the training loss.
DV_SCALE_FLOOR: float = 0.02

# Receptive field of the TCN backbone, in tokens:
#   N_TCN_STACKS * n_convs_per_block(=2) * sum(dilations) + 1
#   = 2 * 2 * (1+2+4+8+16) + 1 ... using the block's 2 causal convs of kernel 3.
# The commonly quoted figure 2*4*31 + 1 = 249 tokens (24.9 s at 10 Hz) counts the
# effective kernel span (kernel_size-1 = 2) per conv across both stacks.


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────
class OutageStepPE(nn.Module):
    """Sinusoidal encoding of the number of steps *elapsed since the outage began*.

    Unlike absolute positional encoding, this counts up only while the outage flag
    is set and resets to zero on each GPS-aided step. This lets the model condition
    on "how long have I been dead-reckoning" — the quantity that governs drift —
    rather than on absolute time within the sequence.
    """

    def __init__(self, d_model: int = D_MODEL, max_steps: int = 211) -> None:
        super().__init__()
        pe = torch.zeros(max_steps, d_model)
        pos = torch.arange(max_steps).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, tokens: torch.Tensor, outage_flag: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S, d_model)   outage_flag: (B, S) in [0, 1]
        flag = (outage_flag > 0.5).long()
        cumsum = flag.cumsum(dim=1)
        reset = cumsum * (1 - flag)                       # holds the count at last aided step
        steps = (cumsum - reset.cummax(dim=1).values).clamp(0, self.pe.shape[0] - 1)
        return tokens + self.pe[steps]


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Convolutional Network backbone
# ─────────────────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    """Two causal dilated 1-D convolutions with LayerNorm, GELU, and a residual.

    Causality is enforced by left-padding the input by ``(kernel_size-1)*dilation``
    and using zero padding in the conv itself, so no future token leaks into the
    current one.
    """

    def __init__(self, d_model: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=0)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, S)
        res = x
        x = self.act(self.norm1(self.conv1(F.pad(x, (self.pad, 0))).transpose(1, 2)).transpose(1, 2))
        x = self.drop(x)
        x = self.act(self.norm2(self.conv2(F.pad(x, (self.pad, 0))).transpose(1, 2)).transpose(1, 2))
        return self.drop(x) + res


class TCNBackbone(nn.Module):
    """Stacked dilated causal TCN.

    ``N_TCN_STACKS`` repetitions of the dilation schedule ``TCN_DILATIONS`` give an
    exponentially growing receptive field (~25 s of context) while every layer
    stays strictly causal.

    Note the attribute name ``self.net``: the published checkpoint stores these
    weights under ``tcn.net.*``, so this name must not change.
    """

    DILATIONS = list(TCN_DILATIONS)

    def __init__(self, d_model: int = D_MODEL, n_stacks: int = N_TCN_STACKS,
                 kernel_size: int = 3, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[TCNBlock(d_model, d, kernel_size, dropout)
              for _ in range(n_stacks) for d in self.DILATIONS]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model) -> (B, S, d_model)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Causal attention with ALiBi position bias
# ─────────────────────────────────────────────────────────────────────────────
class ALiBiCausalAttention(nn.Module):
    """Single-layer multi-head self-attention, causal, with ALiBi linear bias.

    ALiBi (Attention with Linear Biases) adds a per-head, distance-proportional
    penalty to the attention logits instead of learned positional embeddings, which
    extrapolates cleanly to outage lengths not seen at train time. A causal mask
    (upper triangle set to -inf) prevents attending to future tokens.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1).float() / n_heads)
        self.register_buffer("slopes", slopes)

    def _bias(self, S: int, dev: torch.device) -> torch.Tensor:
        pos = torch.arange(S, device=dev).float()
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
        return (-self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)) + causal.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        res = x
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_head)
        Q, K, V = qkv.unbind(2)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attn = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale + self._bias(S, x.device), dim=-1
        ).nan_to_num(0.0)
        out = self.drop(attn).matmul(V).transpose(1, 2).reshape(B, S, -1)
        return res + self.drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# Per-window encoder
# ─────────────────────────────────────────────────────────────────────────────
class WindowEncoder(nn.Module):
    """Encode one IMU window (``WIN_LEN`` samples x channels) into a single token.

    A three-layer 1-D conv stack (kernels 7, 5, 3) with GroupNorm extracts local
    features, then a learned CLS query attention-pools over the time axis to a
    fixed ``d_model`` vector. Producing one token per window keeps the downstream
    temporal model operating at the 10 Hz prediction rate rather than 200 Hz.
    """

    def __init__(self, in_channels: int = N_CHAN, d_model: int = D_MODEL) -> None:
        super().__init__()
        g = min(8, d_model // 6)
        self.conv1 = nn.Conv1d(in_channels, d_model, 7, padding=3)
        self.norm1 = nn.GroupNorm(g, d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.norm2 = nn.GroupNorm(g, d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm3 = nn.GroupNorm(g, d_model)
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*S, in_channels, WIN_LEN)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        x = self.drop(x.transpose(1, 2))                  # (B*S, WIN_LEN, d_model)
        out, _ = self.pool(self.cls.expand(x.size(0), -1, -1), x, x)
        return out.squeeze(1)                             # (B*S, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Output head
# ─────────────────────────────────────────────────────────────────────────────
class VelocityHead(nn.Module):
    """Dual-branch velocity head with additive v_prev fusion.

    Two independent MLP branches specialise for the two regimes:

        head_aided  — GPS available: read velocity straight off the token
        head_dr     — outage (dead reckoning): predict from the token *plus* a
                      projection of the last-known velocity ``v_prev``

    The outage flag ``alpha in [0,1]`` blends them, and ``v_prev`` is injected into
    the token only for the DR branch (scaled by ``alpha``). Feeding ``v_prev`` in
    lets the DR branch anchor on the last GPS velocity — the physical basis of the
    constant-velocity prior.
    """

    def __init__(self, d_model: int = D_MODEL, dropout: float = DROPOUT,
                 persistence_residual: bool = False) -> None:
        super().__init__()
        h = d_model // 2

        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, h),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 3),
            )

        self.head_aided = _branch()
        self.head_dr = _branch()
        self.v_prev_proj = nn.Linear(3, d_model)

        # ── v2: predict a residual over v_prev (default OFF = original v1 head) ──
        # When enabled, the dead-reckoning branch outputs
        #     y_norm = v_prev_norm + (dv_scale / y_iqr) * r
        # so r is a unit-scale velocity *increment* and r = 0 reproduces persistence
        # (holding the last GPS velocity). y_med / y_iqr are the absolute-velocity
        # normalisation stats; dv_scale is the natural per-axis increment scale
        # (DV_IQR_TRUE). Set them with set_normalization() before training/inference.
        # persistent=False: these normalisation constants stay out of the state_dict,
        # so v1 checkpoints (which lack them) still load, and v2 restores them from the
        # checkpoint's top-level DV_* fields via set_normalization().
        self.persistence_residual = persistence_residual
        self.register_buffer("y_med", torch.zeros(3), persistent=False)
        self.register_buffer("y_iqr", torch.ones(3), persistent=False)
        self.register_buffer("dv_scale", torch.ones(3), persistent=False)

    def set_normalization(self, y_med, y_iqr, dv_scale) -> None:
        """Load the velocity normalisation stats used by the residual parametrisation.

        ``dv_scale`` is floored at DV_SCALE_FLOOR so a pathologically small per-axis
        increment scale (the vertical axis is ~8e-4 m/s) cannot blow up the residual.
        The same floor is applied in the loss, so training and inference agree.
        """
        self.y_med = torch.as_tensor(y_med, dtype=torch.float32, device=self.y_med.device)
        self.y_iqr = torch.as_tensor(y_iqr, dtype=torch.float32, device=self.y_iqr.device)
        self.dv_scale = torch.as_tensor(dv_scale, dtype=torch.float32,
                                        device=self.dv_scale.device).clamp_min(DV_SCALE_FLOOR)

    def forward(self, tokens: torch.Tensor, outage_flag: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
        alpha = outage_flag.unsqueeze(-1)                 # (B, S, 1)
        aided = self.head_aided(tokens)
        dr_raw = self.head_dr(tokens + alpha * self.v_prev_proj(v_prev))
        if self.persistence_residual:
            # v_prev is physical (m/s); express the DR output as an absolute-velocity
            # prediction built as persistence + a scaled increment.
            v_prev_norm = (v_prev - self.y_med) / self.y_iqr
            dr = v_prev_norm + (self.dv_scale / self.y_iqr) * dr_raw
        else:
            dr = dr_raw
        return (1.0 - alpha) * aided + alpha * dr


# ─────────────────────────────────────────────────────────────────────────────
# LSTM baseline backbone
# ─────────────────────────────────────────────────────────────────────────────
class LSTMBackbone(nn.Module):
    """Unidirectional 2-layer LSTM replacing TCNBackbone + ALiBiCausalAttention.

    Causal by construction (each step sees only past context). The output is
    projected back to ``d_model`` so ``VelocityHead`` is reused unchanged. This is
    the only structural difference between GateIO and GateIOLSTM, isolating the
    contribution of the TCN+attention backbone.
    """

    def __init__(self, d_model: int = D_MODEL, hidden: int = LSTM_HIDDEN,
                 n_layers: int = LSTM_LAYERS, dropout: float = LSTM_DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            # bidirectional stays False — a causal DR model must not see the future.
        )
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        # src_key_padding_mask accepted for API parity; unused (LSTM is inherently causal).
        out, _ = self.lstm(x)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Full models
# ─────────────────────────────────────────────────────────────────────────────
class GateIO(nn.Module):
    """GateIO main model: WindowEncoder -> OutageStepPE -> TCN -> ALiBi attn -> head.

    forward(x, outage_flag, v_prev):
        x           (B, S, WIN_LEN, N_CHAN)  per-window features. Channels are
                    [IMU 0:10 | v_prev 10:13 | outage_flag 13] at inference time.
        outage_flag (B, S)                   in [0, 1]; fractional ramp allowed.
        v_prev      (B, S, 3)                last-known GPS velocity (physical, Y-space).
    returns:        (B, S, 3)                predicted GPS velocity in normalised
                    (Y_iqr) space; denormalise with y * Y_iqr + Y_median.
    """

    def __init__(self, persistence_residual: bool = False) -> None:
        super().__init__()
        self.window_enc = WindowEncoder(N_CHAN, D_MODEL)
        self.pos_enc = OutageStepPE(D_MODEL)
        self.tcn = TCNBackbone(D_MODEL, N_TCN_STACKS)
        self.attn = ALiBiCausalAttention(D_MODEL, N_HEADS, DROPOUT)
        self.head = VelocityHead(D_MODEL, DROPOUT, persistence_residual=persistence_residual)

    def set_normalization(self, y_med, y_iqr, dv_scale) -> None:
        """Forward velocity normalisation stats to the residual head (v2 only)."""
        self.head.set_normalization(y_med, y_iqr, dv_scale)

    def forward(self, x: torch.Tensor, outage_flag: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
        B, S, W, C = x.shape
        tokens = self.window_enc(x.reshape(B * S, W, C).permute(0, 2, 1).contiguous()).view(B, S, -1)
        tokens = self.pos_enc(tokens, outage_flag)
        tokens = self.tcn(tokens)
        tokens = self.attn(tokens)
        return self.head(tokens, outage_flag, v_prev)


class GateIOLSTM(nn.Module):
    """GateIO LSTM baseline: identical to GateIO but with an LSTM backbone.

    Same forward signature as :class:`GateIO`. Shares WindowEncoder, OutageStepPE,
    and VelocityHead verbatim; swaps TCN+attention for :class:`LSTMBackbone`.
    """

    def __init__(self, persistence_residual: bool = False) -> None:
        super().__init__()
        self.window_enc = WindowEncoder(N_CHAN, D_MODEL)
        self.pos_enc = OutageStepPE(D_MODEL)
        self.backbone = LSTMBackbone()
        self.head = VelocityHead(D_MODEL, DROPOUT, persistence_residual=persistence_residual)

    def set_normalization(self, y_med, y_iqr, dv_scale) -> None:
        """Forward velocity normalisation stats to the residual head (v2 only)."""
        self.head.set_normalization(y_med, y_iqr, dv_scale)

    def forward(self, x: torch.Tensor, outage_flag: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
        B, S, W, C = x.shape
        tokens = self.window_enc(x.reshape(B * S, W, C).permute(0, 2, 1).contiguous()).view(B, S, -1)
        tokens = self.pos_enc(tokens, outage_flag)
        tokens = self.backbone(tokens)
        return self.head(tokens, outage_flag, v_prev)


# Backwards-compatible aliases: the R20/LSTM checkpoints were trained under these
# class names. State-dict keys are unaffected (they use attribute names).
MARSNet = GateIO
MARSNetLSTM = GateIOLSTM


if __name__ == "__main__":
    # Quick self-test: build both models, run a forward pass, print param counts
    # and verify the output shape. No dataset or checkpoint required.
    torch.manual_seed(42)
    B, S = 2, SEQ_LEN
    x = torch.randn(B, S, WIN_LEN, N_CHAN)
    outage_flag = torch.zeros(B, S)
    outage_flag[:, S // 3: S // 3 + 100] = 1.0            # 10 s outage
    v_prev = torch.randn(B, S, 3)

    for name, ctor, expected in [("GateIO", GateIO, 186_390), ("GateIOLSTM", GateIOLSTM, None)]:
        model = ctor().eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            y = model(x, outage_flag, v_prev)
        assert y.shape == (B, S, 3), f"{name}: bad output shape {tuple(y.shape)}"
        tag = f"  (paper: {expected:,})" if expected else ""
        print(f"{name:<11} params={n_params:>8,}{tag}  output={tuple(y.shape)}  OK")

    # v2 residual head: same parameter count, and a zeroed DR branch must reproduce
    # persistence exactly (output_norm == v_prev_norm) on outage windows.
    y_med = torch.tensor([-0.146, 0.820, 0.003])
    y_iqr = torch.tensor([1.516, 7.985, 0.100])
    dv_scale = torch.tensor([0.055, 0.319, 0.00078])
    m2 = GateIO(persistence_residual=True).eval()
    m2.set_normalization(y_med, y_iqr, dv_scale)
    assert sum(p.numel() for p in m2.parameters()) == 186_390, "v2 changed param count"
    # zero the DR branch's final layer so dr_raw == 0 -> output must equal persistence
    with torch.no_grad():
        m2.head.head_dr[-1].weight.zero_(); m2.head.head_dr[-1].bias.zero_()
        vprev_phys = torch.randn(B, S, 3)
        full_outage = torch.ones(B, S)
        out = m2(x, full_outage, vprev_phys)
        vprev_norm = (vprev_phys - y_med) / y_iqr
    err = (out - vprev_norm).abs().max().item()
    assert err < 1e-5, f"residual default != persistence (max err {err})"
    print(f"GateIO v2   params= 186,390  residual default == persistence (max err {err:.1e})  OK")
    print("Self-test passed.")
