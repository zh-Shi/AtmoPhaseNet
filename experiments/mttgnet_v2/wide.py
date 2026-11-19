"""Standalone "wide skip" branch for the clean MTTGNet (all-baselines-v1 protocol).

Motivation (measured on outputs/all_baselines_v1, H=120, seed-fixed comparison):

* the plain MLP baseline (flatten 120x10 -> 128 -> 128 -> Linear(H)) beats MTTGNet
  by 0.0077 K overall RMSE;
* that gap is not distributed evenly -- leads 1-12 (<= 3 d) carry 44 % of the squared
  error gap at a 1.264 RMSE ratio, leads 13-60 carry 52 % at 1.032, and leads 61-120
  (15-30 d) carry 3.6 % at 1.001, i.e. MTTGNet already matches the MLP there;
* MTTGNet's first-epoch validation loss is 0.0238 versus the MLP's 0.0068: the deep
  stack is hard to optimise from a cold start.

The fix is the standard linear/wide skip: a faithful copy of the winning MLP runs in
parallel with the deep stack and is fused per lead.  Initialised at zero, the model
starts exactly at the unmodified MTTGNet and learns only the residual the deep stack
does not explain.

This module deliberately lives outside the files hashed by the frozen benchmark
(``src/*.py``, ``experiments/baselines/*.py``, ``experiments/utils.py``), so the
all-baselines-v1 run stays valid and the baseline table stays untouched.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from src.mttgnet_test import MTTGNetTestWrapper

FUSION_MODES = ("add", "gated_add", "blend", "softmax", "linear", "fusion4")


class LinearTendencyBranch(nn.Module):
    """DLinear-style trend/seasonal extrapolation of the target channel.

    Physically this is the classic "tendency + persistence" term of operational
    forecasting: the window is split into a moving-average trend and the departure from
    it, and each part is mapped linearly to the horizon.  It plays the same role as the
    flat MLP branch but with ~7 K parameters (H=28) instead of ~180 K, and every
    parameter has a reading: the trend map says how the recent tendency persists, the
    seasonal map how the diurnal/annual departure recurs.
    """

    def __init__(self, seq_len: int, horizon: int, target_idx: int = 0, kernel: int = 25):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so the moving average keeps the length")
        self.seq_len, self.horizon = int(seq_len), int(horizon)
        self.target_idx, self.kernel = int(target_idx), int(kernel)
        self.trend_head = nn.Linear(self.seq_len, self.horizon)
        self.seasonal_head = nn.Linear(self.seq_len, self.horizon)

    def forward(self, x_recent: torch.Tensor,
                future_calendar: torch.Tensor | None = None) -> torch.Tensor:
        if x_recent.ndim != 3 or x_recent.shape[1] != self.seq_len:
            raise ValueError(f"LinearTendencyBranch expects [B, {self.seq_len}, F], "
                             f"got {tuple(x_recent.shape)}")
        target = x_recent[:, :, self.target_idx]                     # [B, L]
        trend = torch.nn.functional.avg_pool1d(
            target.unsqueeze(1), self.kernel, stride=1, padding=self.kernel // 2).squeeze(1)
        seasonal = target - trend
        return self.trend_head(trend) + self.seasonal_head(seasonal)  # [B, H]


class WideSkip(nn.Module):
    """Flattened recent window -> 2-layer MLP -> horizon (the MLP baseline's trunk).

    Same geometry as ``experiments.baselines.true_baselines.SimpleMLPForecaster``:
    ``flatten(L*F) -> Linear(128) -> GELU -> Dropout -> Linear(128)`` plus a pooled
    calendar context (``_CalendarContext`` in that module) and a ``Linear(128, H)``
    head.  Re-implemented here rather than imported so this package stays standalone
    and the frozen module is never touched at run time.
    """

    def __init__(self, num_features: int, seq_len: int, horizon: int, hidden: int = 128,
                 dropout: float = 0.1, calendar_dim: int = 5):
        super().__init__()
        for name, value in (("num_features", num_features), ("seq_len", seq_len),
                            ("horizon", horizon), ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.num_features, self.seq_len, self.horizon = num_features, seq_len, horizon
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.calendar = nn.Sequential(
            nn.Linear(calendar_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_recent: torch.Tensor,
                future_calendar: torch.Tensor | None = None) -> torch.Tensor:
        if x_recent.ndim != 3 or x_recent.shape[1] != self.seq_len \
                or x_recent.shape[2] != self.num_features:
            raise ValueError(
                f"WideSkip expects [B, {self.seq_len}, {self.num_features}], "
                f"got {tuple(x_recent.shape)}")
        ctx = self.trunk(x_recent)
        if future_calendar is not None and future_calendar.ndim == 3:
            # Pool over the horizon, exactly like the baseline's _CalendarContext.
            ctx = ctx + self.calendar(future_calendar.mean(dim=1))
        return self.head(ctx)


class MTTGNetWideWrapper(MTTGNetTestWrapper):
    """Clean MTTGNet plus a parallel wide branch, fused per lead.

    Fusion modes (``wide_mode``):

    ``add``        prediction = deep + wide
    ``gated_add``  prediction = deep + g_h * wide          (g_h in (0,1), per lead)
    ``blend``      prediction = (1-g_h) * deep + g_h * wide          (convex)
    ``softmax``    prediction = (1-w_h) * deep + w_h * wide, w = softmax over 2 logits
                   (same geometry as ``blend``, different parameterisation; kept so the
                   two can be compared directly)
    ``fusion4``    the wide branch enters the model's own ``MultiSourceGateFusion`` as a
                   fourth source; no output-level addition (see ``fusion.py``)

    The gate reads the per-lead calendar vector plus the normalised lead position, so
    short and long leads can weight the wide branch differently.  With
    ``wide_zero_init=True`` the head is zeroed and the model reproduces the unmodified
    MTTGNet at initialisation, which makes the "with/without skip" ablation exact.
    """

    def __init__(self, *, wide_mode: str = "gated_add", wide_hidden: int = 128,
                 wide_dropout: float | None = None, wide_zero_init: bool = True,
                 wide_gate_hidden: int = 64, wide_gate_init_bias: float = 0.0,
                 wide_fusion4_logit: float = -2.0, wide_decay_leads: float = 0.0,
                 **model_kwargs):
        super().__init__(**model_kwargs)
        if wide_mode not in FUSION_MODES:
            raise ValueError(f"wide_mode must be one of {FUSION_MODES}, got {wide_mode!r}")
        missing = [key for key in ("num_features", "seq_len", "horizon")
                   if key not in model_kwargs]
        if missing:
            raise ValueError(f"MTTGNetWideWrapper needs keyword arguments {missing}")
        self.wide_mode = wide_mode
        self.wide_use_calendar = bool(model_kwargs.get('use_calendar',True))
        self.wide_zero_init = bool(wide_zero_init)
        # Optional e-folding decay of the branch weight over lead time: a tendency /
        # extrapolation term only has skill over its persistence timescale, so the gate
        # should fade out at long leads instead of staying flat.
        self.wide_decay_leads = float(wide_decay_leads)
        # Debug/attribution hook: a float in [0, 1] forces the fusion gate everywhere,
        # which is how ``attribution.py`` measures "gate off" / "gate on" exactly
        # instead of inverting the mixture algebraically.
        self.force_gate = None
        calendar_dim = int(model_kwargs.get("calendar_dim", 5))
        if model_kwargs.get("use_future_co2"):
            calendar_dim += 1  # the wrapper appends the CO2 level to the calendar vector
        dropout = (float(model_kwargs.get("dropout", 0.15))
                   if wide_dropout is None else float(wide_dropout))
        if wide_mode == "linear":
            # Mechanistic branch: linear trend/seasonal extrapolation of the target
            # channel instead of a flat MLP over all channels.
            self.wide = LinearTendencyBranch(int(model_kwargs["seq_len"]),
                                             int(model_kwargs["horizon"]),
                                             target_idx=int(model_kwargs.get("target_index", 0)))
        else:
            self.wide = WideSkip(int(model_kwargs["num_features"]),
                                 int(model_kwargs["seq_len"]),
                                 int(model_kwargs["horizon"]), int(wide_hidden), dropout,
                                 calendar_dim)
        self.wide_gate = nn.Sequential(
            nn.Linear(calendar_dim + 1, int(wide_gate_hidden)),
            nn.GELU(),
            nn.Linear(int(wide_gate_hidden), 1),
        )
        nn.init.constant_(self.wide_gate[-1].bias, float(wide_gate_init_bias))
        # Two-way softmax over the branches is the convex combination with a different
        # parameterisation (logits instead of a sigmoid); kept separate so the two can be
        # compared directly.
        self.wide_logits = None
        if wide_mode == "softmax":
            self.wide_logits = nn.Sequential(
                nn.Linear(calendar_dim + 1, int(wide_gate_hidden)),
                nn.GELU(),
                nn.Linear(int(wide_gate_hidden), 2),
            )
            with torch.no_grad():
                self.wide_logits[-1].bias.copy_(
                    torch.tensor([0.0, float(wide_gate_init_bias)]))
        # Hidden-level alternative: the wide branch becomes the model fusion's 4th source.
        self.wide_ctx = None
        if wide_mode == "fusion4":
            from experiments.mttgnet_v2.fusion import install_wide_fusion

            hidden = int(model_kwargs.get("hidden_dim", 96))
            # Default init on purpose: a zeroed projector would cut the gradient path
            # from the wide head to the loss and the fourth source could never learn.
            self.wide_ctx = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            install_wide_fusion(self, wide_logit_init=float(wide_fusion4_logit))
        if self.wide_zero_init and hasattr(self.wide, "head"):
            nn.init.zeros_(self.wide.head.weight)
            nn.init.zeros_(self.wide.head.bias)

    def forward(self, x_recent, daily_anchor=None, yearly_anchor=None,
                daily_mask=None, yearly_mask=None, future_calendar=None,
                daily_gaps=None, yearly_gaps=None, co2_override_ppm=None,
                future_co2=None, **kwargs):
        batch = x_recent.shape[0]
        horizon = (future_calendar.shape[1] if future_calendar is not None
                   else int(self.wide.horizon))
        calendar = future_calendar
        if calendar is None:
            calendar = torch.zeros(batch, horizon, 5, device=x_recent.device,
                                   dtype=x_recent.dtype)
        if not self.wide_use_calendar:
            calendar = torch.zeros_like(calendar)
        if self.use_future_co2:
            if future_co2 is None:
                raise ValueError("use_future_co2=True but the batch carried no future_co2.")
            calendar = torch.cat(
                [calendar, future_co2.reshape(batch, horizon, 1).to(calendar.dtype)], dim=-1)
        wide = self.wide(x_recent, calendar)

        if self.wide_mode == "fusion4":
            # Hidden-level fusion: inject the projected wide branch before the model
            # runs, so the decoder consumes one fused context (no output addition).
            self.model.fusion.pending_wide = self.wide_ctx(wide.unsqueeze(-1))
            out = super().forward(
                x_recent=x_recent, daily_anchor=daily_anchor, yearly_anchor=yearly_anchor,
                daily_mask=daily_mask, yearly_mask=yearly_mask,
                future_calendar=future_calendar, daily_gaps=daily_gaps,
                yearly_gaps=yearly_gaps, co2_override_ppm=co2_override_ppm,
                future_co2=future_co2, **kwargs)
            gate = self.model.fusion.last_gate[..., 3:4].squeeze(-1)
            result = dict(out)
            result["wide_prediction"] = wide
            result["wide_gate"] = gate
            return result

        out = super().forward(
            x_recent=x_recent, daily_anchor=daily_anchor, yearly_anchor=yearly_anchor,
            daily_mask=daily_mask, yearly_mask=yearly_mask,
            future_calendar=future_calendar, daily_gaps=daily_gaps,
            yearly_gaps=yearly_gaps, co2_override_ppm=co2_override_ppm,
            future_co2=future_co2, **kwargs)
        base = out["prediction"]
        if base.ndim != 2:
            raise ValueError(f"Unexpected prediction shape {tuple(base.shape)}")
        if wide.shape != base.shape:
            raise ValueError(f"Wide branch shape {tuple(wide.shape)} != {tuple(base.shape)}")
        if self.wide_mode == "add":
            prediction = base + wide
            gate = torch.ones_like(wide)
        else:
            lead = torch.arange(horizon, device=x_recent.device, dtype=calendar.dtype)
            lead = (lead / max(horizon - 1, 1)).view(1, horizon, 1).expand(batch, horizon, 1)
            features = torch.cat([calendar, lead], dim=-1)
            if self.wide_mode == "softmax":
                weights = torch.softmax(self.wide_logits(features), dim=-1)
                gate = weights[..., 1]
                prediction = (1.0 - gate) * base + gate * wide
            else:                                    # gated_add / blend / linear
                gate = torch.sigmoid(self.wide_gate(features))[..., 0]
                prediction = ((base + gate * wide) if self.wide_mode == "gated_add"
                              else (1.0 - gate) * base + gate * wide)
        if self.force_gate is not None:
            gate = torch.full_like(wide, float(self.force_gate))
        if self.wide_decay_leads > 0 and self.wide_mode != "add":
            steps = torch.arange(horizon, device=x_recent.device, dtype=gate.dtype)
            gate = gate * torch.exp(-steps / self.wide_decay_leads).view(1, horizon)
        if self.wide_mode in ("gated_add", "blend", "linear", "softmax"):
            prediction = ((base + gate * wide) if self.wide_mode == "gated_add"
                          else (1.0 - gate) * base + gate * wide)
        result = dict(out)
        result["prediction"] = prediction
        result["wide_prediction"] = wide
        result["wide_gate"] = gate
        return result


@contextlib.contextmanager
def wide_wrapper_patch(wide_kwargs: dict | None):
    """Make the frozen ``arch: clean`` builder construct :class:`MTTGNetWideWrapper`.

    ``scripts/run_multiseed.build_model`` imports ``MTTGNetTestWrapper`` from
    ``src.mttgnet_test`` *inside* the function, so swapping the module attribute for
    the duration of the call reuses the whole parameter-plumbing code path instead of
    duplicating its 80-line ``kw`` dict here.  Nothing on disk is modified.
    """
    import src.mttgnet_test as clean_module

    if wide_kwargs is None:
        yield
        return
    original = clean_module.MTTGNetTestWrapper

    class _Patched(MTTGNetWideWrapper):
        def __init__(self, **model_kwargs):
            super().__init__(**wide_kwargs, **model_kwargs)

    clean_module.MTTGNetTestWrapper = _Patched
    try:
        yield
    finally:
        clean_module.MTTGNetTestWrapper = original


def build_clean_model(cfg: dict, trainer, *, wide_kwargs: dict | None = None):
    """Build the frozen clean MTTGNet, or its wide-skip variant when ``wide_kwargs``."""
    from scripts.run_multiseed import build_model as _build

    with wide_wrapper_patch(wide_kwargs):
        return _build(cfg, trainer)
