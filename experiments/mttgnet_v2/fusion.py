"""Four-source fusion: route the wide branch through the model's own fusion gate.

The clean model fuses ``temporal`` / ``periodic`` / ``memory`` contexts with a softmax
gate over sources (``MultiSourceGateFusion``, ``src/mttgnet.py:489``).  This module adds
the wide branch as a **fourth source** at the hidden level instead of adding a residual
at the output, so the decoder sees a single fused context and the gate decides per lead
how much of the window-based branch enters.

``MultiSourceGateFusion`` refuses ``num_sources`` other than 2 or 3, and the clean model
is hashed by the frozen benchmark, so the replacement is installed at run time:
``install_wide_fusion()`` builds a four-source module, copies every parameter whose
shape still matches, and zeroes the new column so the model starts close to the
unmodified fusion (with the fourth source contributing ~0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WideSourceFusion(nn.Module):
    """``MultiSourceGateFusion`` extended with a fourth, externally supplied source."""

    def __init__(self, hidden_dim: int, calendar_dim: int = 5):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.calendar_dim = int(calendar_dim)
        self.num_sources = 4
        self.gate_predictor = nn.Sequential(
            nn.Linear(self.calendar_dim + 1, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_sources))
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.pending_wide: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    def forward(self, temporal_ctx, periodic_ctx, memory_ctx, future_calendar):
        batch, horizon, _ = future_calendar.shape
        position = torch.linspace(0, 1, horizon, device=future_calendar.device)
        position = position.view(1, horizon, 1).expand(batch, horizon, 1)
        logits = self.gate_predictor(torch.cat([future_calendar, position], dim=-1))
        gate = torch.softmax(logits, dim=-1)
        if memory_ctx is None:
            memory_ctx = torch.zeros_like(temporal_ctx)
        wide_ctx = self.pending_wide
        if wide_ctx is None:
            wide_ctx = torch.zeros_like(temporal_ctx)
        if wide_ctx.shape != temporal_ctx.shape:
            raise ValueError(f"wide context {tuple(wide_ctx.shape)} != "
                             f"{tuple(temporal_ctx.shape)}")
        fused = (gate[..., 0:1] * temporal_ctx + gate[..., 1:2] * periodic_ctx
                 + gate[..., 2:3] * memory_ctx + gate[..., 3:4] * wide_ctx)
        self.last_gate = gate
        return self.norm(fused), gate


def install_wide_fusion(wrapper: nn.Module, wide_logit_init: float = -2.0) -> WideSourceFusion:
    """Swap ``wrapper.model.fusion`` for a four-source module, migrating weights.

    ``wide_logit_init`` is the initial logit of the fourth (wide) source.  ``-2`` starts
    it at ~9 % of the softmax mass; ``0`` makes it compete on equal terms, which is the
    setting that actually lets the branch train.
    """
    inner = getattr(wrapper, "model", None)
    old = getattr(inner, "fusion", None)
    if inner is None or old is None:
        raise TypeError("install_wide_fusion expects the clean MTTGNet wrapper")
    first = old.gate_predictor[0]
    hidden_dim = int(first.out_features)
    calendar_dim = int(first.in_features) - 1
    new = WideSourceFusion(hidden_dim, calendar_dim)
    sources = int(getattr(old, "num_sources", 3))

    old_first, new_first = old.gate_predictor[0], new.gate_predictor[0]
    if old_first.weight.shape == new_first.weight.shape:
        new_first.load_state_dict(old_first.state_dict())
    old_head, new_head = old.gate_predictor[2], new.gate_predictor[2]
    with torch.no_grad():
        new_head.weight.zero_()
        new_head.bias.zero_()
        new_head.weight[:sources] = old_head.weight
        new_head.bias[:sources] = old_head.bias
        # the fourth (wide) source starts at the requested weight
        new_head.bias[sources] = float(wide_logit_init)
        new.norm.load_state_dict(old.norm.state_dict())

    device = next(inner.parameters()).device
    inner.fusion = new.to(device)
    return inner.fusion
