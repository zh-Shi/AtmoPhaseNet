"""Final model with the dedicated CO2-conditioning module removed.

The factory reuses the frozen implementation; no historical training source is
modified.  CO2 inputs are already (ppm-280)/920, and target outputs remain in
training-standardized units until the caller inverts its target scaler.
"""
from __future__ import annotations

import inspect

import torch

from experiments.mttgnet_v2.wide import MTTGNetWideWrapper
from src.mttgnet_test import MTTGNetTestWrapper


DEFAULT_MODEL_PARAMS = dict(
    hidden_dim=64, dropout=0.25, num_gru_layers=2, use_memory=False,
    use_var_gnn=True, use_variable_encoder=True, use_shift_encoder=False,
    use_time_bias=True, use_gate=True, use_calendar=True,
    var_encoder_type="gat", query_refine=True, mem_clean=True,
    decoder_bypass=False, time_bias_mode="additive", co2_mode="counterfactual",
    gnn_dynamic=False, fusion_sources=3, use_delta_head=True,
    wide_mode="blend", wide_zero_init=False, wide_gate_init_bias=0.0,
    wide_decay_leads=0.0,
)


class LatestMainModel(MTTGNetWideWrapper):
    """Current-state/direct blend without a registered shift encoder.

    ``mask_co2`` removes the channel from both the graph/retrieval branch and
    direct MLP.  Merely disabling the shift encoder is not an all-CO2 ablation.
    The retained constant channel preserves shape and is not a capacity control.
    """

    def __init__(self, *, mask_co2=False, **kwargs):
        if kwargs.get("use_shift_encoder", False):
            raise ValueError("The dedicated CO2-conditioning module is removed in final_noshift")
        kwargs["use_shift_encoder"] = False
        super().__init__(**kwargs)
        self.mask_co2 = bool(mask_co2)
        self.input_co2_index = kwargs.get("co2_index", 1)
        self.input_calendar_dim = int(kwargs.get("calendar_dim", 5))
        self.output_horizon = int(kwargs["horizon"])
        if self.model.shift_encoder is not None:
            raise AssertionError("Removed conditioning module must not register parameters")
        if self.mask_co2 and self.model.use_shift_encoder:
            raise ValueError("All-path CO2 removal also requires use_shift_encoder=False")

    def forward(self, x_recent, daily_anchor=None, yearly_anchor=None,
                daily_mask=None, yearly_mask=None, future_calendar=None,
                daily_gaps=None, yearly_gaps=None, co2_override_ppm=None,
                future_co2=None, **kwargs):
        if co2_override_ppm is not None:
            raise ValueError("No dedicated CO2 override remains; provide observed/prescribed input CO2")
        if future_calendar is None:
            future_calendar = x_recent.new_zeros(
                x_recent.shape[0], self.output_horizon, self.input_calendar_dim)
        if self.mask_co2:
            index = self.input_co2_index
            if index is None or not 0 <= index < x_recent.shape[-1]:
                raise ValueError("CO2 mask has no valid input channel")
            def masked(value):
                if value is None:
                    return None
                result = value.clone()
                result[..., index] = 0.0
                return result
            x_recent, daily_anchor, yearly_anchor = map(
                masked, (x_recent, daily_anchor, yearly_anchor))
            if future_co2 is not None:
                future_co2 = torch.zeros_like(future_co2)
            co2_override_ppm = None
        return super().forward(
            x_recent=x_recent, daily_anchor=daily_anchor, yearly_anchor=yearly_anchor,
            daily_mask=daily_mask, yearly_mask=yearly_mask,
            future_calendar=future_calendar, daily_gaps=daily_gaps,
            yearly_gaps=yearly_gaps, co2_override_ppm=co2_override_ppm,
            future_co2=future_co2, **kwargs)


def build_latest_model(*, num_features, seq_len, horizon, calendar_dim=5,
                       co2_index=1, target_index=0, model_params=None,
                       scaler_mean=None, scaler_std=None, use_future_co2=False,
                       device=None):
    """Build the no-conditioning final or a remaining structural control.

    Scaler arguments are accepted for adapter compatibility, never applied here.
    ``num_features`` includes any caller-added time encodings.  Calendar width is
    the actual pre-CO2 calendar width, not a hard-coded global-weather schema.
    A no-direct control sets ``direct_path=False``; this is distinct from an
    unrelated standalone MLP baseline.
    """
    params = dict(DEFAULT_MODEL_PARAMS)
    params.update(model_params or {})
    if params.get("use_shift_encoder") is not False:
        raise ValueError("final_noshift requires use_shift_encoder=False for every experiment")
    params.pop("arch", None)
    if "calendar_dim" in params:
        if int(params.pop("calendar_dim")) != int(calendar_dim):
            raise ValueError("Calendar dimensions disagree")
    if "use_future_co2" in params:
        value = bool(params.pop("use_future_co2"))
        if use_future_co2 and not value:
            raise ValueError("Future CO2 switches disagree")
        use_future_co2 = value
    direct = bool(params.pop("direct_path", True))
    mask_co2 = bool(params.pop("mask_co2", False))
    wide = {key: params.pop(key) for key in list(params) if key.startswith("wide_")}
    accepted = set(inspect.signature(MTTGNetTestWrapper.__init__).parameters)
    accepted |= {"co2_index", "co2_min", "co2_max", "fusion_sources",
                 "co2_attention_modulation", "co2_additive_bias"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(f"Unforwarded model parameters: {sorted(unknown)}")
    if not 0 <= int(target_index) < int(num_features):
        raise ValueError("Target channel is out of range")
    if co2_index is not None and not 0 <= int(co2_index) < int(num_features):
        raise ValueError("CO2 channel is out of range")
    kwargs = dict(num_features=int(num_features), seq_len=int(seq_len),
                  horizon=int(horizon), calendar_dim=int(calendar_dim),
                  co2_index=co2_index, target_index=int(target_index),
                  use_future_co2=bool(use_future_co2), **params)
    if direct:
        model = LatestMainModel(mask_co2=mask_co2, **wide, **kwargs)
    else:
        if mask_co2:
            raise ValueError("Use the explicit all-path mask with direct_path=True")
        model = MTTGNetTestWrapper(**kwargs)
    return model.to(device) if device is not None else model


def build_model(cfg, trainer):
    """Trainer adapter used by the weather worker and checkpoint diagnostics."""
    return build_latest_model(
        num_features=trainer.num_features,
        seq_len=cfg["dataset"]["recent_length"], horizon=cfg["dataset"]["horizon"],
        calendar_dim=cfg.get("model_params", {}).get("calendar_dim", 5),
        co2_index=cfg["data"].get("co2_index"),
        target_index=cfg["data"]["target_index"],
        model_params=cfg.get("model_params"),
        use_future_co2=cfg["dataset"].get("use_future_co2", False),
        device=trainer.device)
