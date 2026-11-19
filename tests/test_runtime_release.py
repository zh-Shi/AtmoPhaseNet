"""Exercise the exported final two-path configurations and recursive inference contract."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.cmip6.run_cmip6_pure_ml import build_model as build_cmip6_model
from scripts.run_multiseed import build_model
from src.scaler import StandardScaler
from src.scenario_rollout import autoregressive_project

ROOT = Path(__file__).resolve().parents[1]


def test_weather_example_models_forward_and_batch_independence():
    expected_params = {'example_h28': 441_990, 'example_h120': 459_746}
    for name in ('example_h28', 'example_h120'):
        cfg = yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text(encoding='utf-8'))
        model = build_model(cfg, SimpleNamespace(num_features=10, device=torch.device('cpu'))).eval()
        assert sum(p.numel() for p in model.parameters()) == expected_params[name]
        assert model.model.use_shift_encoder is False
        assert model.model.shift_encoder is None
        assert model.model.use_delta_head is True
        assert model.wide_mode == 'blend'
        assert model.model.fusion.num_sources == 3
        h = cfg['dataset']['horizon']
        torch.manual_seed(123)
        batch = dict(
            x_recent=torch.rand(2, 120, 10),
            daily_anchor=torch.rand(2, h, 7, 10),
            yearly_anchor=torch.rand(2, h, 3, 10),
            daily_mask=torch.ones(2, h, 7, dtype=torch.bool),
            yearly_mask=torch.ones(2, h, 3, dtype=torch.bool),
            future_calendar=torch.rand(2, h, 5),
            daily_gaps=torch.full((2, h, 7), 7.0),
            yearly_gaps=torch.full((2, h, 3), 365.0),
        )
        with torch.no_grad():
            prediction = model(**batch)['prediction']
            single = model(**{k: v[:1] for k, v in batch.items()})['prediction']
        assert prediction.shape == (2, h) and torch.isfinite(prediction).all()
        torch.testing.assert_close(single[0], prediction[0], atol=1e-5, rtol=1e-5)


def test_monthly_century_example_builds_without_shift_encoder():
    cfg = yaml.safe_load((ROOT / 'configs/example_century.yaml').read_text(encoding='utf-8'))
    model = build_cmip6_model(cfg, num_features=2).eval()
    assert model.model.use_shift_encoder is False
    assert model.model.shift_encoder is None
    assert model.model.use_delta_head is True
    assert model.wide_mode == 'blend'
    torch.manual_seed(123)
    batch = dict(
        x_recent=torch.rand(1, 60, 2),
        daily_anchor=None,
        yearly_anchor=torch.rand(1, 120, 3, 2),
        yearly_mask=torch.ones(1, 120, 3, dtype=torch.bool),
        future_calendar=torch.rand(1, 120, 5),
        future_co2=torch.full((1, 120), 0.25),
    )
    with torch.no_grad():
        pred = model(**batch)['prediction']
    assert pred.shape == (1, 120) and torch.isfinite(pred).all()


def test_rollout_ignores_future_temperature_and_reuses_predictions():
    class Model:
        def __call__(self, **inputs):
            # Forecast depends only on the latest model-space temperature.
            x = inputs['x_recent']
            return {'prediction': x[:, -1, 0, None].repeat(1, 3) + 1.0}

    ts = pd.date_range('2014-01-15', periods=18, freq=pd.DateOffset(months=1))
    raw = np.column_stack([np.arange(18), np.full(18, 400.0)]).astype(np.float32)
    scaler = StandardScaler(mean=np.array([0.0, 400.0]), std=np.array([1.0, 1.0]))

    def project(values):
        return autoregressive_project(
            Model(), 11, np.full(6, 410.0), values, ts, scaler,
            torch.device('cpu'), L=6, H=3, advance=2,
            future_co2_ppm=np.full(6, 410.0), yearly_anchors=1,
        )

    first = project(raw)
    changed = raw.copy()
    changed[12:, 0] = 999999.0
    np.testing.assert_array_equal(first, project(changed))
    np.testing.assert_array_equal(first, [12.0, 12.0, 13.0, 13.0, 14.0, 14.0])
