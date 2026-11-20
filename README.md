# AtmoPhaseNet: Multi-Period Graph-Temporal Modeling for Global-Mean Near-Surface Temperature Prediction

AtmoPhaseNet (Atmospheric Phase-Conditioned Forecasting Network; internal module name `mttgnet` retained for checkpoint compatibility) is a two-path neural architecture for synoptic-to-subseasonal prediction of global-mean near-surface air temperature. Its structured pathway combines recent-window graph-temporal encoding, calendar-aligned daily and annual historical retrieval, and current-state referencing (`y_t + delta`), while a parallel direct pathway maps the flattened recent multivariate window and pooled target-calendar features to absolute temperature across all forecast leads. A calendar- and lead-dependent sigmoid gate blends the two predictions in a single forward pass. Atmospheric CO2 enters as a standard multivariate input covariate without a dedicated CO2-conditioning branch.

Weather examples predict 6-hourly global-mean ERA5 2-m air temperature over 7 days
(`H=28`) and 30 days (`H=120`) using a unified hyperparameter and training recipe.
A separate monthly CMIP6 example supports conditional multi-decadal rollout under
prescribed CO2 pathways.

This is a compact runtime repository containing the model and training code, data
preparation scripts, conditional rollout inference, example configurations, and runtime
tests. Supply your own input data before training. Manuscript sources, figures,
checkpoints, evaluation outputs, private server tooling, and the full external baseline
and ablation matrix are kept outside this public repository.

## Install

Use Python 3.10 or later and a PyTorch build appropriate for your CPU or GPU:

```bash
pip install -r requirements.txt
# Optional: raw ERA5/CMIP6 preparation
pip install xarray netCDF4
# Optional: tests
pip install pytest
```

## Weather training

Prepare `data/processed/weather_values_D.npy` and
`data/processed/weather_timestamps_D.npy`, or update the data paths in the config.

```bash
python run_model.py configs/example_h28.yaml
python run_model.py configs/example_h120.yaml
python scripts/run_multiseed.py configs/example_h28.yaml --seeds 42,123,789
python scripts/run_multiseed.py configs/example_h120.yaml --seeds 42,123,789
```

Single-config runs write under `outputs/example_h28/` or `outputs/example_h120/`.
Multi-seed runs write `outputs/<config-stem>_s<seed>/` and a summary JSON.
Set `device: cpu`, `num_workers: 0`, `pin_memory: false`, and `use_amp: false` for
CPU runs. Both weather examples share the manuscript's final unified architecture
and training recipe (`hidden_dim=64`, `dropout=0.25`, `query_refine=true`,
`use_shift_encoder=false`, `use_delta_head=true`, `wide_mode=blend`,
`short_lead_loss_steps=8`, `short_lead_loss_weight=0.8`), yielding 441,990 parameters
at `H=28` and 459,746 parameters at `H=120`. GPU training is recommended for the full
1980-2024 record.

## Input contract and preparation

| File | Shape | Meaning |
|---|---|---|
| values | `[T, N]`, float32 | Physical variables, temperature in column 0 and CO2 in column 1 |
| timestamps | `[T]`, datetime64 | Matching chronological timestamps |

Dataset D column order is `[T2m, CO2, U10, V10, MSLP, TISR, Nino3.4, TCC]`.
TISR is top-of-atmosphere incident solar radiation. Sine and cosine hour-of-day
channels are appended by the dataset loader (`time_encode: true`), giving 10 model
channels for the global weather examples. The nominal 6-hour record removes February 29
(65,700 steps over 1980-2024). Training, validation, and testing cover 1980-2018,
2019-2020, and 2021-2024. Standardization uses training-period statistics; CO2 uses
the fixed range 280-1200 ppm.

Weather CO2 uses NOAA GML Mauna Loa **daily means**, including the documented
Maunakea substitution during 2022-2023, linearly interpolated to 6 hours. Nino 3.4
uses the NOAA PSL **ERSSTv6 monthly anomaly** CSV, held constant within each calendar
month. These retrospective inputs do not reconstruct historical operational release
vintages.

```bash
# Supply the original CSVs first. Existing CO2 output is protected against overwrite.
python scripts/build_co2.py --csv data/co2/co2_daily_mlo.csv
python scripts/build_nino34.py
python scripts/prepare_datasets.py --help
```

`build_nino34.py` accepts `data/nino34.anom.csv` or the historical filename
`data/nina34.anom.csv`. `prepare_datasets.py` reads your raw ERA5 files; inspect its
input paths before running. It consumes the prepared CO2 and Nino 3.4 arrays.

The exact CO2 CSV download vintage used for the manuscript was not retained.
The archived daily CSV recreates 99.65% of the original preprocessing array;
34 daily observations account for the remaining 227 interpolated steps (maximum
difference 0.17 ppm). Rebuilding with a different CSV version need not reproduce the
stored manuscript inputs bit-for-bit. The public repository contains no source data
or residual CSVs.

## Monthly CMIP6 training and conditional rollout

Prepare the two-column monthly scenario arrays with
`scripts/process_cmip6_co2.py` and `scripts/prepare_cmip6_tas.py` using your own
raw files. The CO2 processor selects the highest-pressure layer for new data and
accounts for below-ground missing cells. Archived manuscript CO2 arrays used
the 100-Pa level from CNRM-ESM2-1 (`r3i1p1f2`); pass `--pressure-pa 100` to retain
that layer choice. Existing arrays are protected against overwrite. The example
config (`configs/example_century.yaml`) names the expected historical and four SSP
arrays; the runner can switch ESM directories with `--model`.

```bash
python experiments/cmip6/run_cmip6_pure_ml.py --config configs/example_century.yaml --model CNRM-ESM2-1 --seed 42 --out outputs/example_century_s42 --train-only
python scripts/project_cmip6.py --run-dir outputs/example_century_s42 --device cuda:0 --advance 12
```

The monthly example uses a 60-month look-back, 120-month direct forecast, training
through 2060, and validation through 2080, with the same two-path no-shift architecture
(`hidden_dim=64`, `dropout=0.25`, `use_shift_encoder=false`, `use_delta_head=true`,
`wide_mode=blend`, `epochs=400`, `patience=40`, no weather short-lead auxiliary loss).
Conditional inference loads the saved checkpoint and scaler, initializes from
historical temperature through December 2014, and iteratively appends its own
predictions without future observed temperature feedback (`--advance 12` by default;
`--advance 120` is a separate diagnostic). Prescribed future CO2 enters the monthly
conditioning features and outer gate.

`configs/example_monthly.yaml` and `run_mttgnet_projections.py` retain an earlier
HadCRUT5/CO2 exploratory example for backward compatibility; that recipe is not the
CMIP6 conditional emulation protocol.

## Implementation and metrics

The final two-path model without a dedicated CO2-conditioning branch is assembled in
`experiments/final_noshift/model.py` (`LatestMainModel`, `build_latest_model`), wrapping
the parallel direct pathway and outer sigmoid blend in `experiments/mttgnet_v2/wide.py`
(`WideSkip`, `MTTGNetWideWrapper`) around the clean structured pathway in
`src/mttgnet_test.py` (`MTTGNetTestWrapper`) and shared building blocks in
`src/mttgnet.py`. Both `run_model.py`, `scripts/run_multiseed.py`, and
`experiments/cmip6/run_cmip6_pure_ml.py` dispatch directly to `build_latest_model`
when `model_type: mttgnet_final_noshift` (or `wide_mode`) is set.

Prototype memory is disabled (`use_memory: false`), while internal multi-source fusion
retains three softmax logits (recent context, periodic context, and an inactive zero
source); `src/fusion_config.py` records and verifies this width on checkpoint load.
Query refinement is enabled at both `H=28` and `H=120`. The historical-anchor GAT and
recent-window GNN are distinct encoders; the anchor GAT can be replaced with the
parameter-matched MLP control via `var_encoder_type: mlp_matched`.

Overall RMSE is `sqrt(mean((prediction - target)**2))` for each trained seed;
reported multi-seed RMSE is the arithmetic mean of per-seed scores (not the RMSE of an
ensemble-averaged forecast). With balanced forecast origins, `sqrt(mean_h(rmse_h**2))`
matches overall RMSE up to numerical rounding.

```bash
python -m pytest tests/ -q
```

Tests cover calendar handling, origin-known anchors, batch independence, fusion
configuration and checkpoint loading, final two-path parameter counts and forward
contracts, and recursive rollout without future temperature feedback.
`release_manifest.json` records the exported file allowlist and SHA-256 content hashes.

## Citation and license

```bibtex
@misc{shi2026atmophasenet,
  title  = {AtmoPhaseNet: Multi-Period Graph-Temporal Modeling for Global-Mean Near-Surface Temperature Prediction},
  author = {Shi, Zihao and Wang, Zhiguo and Chen, Ziwei},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/zh-Shi/AtmoPhaseNet}
}
```

MIT license; see `LICENSE`.
