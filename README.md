# Physics-informed soil-moisture models for landslide early warning

Code and data for the revised manuscript **HYDROL86065**, *Journal of Hydrology*:
a Richards-equation-constrained neural network (PINN) for IoT soil-moisture monitoring on a 33° hillslope, with slope stability, boundary-condition tests, transfer between two monitoring nodes and to two external sites (Puerto Rico, California), and an alert layer.

All result tables and figures of the revision are produced by the two notebooks in `notebooks/` from the raw node files in `data/`.

## Repository layout

```
PINN-LEWS-HYDROL86065/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md                      column definitions, node and soil properties
│   ├── dev_108_updated.csv            source node (model development)
│   ├── dev_107_updated.csv            second node (transfer target)
│   └── external/                      public USGS records: Utuado (Puerto Rico), Two Towers (California)
├── analysis/                          FoS sensitivity to the assumed c′ and φ′
├── baselines/                         classical and process-based baselines (CPU scripts, see baselines/README.md)
├── external_sites/                    transfer to the external sites (see external_sites/README.md)
├── notebooks/
│   ├── 01_main_chronological.ipynb    main results (SPLIT_MODE = "chrono"); loads the stored checkpoints, no training by default
│   ├── 02_out_of_season.ipynb         out-of-season test (SPLIT_MODE = "season"); same, no training by default
│   └── 03_baselines_external_analysis.ipynb   runs baselines/, analysis/ and external_sites/ (Colab, one click)
└── results/
    ├── README.md
    ├── architecture_search/           tables of the network search
    ├── baselines/                     ARX, random forest, gradient boosting, Richards solver (Table 3, Table 5)
    ├── chrono/                        outputs of notebook 01
    ├── external/                      outputs of external_sites/ (reference/: values of the manuscript)
    └── season/                        outputs of notebook 02
```

The two notebooks contain the same code; only `SPLIT_MODE` in the setup cell differs.

## How to run

**By default the notebooks do not train.** The setup cell has `REUSE_STORED = True`: the trained networks, resume caches and finished analyses stored in `results/<split>/` are loaded, and every table and figure is regenerated from them (about 20 s on a CPU, measured). The stored results are never overwritten; the regenerated files are written to `results_rerun/<split>/`. No GPU and no Google Drive are needed.

**Google Colab.**

1. Open the notebook directly from GitHub: click the *Open in Colab* badge at the top of the notebook on GitHub, or use *File → Open notebook → GitHub*, enter `Ushaloy/PINN-LEWS-HYDROL86065` and choose `notebooks/01_main_chronological.ipynb`.
   Colab loads only the notebook file. The setup cell therefore clones this repository (code, `data/` and the stored results) to `/content/PINN-LEWS-HYDROL86065` by itself, so no manual step is needed. The repository must be public.
2. *Runtime → Run all.* The printed log shows `REUSE_STORED = True: loading the stored PIML checkpoint, no training`. The new files are in `/content/PINN-LEWS-HYDROL86065/results_rerun/chrono/` (file panel on the left); they can be compared with `results/chrono/`.
3. Repeat with `02_out_of_season.ipynb`.
4. Baselines, external sites and the strength sensitivity: open `notebooks/03_baselines_external_analysis.ipynb` the same way (it clones the repository itself).

**Local machine.**

```bash
git clone https://github.com/Ushaloy/PINN-LEWS-HYDROL86065
cd PINN-LEWS-HYDROL86065
pip install -r requirements.txt
jupyter notebook notebooks/01_main_chronological.ipynb
```

Outputs go to `results_rerun/<split>/`.

**Train from scratch.** Set `REUSE_STORED = False` in the setup cell. Everything is then trained again (table below). On Colab the outputs are written to Google Drive, `MyDrive/PINN-LEWS-HYDROL86065_results/<split>/`, so nothing is lost if the session disconnects. A GPU runtime is required (*Runtime → Change runtime type → T4 GPU*).

After a disconnect during a from-scratch run, run these cells in order:

1. the setup cell;
2. the main cell, with `PREP_ONLY = True` so it does not re-train;
3. the boundary-mode cell;
4. the cell you were in.

Every multi-run analysis saves after each run and skips finished runs.

**Pipeline check.** Set `QUICK_TEST = True` in the setup cell. Every cell then runs with tiny budgets and 2 seeds, writing to `quicktest_<split>/`. This trains, takes a few minutes on a Colab GPU, and the numbers are not results.

### Run time and hardware

| Notebook | Main contents | Stored checkpoints (default) | From scratch, Colab T4 |
|---|---|---|---|
| 01 (chrono) | main run, λ check, boundary scenarios, multi-seed, LSTM, transfer, alerts | ≈ 20 s on a CPU | ≈ 8–9 h |
| 02 (season) | main run, references, multi-seed | ≈ 20 s on a CPU | ≈ 2.5 h |

Training (not re-use) needs a GPU, or a machine with at least 12 GB RAM: the L-BFGS phase is full-batch and needs about 6 GB of memory.

## Baselines

The comparison with persistence, ARX, random forest, gradient boosting and a process-based Richards solver (with and without IoT updates) is produced by the CPU scripts in `baselines/` on exactly the notebooks' inputs and splits (about 15 min; see `baselines/README.md`).

## External sites

The transfer to two public USGS records — Utuado, Puerto Rico (laboratory soil parameters, 15-min data) and Two Towers, California (no measured soil data, hourly) — is run by the scripts in `external_sites/` with the notebook code (see `external_sites/README.md`).

- At Utuado, the Dev108 network used on day one with the measured soil parameters gives R² = 0.98 (RMSE 0.0065 vs 0.0040 for persistence).
- With the Dev108 soil values instead, the day-one error is 0.0121.
- With only the Utuado θs and θr, it is again 0.0065: on day one the model needs the moisture range of the new probe, while α, n and Ks act only in retraining and in the pressure head.
- After 14 days of local data, the fine-tuned network matches networks trained on the full local record.
- In California, transfer fails (day one 0.0148 vs 0.0006 for persistence).

## Method in brief

**Data**
- The 2-min telemetry is cleaned: duplicates removed, sensor warm-up trimmed, logger-filled straight-line runs removed.
- Rain is rebuilt from the tipping-bucket counter (0.2794 mm per tip) as a 30-min depth.
- Readings are placed on a fixed 2-min grid:
  - outages ≤ 30 min are bridged and used as inputs only;
  - longer outages start a new segment.

**Model**
- Network: θ(z, t, inputs), with inputs R(t) and 15 lags of θ (30 min). Target: θ at 0.30 m, rolling median of 5 values.
- Loss = data misfit + λ_phys · Richards residual (z positive downward, van Genuchten–Mualem, autograd derivatives) + initial-condition and free-drainage lower-boundary terms + a depth-gradient term.

**Baselines**
- The data-only baseline has the identical network and training, with all λ = 0.
- Persistence is θ(t) = θ(t − 2 min).
- Skill = 1 − MSE / MSE_persistence.

**Final configuration**

| Item | Value |
|---|---|
| Network | 5 × 128 + 64, tanh, 76,801 parameters |
| Loss weights | λ_phys 0.1, λ_BC 0 (surface flux off), λ_IC 0.1, λ_LBC 0.1 (free drainage), λ_zvar 0.1 |
| Training | 400 AdamW (warm-up + cosine) + 200 L-BFGS; best-validation checkpoint |
| Seeds | 42, 43, 44 |
| Split (chrono) | 80 / 10 / 10 % of measured points in time; test 12–29 Sep 2026 |
| Split (season) | train 1 Nov – 1 Apr, validation 1–18 Apr, test 1 May – 29 Sep 2026 |
| Slope | infinite slope, 33°, c′ 5 kPa, φ′ 28°, γ 14.715 kN m⁻³, H 1 m, u = Se·γw·ψ |

## Outputs and where they are used

The stored files are in `results/<split>/`. A re-run with `REUSE_STORED = True` writes the same file names to `results_rerun/<split>/`.

| File | Content |
|---|---|
| `multiseed_summary.csv` | **main result**: per-seed test R² / RMSE, PIML and data-only |
| `leakage_check_raw_target.csv` | scores against the smoothed target and the raw sensor readings; persistence |
| `main_run_summary.csv`, `fig01`–`fig11` | main-run (seed 42) models: fit, residuals, Richards residual, FoS |
| `sensitivity_lambda*.csv`, `fig_sensitivity_lambda.png` | λ check around the final weights |
| `sensitivity_skill_vs_persistence.csv` | physics vs data-only at the λ-check budget |
| `sensitivity_lower_bc_ic_*.csv`, `fig_sensitivity_lower_bc_ic.png` | lower-boundary and initial-condition scenarios |
| `lstm_results.csv` | LSTM baseline |
| `transfer_107_*.csv`, `fig_transfer_108_to_107.png` | transfer Dev108 → Dev107 |
| `fos_timeseries_main.csv` | whole-record θ, ψ, u and FoS of the main-run models |
| `fos_sensitivity_slope.csv`, `fos_sensitivity_ks.csv`, `fos_sensitivity_grid_min.csv` | FoS vs slope angle and the Ks proxy |
| `fos_sensitivity_strength.csv` | FoS vs the assumed c′ and φ′ (made by `analysis/fos_strength_sensitivity.py`, chrono only) |
| `alert_stream_summary.txt`, `alert_episodes.csv`, `alert_debounce_policies.csv`, `fig_alert_stream.png` | alert levels and lead time |
| `run_log.txt`, `environment.txt` | full printed log; package versions, GPU, data SHA-256 |

PIML in file and column names means the physics-informed network (PINN). ML means the data-only network.

## Manuscript tables and where they come from

| Manuscript | Files |
|---|---|
| Table 3 (comparison with other methods) | `results/chrono/multiseed_summary.csv`, `results/season/multiseed_summary.csv`, `results/chrono/lstm_results.csv`, `results/baselines/*.csv` |
| Table 4 (lower boundary and initial condition) | `results/chrono/sensitivity_lower_bc_ic_summary.csv`, `sensitivity_lower_bc_identifiability.csv`, `sensitivity_lower_bc_ic_per_seed.csv` |
| Table 5 (transfer to Dev107) | `results/chrono/transfer_107_summary.csv`, `results/baselines/transfer_ml.csv` |
| Alert statistics (Section 4.8) | `results/chrono/alert_stream_summary.txt`, `alert_episodes.csv`, `alert_debounce_policies.csv` |
| External sites | `results/external/reference/` |

The figures written by the notebooks are the PNG files in `results/<split>/`. Their layout differs from the manuscript figures. The code in `baselines/` and `external_sites/` writes tables only; the manuscript figures built from those tables are not drawn by code in this repository.

## Reference results (Colab T4, October 2026)

GPU training is not bit-for-bit deterministic, so a re-run can differ in the last digits.

**Dev108, chronological test.** Test RMSE in m³ m⁻³, 3 seeds.

| Model | Test RMSE (mean ± SD) | R² | Skill |
|---|---|---|---|
| PIML | 0.00044 ± 0.00003 | 0.9995 | −1.13 |
| Data-only | 0.00037 ± 0.00001 | 0.9997 | −0.49 |
| LSTM (current rain) | 0.00043 ± 0.00007 | | |
| Persistence | 0.00030 | 0.9998 | 0 |

- Normalised Richards residual on the validation inputs of the λ-check runs (one-third training budget, 3 seeds; `sensitivity_lambda.csv`, column `val_phys`): PIML 2.7 × 10⁻⁶ vs data-only 0.83.
- Full-budget main run (seed 42; `main_run_summary.csv`, column `richards_resid_sd`): residual SD 4.1 × 10⁻¹¹ (PIML) vs 1.2 × 10⁻⁷ (data-only) m³ m⁻³ s⁻¹.
- Against the raw readings (seed 42), the three are equal within sensor noise: persistence 0.000597, PIML 0.000601, data-only 0.000586.

**Out-of-season test (May–Sep).**

| Model | Test RMSE |
|---|---|
| PIML | 0.00316 ± 0.00068 |
| Data-only | 0.00241 ± 0.00054 |
| Persistence | 0.00121 |

The PIML vs data-only difference is not significant (paired p = 0.12).

**Boundary scenarios.**
- Free drainage is the only lower boundary consistent with the data: no-flux costs about 5× in validation error, and a water table at 0.5 m cannot be imposed.
- The initial profile changes θ by ≤ 0.0001.
- Minimum FoS is 1.66–1.69 in every scenario that met its condition.

**Transfer Dev108 → Dev107.** Test RMSE by number of Dev107 days used.

| Strategy | 3 d | 7 d | 14 d | 30 d |
|---|---|---|---|---|
| recal (2 parameters) | 0.00103 | 0.00103 | 0.00103 | 0.00103 |
| full_pinn | 0.00191 | 0.00150 | 0.00127 | 0.00108 |
| scratch_pinn | 0.00316 | 0.00583 | 0.00600 | 0.00243 |

Persistence is 0.00095. Transfer beats training from scratch at every N.

**Slope stability.** Minimum FoS is 1.590 (PIML; 1.589 for the data-only network) over the full record, with no FoS < 1.3 at 33°. This holds for the assumed strength parameters c′ = 5 kPa and φ′ = 28° only; see the first note below.

## Notes and limitations

- **Strength parameters are assumed.** c′ = 5 kPa and φ′ = 28° were not measured and are not taken from a published source. The minimum FoS depends strongly on c′: with φ′ = 28°, it is 1.59 for c′ = 5 kPa, 1.29 for 3 kPa, 1.14 for 2 kPa and 0.85 for 0 kPa (`results/chrono/fos_sensitivity_strength.csv`). The FoS values and the alert levels are conditional on these assumptions. The soil-moisture results do not depend on them.
- **Transfer scope.** The two nodes have the same texture (sandy clay loam). Across sites, transfer was tested at one external site with measured soil (silty sand, similar hydraulic properties) and one without soil data. Transfer between soils with contrasting hydraulic behaviour is untested.
- **Ks sensitivity is a proxy.** `compute_fos_series` represents a Ks change as ψ / ks_scale on fixed θ predictions; the flow problem is not re-solved. The slope-angle sensitivity is exact.
- **Smoothed target.** Models are trained on a rolling-median target; the raw-reading check reports every key score against the unsmoothed readings too.
- **Architecture search data.** The search (`results/architecture_search/`) was run on the Nov 2025 – Jan 2026 part of the Dev108 record. The optional cell 12 repeats it on the full record.
- **Alert definitions.** The lead-time and alert definitions follow the submitted manuscript (Eq. 17, Section 3.8). The three showcase events are the storms of 10 Nov, 18–19 Nov 2025 and 24–25 Jan 2026.
- **Dev101 excluded.** The third node (campus, sandy clay) is not used. Its readings exceed the probe's specified range and the soil porosity, and storm rises exceed the rain supplied (see the revision's response letter).

## License

The code is released under the MIT License (see `LICENSE`). See `data/README.md` for the data terms.
