# External-site tests

Scripts for the transfer of the Dev108 networks to two public USGS landslide-monitoring records (manuscript Sections 2.5, 3.7 and 4.6; Table 6; Fig. 12; Tables S9–S11).

| Site | Record | Probe | Step | Soil parameters |
|---|---|---|---|---|
| Utuado, Puerto Rico | 1 Dec 2018 – 31 Aug 2019 | 0.27 m | 15 min | measured (laboratory, drying branch, 0.22 m) |
| Two Towers, California | 19 Jan 2017 – 29 Apr 2020 | 0.19 m | 1 h | none published; Dev108 values, θs 0.48 |

The scripts run the code of `notebooks/01_main_chronological.ipynb` unchanged. Only the site settings in `SITES` (`run_external.py`) are substituted:

- the time step, with gap bridging and rain window at the hourly step;
- no target smoothing;
- the soil parameters;
- the sensor depth.

Every substitution is checked: if the notebook text changes, the script stops with a message instead of running a different model.

## Data

`data/external/utuado_formatted.csv` and `data/external/california_formatted.csv` have the columns `timestamp`, `theta` and `rain`, where `rain` is cumulative in mm. They are taken from the USGS data releases, which are public domain:

- Smith, J.B., Thomas, M.A., Ashland, F.A., Michel, A., Mirus, B.B., Wayllace, A., 2020. Hillslope hydrologic monitoring data following Hurricane Maria in 2017, Puerto Rico, July 2018 to June 2020. U.S. Geological Survey data release. https://doi.org/10.5066/P9548YK2
- Schulz, W.H., Smith, J.B., 2021. Precipitation and soil-moisture data from the Two Towers landslide, Trinity County, California. U.S. Geological Survey data release. https://doi.org/10.5066/P9NSUMOY

`prepare_utuado.py` rebuilds the Utuado file from the 15-min file of the release.

## Runs

```bash
# networks trained on the external records (Table S11)
python external_sites/run_external.py local --site utuado --seeds 42,43
python external_sites/run_external.py local --site california --seeds 42,43
python external_sites/baselines_external.py --site utuado
python external_sites/baselines_external.py --site california

# transfer Dev108 -> external site (Table 6, Table S10, Fig. 12)
python external_sites/run_external.py transfer --site utuado --soil measured
python external_sites/run_external.py transfer --site utuado --soil range     # theta_s, theta_r only
python external_sites/run_external.py transfer --site utuado --soil dev108    # no site soil information
python external_sites/run_external.py transfer --site california --soil measured
```

Add `--quick` to check the pipeline in a few minutes; with `--quick` the numbers are not results.

**Outputs.** Results are written to `results/external/`. The reference results of the manuscript are in `results/external/reference/`.

**Run time.** On 2 CPU cores:
- one local seed takes about 30 min;
- one transfer run takes about 40 min, including re-training the Dev108 source network on the target time step;
- the baselines take a few minutes.

**Memory.** L-BFGS is full-batch and needs about 2–3 GB for these records. Run one job at a time on small machines.
