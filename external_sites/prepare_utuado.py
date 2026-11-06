"""Builds data/external/utuado_formatted.csv from the USGS 15-min Utuado file (Smith et al., 2020,
https://doi.org/10.5066/P9548YK2). Columns written: timestamp (UTC), theta (probe SP1 at 0.27 m, m3/m3),
rain (cumulative rainfall, mm; the notebooks convert it to increments).

    python external_sites/prepare_utuado.py path/to/utuado_15min.csv
"""
import os, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1]
d = pd.read_csv(src, parse_dates=["timestamp_UTC"])
d = d[(d.timestamp_UTC >= "2018-12-01") & (d.timestamp_UTC < "2019-09-01")]
out = pd.DataFrame({"timestamp": d.timestamp_UTC.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "theta": d["vwc_SP1_27cm_ccpercc"],
                    "rain": d["precipitation_mm"].fillna(0).cumsum().round(3)})
p = os.path.join(REPO, "data", "external", "utuado_formatted.csv")
os.makedirs(os.path.dirname(p), exist_ok=True); out.to_csv(p, index=False)
print(f"{len(out)} rows, {out.theta.notna().sum()} readings, {out.rain.iloc[-1]:.0f} mm -> {p}")
