"""Physics-only baselines on the test windows.
free-run : start from the observed theta (uniform profile) at the start of each segment, then rain forcing only.
one-step : before each 2-min step the profile is shifted so that theta at 0.30 m equals the last reading
           (direct insertion of the IoT reading), then integrated 2 min ahead -> forecast for the next reading."""
import os
REPO = os.environ.get("PINN_REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.environ.get("BASELINE_OUT", os.path.join(REPO, "results", "baselines"))
os.makedirs(OUT, exist_ok=True)

import pickle, sys, numpy as np, pandas as pd, time
from richards1d import Column
from sklearn.metrics import r2_score, mean_squared_error

split = sys.argv[1]; mode = sys.argv[2]
VG = dict(theta_r=0.05, theta_s=0.55, alpha=5.9, n=1.48, Ks=0.0043)
F = pickle.load(open(f"{OUT}/feat_{split}.pkl", "rb"))
col = Column(VG)
d = F["df_te"].reset_index(drop=True)          # 2-min grid incl. held points; theta already smoothed
th = d["theta"].values; rin = d["rain_inc"].values; seg = d["seg"].values; ts = d["timestamp"].values
pred = np.full(len(d), np.nan); t0 = time.time()
h = None
for i in range(len(d)):
    new_seg = i == 0 or seg[i] != seg[i - 1]
    if new_seg:
        h = col.h_of_theta(np.full(col.N, th[i])); pred[i] = th[i]; continue
    if mode == "onestep":
        thp = col.theta(h); delta = th[i - 1] - thp[col.i_s]
        h = col.h_of_theta(np.clip(thp + delta, col.tr + 1e-4, col.ts - 1e-4))
    dt = (ts[i] - ts[i - 1]) / np.timedelta64(1, "s")
    q = rin[i] * 1e-3 / dt                      # rain in this 2-min interval (mm) -> m/s
    h = col.advance(h, dt, q)
    pred[i] = col.theta(h)[col.i_s]
    if i % 2000 == 0: print(i, len(d), f"{time.time()-t0:.0f}s", flush=True)
s = pd.Series(pred, index=pd.to_datetime(ts))
p = s.reindex(pd.to_datetime(F["ts_te"])).values
yte = F["theta_te"]; raw = F["THETA_RAW_BY_TS"].reindex(pd.to_datetime(F["ts_te"])).values; ok = ~np.isnan(raw) & ~np.isnan(p)
pers = F["X_te_raw"][:, 1]
res = dict(split=split, model=f"Richards solver ({mode})", test_rmse=float(np.sqrt(np.nanmean((yte - p) ** 2))),
           test_r2=float(r2_score(yte[~np.isnan(p)], p[~np.isnan(p)])),
           skill=1 - np.nanmean((yte - p) ** 2) / mean_squared_error(yte, pers),
           test_rmse_raw=float(np.sqrt(np.mean((raw[ok] - p[ok]) ** 2))), n=int((~np.isnan(p)).sum()), runtime_s=time.time() - t0)
print(res)
pd.DataFrame([res]).to_csv(f"{OUT}/phys_{split}_{mode}.csv", index=False)
pd.DataFrame(dict(timestamp=F["ts_te"], obs=yte, pred=p)).to_csv(f"{OUT}/phys_pred_{split}_{mode}.csv", index=False)
