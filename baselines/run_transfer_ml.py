"""Transfer baselines Dev108 -> Dev107 for ARX and random forest (same windows as the notebook's transfer cell).
Zero-shot: trained on Dev108 in effective-saturation space, applied to Dev107 with Dev107's theta_s.
From scratch: trained only on the last N days of Dev107 data."""
import os
REPO = os.environ.get("PINN_REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.environ.get("BASELINE_OUT", os.path.join(REPO, "results", "baselines"))
os.makedirs(OUT, exist_ok=True)

import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import prepare_site, flag_linear_fill, counter_to_increments   # noqa (same functions as the notebook)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

LAG = 15
def build_features(d, lag=LAG):
    th, seg = d["theta"].values.astype(np.float64), d["seg"].values
    idx = np.arange(lag, len(d)); idx = idx[seg[idx] == seg[idx - lag]]
    if "held" in d: idx = idx[~d["held"].values[idx].astype(bool)]
    lags = np.stack([th[idx - k] for k in range(1, lag + 1)], axis=1)
    X = np.column_stack([d["rain"].values[idx], lags]).astype(np.float32)
    return X, th[idx].astype(np.float32), d["timestamp"].values[idx]
def norm(X, ts_):
    Xn = X.copy(); Xn[:, 0] = X[:, 0] / 75.0; Xn[:, 1:] = (X[:, 1:] - 0.05) / (ts_ - 0.05); return Xn

F8 = pickle.load(open(OUT + "/feat_chrono.pkl", "rb"))
D = REPO + "/data/"
d7 = prepare_site(D + "dev_107_updated.csv", 107, warmup_theta=None, label="Dev107")
d7["theta_raw"] = d7["theta"].values
d7["theta"] = d7.groupby("seg")["theta"].transform(lambda s: s.rolling(5, min_periods=1).median())
T0, T1 = pd.Timestamp(F8["ts_te"][0]), pd.Timestamp(F8["ts_te"][-1])
d7_test = d7[(d7.timestamp >= T0) & (d7.timestamp <= T1)].reset_index(drop=True)
d7_pre = d7[d7.timestamp < T0].reset_index(drop=True)
Xte, yte, tste = build_features(d7_test)
raw = d7_test.set_index("timestamp")["theta_raw"].reindex(pd.to_datetime(tste)).values; ok = ~np.isnan(raw)
pers = Xte[:, 1]; mse_p = mean_squared_error(yte, pers); mse_pr = np.mean((raw[ok] - pers[ok]) ** 2)
print("Dev107 test points", len(yte), "persistence RMSE", np.sqrt(mse_p))
rows = []
def score(name, n, seed, p):
    rows.append(dict(model=name, n_days=n, seed=seed, test_rmse=float(np.sqrt(mean_squared_error(yte, p))),
                     skill=1 - mean_squared_error(yte, p) / mse_p,
                     test_rmse_raw=float(np.sqrt(np.mean((raw[ok] - p[ok]) ** 2))), skill_raw=1 - np.mean((raw[ok] - p[ok]) ** 2) / mse_pr))
    print(name, n, seed, f"{rows[-1]['test_rmse']:.6f}", f"raw skill {rows[-1]['skill_raw']:+.3f}")
# zero-shot in Se space
X8 = norm(np.vstack([F8["X_tr_raw"], F8["X_val_raw"]]), 0.55); y8 = (np.concatenate([F8["theta_tr"], F8["theta_val"]]) - 0.05) / 0.5
Xn7 = norm(Xte, 0.463); back = lambda se: 0.05 + se * (0.463 - 0.05)
score("ARX zero-shot", 0, 0, back(Ridge(alpha=0.01).fit(X8, y8).predict(Xn7)))
for s in [42, 43, 44]:
    score("RF zero-shot", 0, s, back(RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=s).fit(X8, y8).predict(Xn7)))
# from scratch on N days
days = pd.Series(d7_pre.timestamp.dt.normalize().unique()).sort_values()
for n in [3, 7, 14, 30]:
    keep = set(days.iloc[-n:]); dd = d7_pre[d7_pre.timestamp.dt.normalize().isin(keep)].reset_index(drop=True)
    X, y, _ = build_features(dd); Xn = norm(X, 0.463)
    score("ARX from scratch", n, 0, Ridge(alpha=0.01).fit(Xn, y).predict(Xn7))
    for s in [42, 43, 44]:
        score("RF from scratch", n, s, RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=s).fit(Xn, y).predict(Xn7))
pd.DataFrame(rows).to_csv(OUT + "/transfer_ml.csv", index=False)
