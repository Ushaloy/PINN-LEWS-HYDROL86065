"""Classical and process-based baselines at the external sites (Table S11), on exactly the inputs and split
of run_external.py: persistence, ARX (ridge), random forest, gradient boosting, Richards solver without IoT update.

    python external_sites/baselines_external.py --site utuado
    python external_sites/baselines_external.py --site california

Output: results/external/baselines_<site>.csv (CPU, a few minutes).
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
import run_external as R
from richards1d import Column

ap = argparse.ArgumentParser(); ap.add_argument("--site", choices=list(R.SITES), required=True); a = ap.parse_args()
site, S = a.site, R.SITES[a.site]
cells = R.load_cells()
out = os.path.join(R.REPO, "results", "external", f"baselines_{site}")
g = R.setup(cells, out, quick=False, prep_only=True)
g["DATA_CSV"] = os.path.join(R.REPO, "data", "external", S["csv"])
R.run_main(R.target_main_cell(cells, site, R.soil_values(site, "measured")), g, "main_prep")

ts_, tr_ = S["theta_s"], 0.05
step_s = 3600.0 if S["hourly"] else 900.0
norm = lambda X: np.column_stack([X[:, 0] / 75.0, (X[:, 1:] - tr_) / (ts_ - tr_)])   # rain, lagged Se
Xtr, Xva, Xte = norm(g["X_tr_raw"]), norm(g["X_val_raw"]), norm(g["X_te_raw"])
ytr, yva, yte = g["theta_tr"], g["theta_val"], g["theta_te"]
pers = g["X_te_raw"][:, 1]; mp = mean_squared_error(yte, pers); rows = []
rm = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
def score(name, seed, p):
    rows.append(dict(model=name, seed=seed, rmse=rm(yte, p), r2=float(r2_score(yte, p)), skill=1 - mean_squared_error(yte, p) / mp))
    print(rows[-1], flush=True)

score("persistence", 0, pers)
alpha = min([(rm(yva, Ridge(alpha=x).fit(Xtr, ytr).predict(Xva)), x) for x in [1e-6, 1e-4, 1e-2, 1]])[1]
score("ARX (ridge)", 0, Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte))
leaf = min([(rm(yva, RandomForestRegressor(200, min_samples_leaf=l, max_features=0.5, n_jobs=1, random_state=42).fit(Xtr, ytr).predict(Xva)), l)
            for l in [1, 5, 20]])[1]
for s in [42, 43, 44]:
    score("Random forest", s, RandomForestRegressor(300, min_samples_leaf=leaf, max_features=0.5, n_jobs=1, random_state=s).fit(Xtr, ytr).predict(Xte))
for s in [42, 43, 44]:
    score("Gradient boosting", s, HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=63, max_iter=1000, early_stopping=True,
                                                                validation_fraction=0.1, random_state=s).fit(Xtr, ytr).predict(Xte))
# Richards solver driven by rain only over the test window, started from the reading before it
d = g["df"]; te0 = pd.Timestamp(g["ts_te"][0])
w = d[d.timestamp >= te0 - pd.Timedelta(seconds=15 * step_s)].reset_index(drop=True)
col = Column(dict(theta_r=tr_, theta_s=ts_, alpha=S["alpha"], n=S["n"], Ks=S["Ks"]), z_sensor=S["z"])
h = col.h_of_theta(np.full(col.N, w.theta.iloc[0])); pred = []
for i in range(len(w)):
    if i:
        h = col.advance(h, step_s, w.rain_inc.iloc[i] * 1e-3 / step_s)
    pred.append(col.theta(h)[col.i_s])
score("Richards solver, no IoT update", 0, pd.Series(pred, index=w.timestamp).reindex(pd.to_datetime(g["ts_te"])).values)
os.makedirs(os.path.join(R.REPO, "results", "external"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(R.REPO, "results", "external", f"baselines_{site}.csv"), index=False)
