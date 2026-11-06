"""Classical baselines on exactly the notebook's features: ridge ARX, random forest, gradient boosting.
Hyper-parameters chosen on the validation split; three seeds for the stochastic models."""
import os
REPO = os.environ.get("PINN_REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.environ.get("BASELINE_OUT", os.path.join(REPO, "results", "baselines"))
os.makedirs(OUT, exist_ok=True)

import pickle, sys, json, numpy as np, pandas as pd, time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

VG = dict(theta_r=0.05, theta_s=0.55)
def norm(X):
    Xn = X.copy(); Xn[:, 0] = X[:, 0] / 75.0; Xn[:, 1:] = (X[:, 1:] - VG["theta_r"]) / (VG["theta_s"] - VG["theta_r"]); return Xn
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

rows = []
for split in ["chrono", "season"]:
    F = pickle.load(open(f"{OUT}/feat_{split}.pkl", "rb"))
    Xtr, Xva, Xte = norm(F["X_tr_raw"]), norm(F["X_val_raw"]), norm(F["X_te_raw"])
    ytr, yva, yte = F["theta_tr"], F["theta_val"], F["theta_te"]
    raw = F["THETA_RAW_BY_TS"].reindex(pd.to_datetime(F["ts_te"])).values; ok = ~np.isnan(raw)
    pers = F["X_te_raw"][:, 1]; mse_p = mean_squared_error(yte, pers)
    def score(name, seed, p, hp, fit_s):
        rows.append(dict(split=split, model=name, seed=seed, hp=hp, test_rmse=rmse(yte, p), test_r2=float(r2_score(yte, p)),
                         skill=1 - mean_squared_error(yte, p) / mse_p, test_rmse_raw=rmse(raw[ok], p[ok]), fit_s=fit_s))
        print(split, name, seed, hp, f"{rows[-1]['test_rmse']:.6f}", f"raw {rows[-1]['test_rmse_raw']:.6f}", flush=True)
    score("persistence", 0, pers, "", 0)
    # ridge ARX
    best = min([(rmse(yva, Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)), a) for a in [1e-6, 1e-4, 1e-2, 1]])
    t = time.time(); m = Ridge(alpha=best[1]).fit(Xtr, ytr); score("ARX (ridge)", 0, m.predict(Xte), f"alpha={best[1]:g}", time.time() - t)
    # random forest
    best = min([(rmse(yva, RandomForestRegressor(200, min_samples_leaf=l, max_features=0.5, n_jobs=2, random_state=42).fit(Xtr, ytr).predict(Xva)), l) for l in [1, 5, 20]])
    for s in [42, 43, 44]:
        t = time.time(); m = RandomForestRegressor(300, min_samples_leaf=best[1], max_features=0.5, n_jobs=2, random_state=s).fit(Xtr, ytr)
        score("Random forest", s, m.predict(Xte), f"leaf={best[1]}", time.time() - t)
    # gradient boosting
    grid = [(lr, ml) for lr in [0.05, 0.1] for ml in [15, 63]]
    best = min([(rmse(yva, HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=ml, max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=42).fit(Xtr, ytr).predict(Xva)), lr, ml) for lr, ml in grid])
    for s in [42, 43, 44]:
        t = time.time(); m = HistGradientBoostingRegressor(learning_rate=best[1], max_leaf_nodes=best[2], max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=s).fit(Xtr, ytr)
        score("Gradient boosting", s, m.predict(Xte), f"lr={best[1]},leaves={best[2]}", time.time() - t)
pd.DataFrame(rows).to_csv(OUT + "/ml_baselines.csv", index=False)
