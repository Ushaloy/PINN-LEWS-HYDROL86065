import os
REPO = os.environ.get("PINN_REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.environ.get("BASELINE_OUT", os.path.join(REPO, "results", "baselines"))
os.makedirs(OUT, exist_ok=True)

import json, sys, os, pickle, matplotlib; matplotlib.use("Agg")
split = sys.argv[1]
os.environ["PINN_REPO_DIR"] = REPO
os.environ["PINN_RESULTS_DIR"] = OUT + "/res"
nb = json.load(open(REPO + "/notebooks/01_main_chronological.ipynb"))
code = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__", "display": lambda *a, **k: None}
s0 = code[0].replace('SPLIT_MODE = "chrono"', f'SPLIT_MODE = "{split}"').replace("PREP_ONLY = False", "PREP_ONLY = True").replace("\nREUSE_STORED = True\n", "\nREUSE_STORED = False\n")
exec(compile(s0, "setup", "exec"), g)
try:
    exec(compile(code[1], "main", "exec"), g)
except SystemExit:
    pass
keep = ["X_tr_raw","X_val_raw","X_te_raw","theta_tr","theta_val","theta_te","rain_tr","rain_val","rain_te","t_sec_tr","t_sec_val","t_sec_te","ts_tr","ts_val","ts_te","df","df_tr","df_val","df_te"]
out = {k: g[k] for k in keep}
out["THETA_RAW_BY_TS"] = g["THETA_RAW_BY_TS"]
pickle.dump(out, open(f"{OUT}/feat_{split}.pkl", "wb"))
import sys as _s; _s.stdout = _s.__stdout__
print(split, {k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in out.items() if k.startswith(("X_", "theta_"))})
