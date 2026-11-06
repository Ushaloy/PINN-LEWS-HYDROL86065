"""External-site tests (manuscript Sections 2.5, 3.7 and 4.6; Table 6; Fig. 12; Tables S9-S11).

The script runs the code of notebooks/01_main_chronological.ipynb unchanged, except for the site
settings listed in SITES, so that the external sites are processed and modelled exactly like Dev108.

  Local models   (networks trained on the external record itself, Table S11):
      python external_sites/run_external.py local    --site utuado     --seeds 42,43
      python external_sites/run_external.py local    --site california --seeds 42,43
  Transfer Dev108 -> external site (Table 6, Table S10, Fig. 12):
      python external_sites/run_external.py transfer --site utuado     --soil measured
      python external_sites/run_external.py transfer --site utuado     --soil range
      python external_sites/run_external.py transfer --site utuado     --soil dev108
      python external_sites/run_external.py transfer --site california --soil measured

  --soil measured : target soil parameters of SITES (for California: Dev108 values with theta_s 0.48, no soil data exist)
  --soil range    : target theta_s / theta_r, Dev108 alpha, n, Ks
  --soil dev108   : all Dev108 soil values (only the sensor depth is site-specific)
  --quick         : tiny budgets (the notebook's QUICK_TEST); checks that the pipeline runs, numbers are not results

Outputs: results/external/<mode>_<site>[_<soil>]/chrono/ (multiseed_summary.csv or transfer_<site>_summary.csv).
Memory: the L-BFGS phase is full-batch; about 2-3 GB for these records. On 2 CPU cores, one local seed takes about
30 min, one transfer run about 40 min (re-training the Dev108 source network on the target time step included).
"""
import argparse, json, os, sys, time
import matplotlib
matplotlib.use("Agg")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "notebooks", "01_main_chronological.ipynb")
DEV108 = dict(theta_s=0.55, alpha=5.9, n=1.48, Ks=0.0043)          # Table 1
SITES = {
    # Utuado, Puerto Rico (Smith et al., 2020): probe at 0.27 m, 15-min data; laboratory drying branch, sample at 0.22 m
    # (theta_s 0.46, alpha 0.042 1/cm, n 1.586, Ks 1.0e-4 cm/s = 0.0036 m/h); theta_r 0.05 (record minimum 0.086 < lab 0.10)
    "utuado": dict(csv="utuado_formatted.csv", label="Utuado", grid="15min", hourly=False, z=0.27,
                   theta_s=0.46, alpha=4.2, n=1.586, Ks=0.0036),
    # Two Towers landslide, California (Schulz and Smith, 2021): probe at 0.19 m, hourly data; no soil data:
    # theta_s 0.48 (record maximum 0.47), alpha, n, Ks of Dev108
    "california": dict(csv="california_formatted.csv", label="California", grid="1h", hourly=True, z=0.19,
                       theta_s=0.48, alpha=5.9, n=1.48, Ks=0.0043),
}


def load_cells():
    nb = json.load(open(NB, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def sub(code, pairs, where):
    for a, b in pairs:
        if a not in code:
            raise SystemExit(f"[{where}] notebook text not found (notebook changed?): {a[:70]}")
        code = code.replace(a, b)
    return code


def time_step_subs(site):
    """Grid, gap bridging and rain window for the target time step; no target smoothing."""
    s = SITES[site]
    p = [('GRID   = "2min"', f'GRID   = "{s["grid"]}"'), ("ROLL_W = 5", "ROLL_W = 1")]
    if s["hourly"]:   # 1-h step: one missing reading is bridged; the rain input is the 1-h depth
        p += [("GAP_S  = 1800", "GAP_S  = 3600"), ("RAIN_WINDOW_S    = 1800.0", "RAIN_WINDOW_S    = 3600.0"),
              ('raw["R30"] = raw.set_index("timestamp")["rain_inc"].rolling("30min").sum().values', 'raw["R30"] = raw["rain_inc"].values')]
    return p


def setup(cells, out, quick, prep_only):
    os.environ["PINN_REPO_DIR"] = REPO
    os.environ["PINN_RESULTS_DIR"] = out
    s0 = sub(cells[0], [("\nREUSE_STORED = True\n", "\nREUSE_STORED = False\n")], "setup")   # these scripts train on the external site
    if prep_only:
        s0 = sub(s0, [("PREP_ONLY = False", "PREP_ONLY = True")], "setup")
    if quick:
        s0 = sub(s0, [("QUICK_TEST = False", "QUICK_TEST = True")], "setup")
    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    exec(compile(s0, "setup", "exec"), g)
    return g


def target_main_cell(cells, site, soil):
    s = SITES[site]
    p = time_step_subs(site) + [
        ('df = prepare_site(DATA_CSV, 108, warmup_theta=WARMUP_THETA_OK, label="Dev108")',
         f'df = prepare_site(DATA_CSV, 0, warmup_theta=None, label="{s["label"]}")'),
        ("    theta_r=0.05, theta_s=0.55,", f"    theta_r=0.05, theta_s={soil['theta_s']},"),
        ("    alpha=5.9, n=1.48, m=1 - 1/1.48,", f"    alpha={soil['alpha']}, n={soil['n']}, m=1 - 1/{soil['n']},"),
        ("    Ks=0.0043,", f"    Ks={soil['Ks']},"),
        ("Z_SENSOR    = 0.30", f"Z_SENSOR    = {s['z']}"),
        ("Z_COLLOC    = np.array([ 0.30, 0.50], dtype=np.float32)", f"Z_COLLOC    = np.array([ {s['z']}, 0.50], dtype=np.float32)")]
    return sub(cells[1], p, "main")


def run_main(code, g, name):
    """The main cell ends with SystemExit when PREP_ONLY = True."""
    try:
        exec(compile(code, name, "exec"), g)
    except SystemExit:
        pass
    sys.stdout = sys.__stdout__


def soil_values(site, soil):
    s = SITES[site]
    if soil == "measured":
        return dict(theta_s=s["theta_s"], alpha=s["alpha"], n=s["n"], Ks=s["Ks"])
    if soil == "range":
        return dict(theta_s=s["theta_s"], alpha=DEV108["alpha"], n=DEV108["n"], Ks=DEV108["Ks"])
    return dict(DEV108)


def target_test_window(cells, site, soil, quick, out):
    """Test window of the target's own chronological split (data preparation only, no training)."""
    g = setup(cells, out + "_prep", quick, prep_only=True)
    g["DATA_CSV"] = os.path.join(REPO, "data", "external", SITES[site]["csv"])
    run_main(target_main_cell(cells, site, soil_values(site, soil)), g, "main_target_prep")
    return str(g["ts_te"][0]), str(g["ts_te"][-1])


def local(args, cells):
    out = os.path.join(REPO, "results", "external", f"local_{args.site}")
    seeds = [int(x) for x in args.seeds.split(",")]
    g = setup(cells, out, args.quick, prep_only=True)
    g["DATA_CSV"] = os.path.join(REPO, "data", "external", SITES[args.site]["csv"])
    run_main(target_main_cell(cells, args.site, soil_values(args.site, "measured")), g, "main")
    for idx, name in [(2, "boundary_modes"), (3, "sensitivity")]:
        c = cells[idx]
        if idx == 3:
            c = sub(c, [("RUN_SENSITIVITY   = True", "RUN_SENSITIVITY   = False")], name)
        exec(compile(c, name, "exec"), g)
    ms = [c for c in cells if "MS_SEEDS      = BUDGET" in c][0]
    if not args.quick:
        ms = sub(ms, [('MS_SEEDS      = BUDGET["seeds"]', f"MS_SEEDS      = {seeds}")], "multiseed")
    t = time.time(); exec(compile(ms, "multiseed", "exec"), g); sys.stdout = sys.__stdout__
    print(f"LOCAL DONE ({time.time() - t:.0f} s) -> {out}", flush=True)


def transfer(args, cells):
    site, s = args.site, SITES[args.site]
    out = os.path.join(REPO, "results", "external", f"transfer_{site}_{args.soil}")
    t0, t1 = target_test_window(cells, site, args.soil, args.quick, out)
    soil = soil_values(site, args.soil)
    # 1) Dev108 source networks re-trained on the target time step (final configuration, seed 42)
    g = setup(cells, out, args.quick, prep_only=False)
    g["DATA_CSV_107"] = os.path.join(REPO, "data", "external", s["csv"])
    run_main(sub(cells[1], time_step_subs(site), "main_dev108"), g, f"main_dev108_{s['grid']}")
    for idx in (2, 3):
        exec(compile(sub(cells[idx], [("RUN_SENSITIVITY   = True", "RUN_SENSITIVITY   = False")], "cell3") if idx == 3 else cells[idx],
                     f"cell{idx}", "exec"), g)
    # 2) the notebook's transfer cell, with the external site as target
    f = [c for c in cells if "TRANSFER TO A SECOND NODE" in c or "Transfer to a second node" in c][0]
    z = s["z"]
    f = sub(f, [
        ('VG107 = dict(theta_r=0.05, theta_s=0.463, alpha=5.9, n=1.48, m=1 - 1 / 1.48, Ks=0.0041)',
         f'VG107 = dict(theta_r=0.05, theta_s={soil["theta_s"]}, alpha={soil["alpha"]}, n={soil["n"]}, m=1 - 1 / {soil["n"]}, Ks={soil["Ks"]})'),
        ('d7 = prepare_site(DATA_CSV_107, 107, warmup_theta=None, label="Dev107")',
         f'd7 = prepare_site(DATA_CSV_107, 0, warmup_theta=None, label="{s["label"]}")'),
        # time axis: the target test window starts where the Dev108 test window starts
        ('d7["t_sec"] = np.interp(d7["timestamp"].values.astype("datetime64[ns]").astype(np.int64), _ts8, df["t_sec"].values)',
         f'_i0 = int(np.searchsorted(d7["timestamp"].values, np.datetime64("{t0}"))); d7["t_sec"] = d7["t_sec"] - d7["t_sec"].iloc[_i0] + float(t_sec_te[0])'),
        ('TEST_START_TL = pd.Timestamp(ts_te[0]); TEST_END_TL = pd.Timestamp(ts_te[-1])',
         f'TEST_START_TL = pd.Timestamp("{t0}"); TEST_END_TL = pd.Timestamp("{t1}")'),
        # target soil and sensor depth while the model is used at the target
        ('            self._saved = dict(VG); VG.update(VG107); return self',
         '            global z_sens_norm, Z_COLLOC\n            self._saved = dict(VG); VG.update(VG107)\n'
         f'            self._z = (z_sens_norm, Z_COLLOC); z_sens_norm = torch.tensor([[{z} / Z_MAX]], dtype=torch.float32, device=DEVICE); '
         f'Z_COLLOC = np.array([{z}, 0.50], dtype=np.float32); return self'),
        ('            VG.clear(); VG.update(self._saved)',
         '            global z_sens_norm, Z_COLLOC\n            VG.clear(); VG.update(self._saved); z_sens_norm, Z_COLLOC = self._z'),
        ('TL_CSV = os.path.join(OUT, f"transfer_107_{RUN_TAG}.csv")', f'TL_CSV = os.path.join(OUT, "transfer_{site}.csv")'),
        ('summ.to_csv(os.path.join(OUT, "transfer_107_summary.csv"), index=False)',
         f'summ.to_csv(os.path.join(OUT, "transfer_{site}_summary.csv"), index=False)'),
    ], "transfer")
    exec(compile(f, f"transfer_{site}", "exec"), g); sys.stdout = sys.__stdout__
    print(f"TRANSFER DONE -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["local", "transfer"])
    ap.add_argument("--site", choices=list(SITES), required=True)
    ap.add_argument("--seeds", default="42,43", help="local mode: comma-separated seeds")
    ap.add_argument("--soil", choices=["measured", "range", "dev108"], default="measured", help="transfer mode: soil values at the target")
    ap.add_argument("--quick", action="store_true", help="pipeline check with tiny budgets")
    a = ap.parse_args()
    (local if a.mode == "local" else transfer)(a, load_cells())
