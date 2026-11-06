"""FoS sensitivity to the assumed strength parameters c' and phi'.

Reads results/<split>/fos_timeseries_main.csv (column u_piml_kPa = pore pressure
of the main-run PIML) and recomputes the infinite-slope FoS (Eq. 13) on a grid of
c' and phi'. Nothing is retrained; only the strength parameters change.

Usage:  python fos_strength_sensitivity.py results/chrono
Output: <dir>/fos_sensitivity_strength.csv
"""
import sys, itertools
import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results/chrono"
f = pd.read_csv(f"{d}/fos_timeseries_main.csv")
u = f["u_piml_kPa"].to_numpy()            # kPa, negative in suction

GAMMA, H, BETA = 14.715, 1.0, np.radians(33.0)   # kN m-3, m, slope angle (Table 1)
BASE = (5.0, 28.0)                                # c' (kPa), phi' (deg) of the paper

def fos(c, phi):
    num = c + (GAMMA * H * np.cos(BETA) ** 2 - u) * np.tan(np.radians(phi))
    return num / (GAMMA * H * np.sin(BETA) * np.cos(BETA))

# sanity check against the saved series
chk = fos(*BASE).min() - f["FoS_piml"].min()
assert abs(chk) < 1e-6, f"base case does not reproduce saved FoS (diff {chk})"

rows = []
for c, phi in itertools.product([0, 1, 2, 3, 5, 7.5, 10, 15, 20], [20, 22, 25, 28, 31, 34]):
    x = fos(c, phi)
    rows.append(dict(c_kPa=c, phi_deg=phi, FoS_min=x.min(), FoS_mean=x.mean(),
                     n_below_1p3=int((x < 1.3).sum()), n_below_1p0=int((x < 1.0).sum()),
                     frac_below_1p3=float((x < 1.3).mean())))
out = pd.DataFrame(rows)
out.to_csv(f"{d}/fos_sensitivity_strength.csv", index=False)
print(out.pivot(index="c_kPa", columns="phi_deg", values="FoS_min").round(2))
