import numpy as np, pandas as pd
def counter_to_increments(c, eps=1e-6):
    """Positive steps of the counter are new rain; at each reset the first
    post-reset value is counted as rain already fallen in the new cycle."""
    c = np.asarray(c, float); inc = np.zeros(len(c)); falling = False
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        if d < -eps:
            falling = True
        else:
            if falling:
                inc[i - 1] += c[i - 1]; falling = False
            inc[i] += max(d, 0.0)
    return inc

def flag_linear_fill(theta, min_run=10, tol=1e-7):
    """Straight-line runs (constant non-zero slope for >= min_run steps) are values
    filled in between logger uploads, not measurements. They are dropped."""
    x = np.asarray(theta, float); d1 = np.diff(x)
    lin = (np.abs(np.diff(d1)) < tol) & (np.abs(d1[1:]) > tol)
    out = np.zeros(len(x), bool); i = 0
    while i < len(lin):
        if lin[i]:
            j = i
            while j < len(lin) and lin[j]: j += 1
            if j - i >= min_run: out[i:j + 2] = True
            i = j
        else:
            i += 1
    return out

GRID   = "2min"    # fixed time step: 15 lags = the previous 30 minutes, in every season
GAP_S  = 1800      # outages up to 30 min are bridged; a longer outage starts a new segment
WARMUP_THETA_OK = 0.40

def prepare_site(csv, dev_id, warmup_theta=None, label=""):
    """Raw telemetry -> fixed 2-min grid with 30-min rain depth, segments and gap-compressed time."""
    raw = pd.read_csv(csv, low_memory=False)
    if "theta" not in raw.columns: raw = raw.rename(columns={"soil": "theta"})
    if "devID" in raw.columns: raw = raw[raw["devID"].astype(int) == dev_id]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed", errors="coerce")
    raw = raw.dropna(subset=["timestamp", "theta", "rain"]).sort_values("timestamp")
    n0 = len(raw); raw = raw.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    n_dup = n0 - len(raw)
    raw["rain_inc"] = counter_to_increments(raw["rain"].values)
    raw["R30"] = raw.set_index("timestamp")["rain_inc"].rolling("30min").sum().values   # mm / 30 min
    raw["fill"] = flag_linear_fill(raw["theta"].values)
    i0 = int(np.argmax(raw["theta"].values >= warmup_theta)) if warmup_theta else 0     # sensor warm-up
    raw = raw.iloc[i0:].reset_index(drop=True)
    raw["bin"] = raw["timestamp"].dt.floor(GRID)
    g = (raw.groupby("bin").agg(theta=("theta", "last"), rain=("R30", "last"),
                                rain_inc=("rain_inc", "sum"), fill=("fill", "all"))
            .reset_index().rename(columns={"bin": "timestamp"}))
    n_fill = int(g["fill"].sum())
    g = g.loc[~g["fill"]].drop(columns="fill").reset_index(drop=True)
    _dt = g["timestamp"].diff().dt.total_seconds().fillna(0.0)
    g["seg"] = (_dt > GAP_S).cumsum().astype(int)
    # Short outages (LoRaWAN packet loss, <= 30 min): every missing 2-min point gets the last
    # received reading, as the operational system would see it. These "held" points are used
    # only as lagged INPUTS, never as training targets or in any score.
    parts = []
    for _, sg in g.groupby("seg"):
        full = pd.date_range(sg["timestamp"].iloc[0], sg["timestamp"].iloc[-1], freq=GRID)
        sg = sg.set_index("timestamp").reindex(full)
        sg["held"] = sg["theta"].isna()
        sg[["theta", "rain", "seg"]] = sg[["theta", "rain", "seg"]].ffill()
        sg["rain_inc"] = sg["rain_inc"].fillna(0.0)
        parts.append(sg.rename_axis("timestamp").reset_index())
    n_real = len(g); g = pd.concat(parts, ignore_index=True); g["seg"] = g["seg"].astype(int)
    _dt = g["timestamp"].diff().dt.total_seconds().fillna(0.0)
    g["t_sec"] = _dt.clip(upper=GAP_S).cumsum()          # long outages count as GAP_S on the model time axis
    print(f"  [{label}] {n0:,} raw rows | {n_dup:,} duplicates | "
          f"warm-up {i0} rows | {n_real:,} measured 2-min points ({n_fill:,} filled-in points dropped, "
          f"{int(g['held'].sum()):,} short-outage points held as inputs only) | "
          f"{g['timestamp'].min():%Y-%m-%d} -> {g['timestamp'].max():%Y-%m-%d}")
    print(f"  [{label}] rain {raw['rain_inc'].sum():.0f} mm | max 30-min depth {raw['R30'].max():.1f} mm | "
          f"θ {g['theta'].min():.3f}–{g['theta'].max():.3f} | {int(g['seg'].max()) + 1} segments, "
          f"longest gap {_dt.max() / 86400:.1f} d")
    return g

