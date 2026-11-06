"""Physics-only baseline: 1-D Richards equation (mixed form, modified Picard; Celia et al., 1990).
z positive downward, 0 <= z <= L. Top: rain flux capped at Ks (excess runs off). Bottom: free drainage.
Same van Genuchten–Mualem parameters and domain as the PINN."""
import numpy as np


class Column:
    def __init__(self, vg, L=0.5, dz=0.01, z_sensor=0.30):
        self.tr, self.ts, self.a, self.n = vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n"]
        self.m = 1 - 1 / self.n; self.Ks = vg["Ks"] / 3600.0
        self.z = np.arange(0, L + 1e-9, dz); self.dz = dz; self.N = len(self.z)
        self.i_s = int(round(z_sensor / dz))

    # constitutive functions (h in m, negative unsaturated)
    def theta(self, h):
        Se = np.where(h < 0, (1 + np.abs(self.a * h) ** self.n) ** (-self.m), 1.0)
        return self.tr + (self.ts - self.tr) * Se

    def h_of_theta(self, th):
        Se = np.clip((th - self.tr) / (self.ts - self.tr), 1e-6, 1 - 1e-9)
        return -((Se ** (-1 / self.m) - 1) ** (1 / self.n)) / self.a

    def K(self, h):
        Se = np.where(h < 0, (1 + np.abs(self.a * h) ** self.n) ** (-self.m), 1.0)
        return self.Ks * np.sqrt(Se) * (1 - (1 - Se ** (1 / self.m)) ** self.m) ** 2

    def C(self, h):
        x = np.abs(self.a * h) ** self.n
        c = (self.ts - self.tr) * self.a * self.n * self.m * np.abs(self.a * h) ** (self.n - 1) * (1 + x) ** (-self.m - 1)
        return np.where(h < 0, c, 1e-8) + 1e-10

    def step(self, h, dt, q_top):
        """Advance h by dt (s) with downward top flux q_top (m/s, >= 0). Returns h, converged."""
        th_old = self.theta(h); hm = h.copy(); dz = self.dz; N = self.N
        for _ in range(30):
            Kn = self.K(hm); Kh = 0.5 * (Kn[1:] + Kn[:-1])          # interface conductivities
            Cm = self.C(hm); thm = self.theta(hm)
            A = np.zeros(N); B = np.zeros(N); Cc = np.zeros(N); R = np.zeros(N)
            # interior: d(theta)/dt = d/dz[K(dh/dz - 1)]  (z downward)
            for_i = slice(1, N - 1)
            A[for_i] = -Kh[:-1] / dz**2
            Cc[for_i] = -Kh[1:] / dz**2
            B[for_i] = Cm[1:-1] / dt + (Kh[:-1] + Kh[1:]) / dz**2
            R[for_i] = (Cm[1:-1] * hm[1:-1] - (thm[1:-1] - th_old[1:-1])) / dt - (Kh[1:] - Kh[:-1]) / dz
            # top cell (half cell): inflow q_top, outflow K(dh/dz - 1) downward flux = -K(dh/dz-1)
            B[0] = Cm[0] / dt * dz / 2 + Kh[0] / dz; Cc[0] = -Kh[0] / dz
            R[0] = (Cm[0] * hm[0] - (thm[0] - th_old[0])) / dt * dz / 2 + q_top - Kh[0]
            # bottom (half cell): free drainage outflow K(h_N)
            A[-1] = -Kh[-1] / dz; B[-1] = Cm[-1] / dt * dz / 2 + Kh[-1] / dz
            R[-1] = (Cm[-1] * hm[-1] - (thm[-1] - th_old[-1])) / dt * dz / 2 + Kh[-1] - Kn[-1]
            hn = thomas(A, B, Cc, R)
            hn = np.minimum(hn, 0.0)                                   # ponding not modelled
            if np.max(np.abs(hn - hm)) < 1e-4:
                return hn, True
            hm = hn
        return hm, False

    def advance(self, h, T, q_top, dt0=30.0):
        """Advance by T seconds with constant top flux; adaptive sub-steps."""
        t, dt = 0.0, dt0
        q = min(q_top, self.Ks)
        while t < T - 1e-9:
            d = min(dt, T - t)
            hn, ok = self.step(h, d, q)
            if ok:
                h = hn; t += d; dt = min(dt * 1.5, 600.0)
            else:
                dt = d / 3
                if dt < 1e-3:
                    h = hn; t += d
        return h


def thomas(a, b, c, d):
    n = len(d); cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.zeros(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
