"""Small statistics helpers: Clopper-Pearson intervals and probit fits of q profiles."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.stats import norm

Z95 = 1.6448536269514722      # width(5-95%) = 2 * Z95 * w for q = Phi((s - s50) / w)


def cp_interval(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def probit_fit(s, k, M):
    """MLE of q(s) = Phi((s - s50) / w) for counts k out of M (arrays). Returns s50, w."""
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    M = np.broadcast_to(np.asarray(M, dtype=float), s.shape)
    s0, sc = s.mean(), s.std() if s.std() > 0 else 1.0

    def nll(p):
        q = np.clip(norm.cdf(((s - s0) / sc - p[0]) / np.exp(p[1])), 1e-12, 1 - 1e-12)
        return -(k * np.log(q) + (M - k) * np.log1p(-q)).sum()
    r = minimize(nll, [0.0, np.log(0.3)], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000})
    return s0 + sc * r.x[0], sc * np.exp(r.x[1])


def probit_with_se(s, k, M, n_boot=200, seed=0):
    """Probit s50 and width(5-95%) with parametric-bootstrap standard errors."""
    rng = np.random.default_rng(seed)
    s50, w = probit_fit(s, k, M)
    Mi = np.broadcast_to(np.asarray(M, dtype=int), np.shape(s))
    bs = np.array([probit_fit(s, rng.binomial(Mi, norm.cdf((np.asarray(s) - s50) / w)), M) for _ in range(n_boot)])
    return {"s50": float(s50), "s50_se": float(bs[:, 0].std()), "width": float(2 * Z95 * w),
            "width_se": float(2 * Z95 * bs[:, 1].std())}


def iso_crossings(s, k, M, levels=(0.05, 0.5, 0.95)):
    """Model-free: non-decreasing binomial-weighted least-squares fit of q(s) (PAVA), then the first s at which
    the fit reaches each level, by linear interpolation. NaN where a level is not reached."""
    from sklearn.isotonic import IsotonicRegression
    s = np.asarray(s, dtype=float)
    q = np.asarray(k, dtype=float) / np.broadcast_to(np.asarray(M, dtype=float), s.shape)
    f = IsotonicRegression(y_min=0, y_max=1).fit(s, q, sample_weight=np.broadcast_to(np.asarray(M, dtype=float), s.shape)).predict(s)
    out = []
    for L in levels:
        j = int(np.argmax(f >= L))
        if f[j] < L:
            out.append(float("nan"))
        elif j == 0:
            out.append(float(s[0]))
        else:
            out.append(float(s[j - 1] + (L - f[j - 1]) / (f[j] - f[j - 1]) * (s[j] - s[j - 1])))
    return out


def iso_with_se(s, k, M, n_boot=400, seed=0):
    """Isotonic s50 and width (5-95%) with a nonparametric binomial bootstrap (resample counts at each point)."""
    rng = np.random.default_rng(seed)
    Mi = np.broadcast_to(np.asarray(M, dtype=int), np.shape(s))
    q = np.asarray(k, dtype=float) / Mi
    s05, s50, s95 = iso_crossings(s, k, Mi)
    bs = np.array([iso_crossings(s, rng.binomial(Mi, q), Mi) for _ in range(n_boot)])
    w = bs[:, 2] - bs[:, 0]
    return {"s50": s50, "s50_se": float(np.nanstd(bs[:, 1])), "width": s95 - s05, "width_se": float(np.nanstd(w))}


def probit_lapse_fit(s, k, M):
    """MLE of q(s) = (1 - lam) Phi((s - s50') / w) with an upper lapse lam in [0, 0.2] (q tends to 1 - lam);
    s50 reported as the s where the fitted q = 0.5. Returns s50, width(5-95% of the fitted curve), lam."""
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    M = np.broadcast_to(np.asarray(M, dtype=float), s.shape)
    s0, sc = s.mean(), s.std() if s.std() > 0 else 1.0

    def nll(p):
        lam = 0.2 / (1 + np.exp(-p[2]))
        q = np.clip((1 - lam) * norm.cdf(((s - s0) / sc - p[0]) / np.exp(p[1])), 1e-12, 1 - 1e-12)
        return -(k * np.log(q) + (M - k) * np.log1p(-q)).sum()
    r = minimize(nll, [0.0, np.log(0.3), -3.0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 8000})
    lam = 0.2 / (1 + np.exp(-r.x[2]))
    w = sc * np.exp(r.x[1])
    mu = s0 + sc * r.x[0]
    inv = lambda L: mu + w * norm.ppf(min(L / (1 - lam), 1 - 1e-12))
    s50 = inv(0.5)
    width = (inv(0.95) - inv(0.05)) if lam < 0.05 else float("nan")
    return float(s50), float(width), float(lam)
