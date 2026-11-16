"""Threshold-prediction metric (Gate 3 primary metric, user decision of 2026-11-15).

For a coordinate c tabulated along rays on an increasing amplitude grid s_1 < ... < s_G (C[r, j] = c(lam + s_j d_r)):
s_pred(c*) is the first amplitude at which side * c reaches side * c* (side = +1: B where c >= c*), with linear
interpolation of ln s between the two grid points that bracket the first crossing. If the first grid point already
satisfies the rule, s_pred = s_1 (flag below_grid); if no grid point does, s_pred = +inf (|ln| = +inf, counted).

c* rule: minimise the median of |ln(s_pred / s_c)| over the training rays; candidate thresholds are quantiles of
the coordinate's values at the training edges (c interpolated at s_c), then a local refinement between the two
quantiles adjacent to the best one. The side is the one with the lower training median.
"""
import numpy as np
import torch


def _t(x, device):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)


def clean(C):
    """NaN kept (handled per side in RayCoordinate: an undefined value never reaches a threshold on either side);
    +inf kept (overflowed gain: above every threshold)."""
    return np.array(C, dtype=np.float64, copy=True)


def qinf(x, q):
    """Quantile with numpy's default linear interpolation, made safe for +-inf entries (np.quantile returns NaN when
    it interpolates towards an infinite neighbour). x: 1-d; q: scalar or sequence. NaN entries are not allowed."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    qs = np.atleast_1d(np.asarray(q, dtype=np.float64))
    out = np.empty(len(qs))
    for k, qq in enumerate(qs):
        pos = qq * (n - 1)
        i = int(np.floor(pos))
        t_ = pos - i
        lo, hi = x[i], x[min(i + 1, n - 1)]
        if t_ == 0.0 or lo == hi:
            out[k] = lo
        elif lo == -np.inf:
            out[k] = -np.inf if hi < np.inf else np.nan
        elif hi == np.inf:
            out[k] = np.inf
        else:
            out[k] = lo + t_ * (hi - lo)
    return out if np.ndim(q) else float(out[0])


def _dinf(a, b):
    """a - b with inf - inf := 0 (ties at infinity)."""
    return 0.0 if (np.isinf(a) and a == b) else a - b


def median0(E):
    """Median along dim 0, midpoint convention of np.median (the torch median op returns the lower middle value)."""
    s = torch.sort(E, dim=0).values
    n = E.shape[0]
    return 0.5 * (s[(n - 1) // 2] + s[n // 2])


class RayCoordinate:
    """c tabulated on (R, G); thresholds applied in units of side * c."""

    def __init__(self, C, s_grid, device="cuda"):
        self.device = device
        self.C = _t(clean(C), device)
        self.ln_s = _t(np.log(s_grid), device)
        self.G = self.C.shape[1]
        self._cm = {}
        self._sc = {}

    def sc(self, side):
        """side * C with NaN -> -inf (an undefined value never reaches a threshold, whichever the side)."""
        if side not in self._sc:
            S = side * self.C
            self._sc[side] = torch.where(torch.isnan(S), torch.full_like(S, -float("inf")), S)
        return self._sc[side]

    def cm(self, side):
        if side not in self._cm:
            self._cm[side] = torch.cummax(self.sc(side), dim=1).values.contiguous()
        return self._cm[side]

    def ln_s_pred(self, thr, side, rows=None, extrapolate_below=False):
        """thr: (T,) thresholds in side*c units. Returns (R', T) ln s_pred (+inf: no crossing), below-grid mask.
        extrapolate_below: where the first grid point already satisfies the rule, extend c linearly in ln s from the
        first two grid points instead of returning ln s_1 (descriptive only; the metric itself is defined on the grid)."""
        if rows is not None:
            rows = torch.as_tensor(np.asarray(rows) if not torch.is_tensor(rows) else rows, device=self.device)
        Cs = self.sc(side) if rows is None else self.sc(side)[rows]
        Cm = self.cm(side) if rows is None else self.cm(side)[rows]
        thr = _t(np.atleast_1d(thr), self.device)
        vals = thr[None, :].expand(Cs.shape[0], -1).contiguous()
        j = torch.searchsorted(Cm, vals)                       # first index with running max >= thr
        no = j >= self.G
        jc = j.clamp(max=self.G - 1)
        jl = (jc - 1).clamp(min=0)
        c_hi = torch.gather(Cs, 1, jc)
        c_lo = torch.gather(Cs, 1, jl)
        den = c_hi - c_lo
        w_lin = (vals - c_lo) / den
        w = torch.where(torch.isfinite(den) & (den > 0), w_lin,
                        torch.where(c_lo == -float("inf"), torch.ones_like(vals), torch.zeros_like(vals))).clamp(0.0, 1.0)
        ln = self.ln_s[jl] + w * (self.ln_s[jc] - self.ln_s[jl])
        below = j == 0
        if extrapolate_below:
            c0, c1 = Cs[:, :1], Cs[:, 1:2]
            d01 = c1 - c0
            ok = torch.isfinite(c0) & torch.isfinite(d01) & (d01 > 0)
            ln_x = self.ln_s[0] + (vals - c0) / torch.where(ok, d01, torch.ones_like(d01)) * (self.ln_s[1] - self.ln_s[0])
            ln0 = torch.where(ok, torch.minimum(ln_x, self.ln_s[0].expand_as(ln_x)), self.ln_s[0].expand_as(ln_x))
        else:
            ln0 = self.ln_s[0].expand_as(ln)
        ln = torch.where(below, ln0, ln)
        ln = torch.where(no, torch.full_like(ln, float("inf")), ln)
        return ln, below

    def values_at(self, ln_s_query, rows=None):
        """c interpolated (linearly in ln s) at one amplitude per ray (rays = rows, default all)."""
        q = _t(ln_s_query, self.device)
        j = torch.searchsorted(self.ln_s, q).clamp(1, self.G - 1)
        r = torch.arange(self.C.shape[0], device=self.device) if rows is None else torch.as_tensor(rows, device=self.device)
        lo, hi = self.C[r, j - 1], self.C[r, j]
        w = ((q - self.ln_s[j - 1]) / (self.ln_s[j] - self.ln_s[j - 1])).clamp(0.0, 1.0)
        out = lo + w * (hi - lo)
        out = torch.where(torch.isfinite(lo) & torch.isfinite(hi), out, torch.where(w > 0.5, hi, lo))
        return out.cpu().numpy()


def abs_err(ln_pred, ln_sc):
    e = torch.abs(ln_pred - _t(ln_sc, ln_pred.device)[:, None])
    return torch.where(torch.isnan(e), torch.full_like(e, float("inf")), e)


def fit_cstar(rc, rows, ln_sc, n_levels=1000, n_refine=1000, sides=(1, -1)):
    """rows: training ray indices into rc; ln_sc: ln s_c of those rows (aligned). Returns dict(c_star, side, ...)."""
    rows_t = torch.as_tensor(rows, device=rc.device)
    edge = rc.values_at(np.asarray(ln_sc), rows)
    best = None
    for side in sides:
        ev = side * edge
        ev = ev[np.isfinite(ev)]
        if len(ev) == 0:
            continue
        qs = np.unique(np.quantile(ev, np.linspace(0.0, 1.0, n_levels)))
        ln_p, _ = rc.ln_s_pred(qs, side, rows_t)
        med = median0(abs_err(ln_p, ln_sc)).cpu().numpy()
        k = int(np.argmin(med))
        lo, hi = qs[max(k - 1, 0)], qs[min(k + 1, len(qs) - 1)]
        fine = np.unique(np.concatenate([np.linspace(lo, hi, n_refine), [qs[k]]]))
        ln_p, _ = rc.ln_s_pred(fine, side, rows_t)
        medf = median0(abs_err(ln_p, ln_sc)).cpu().numpy()
        kf = int(np.argmin(medf))
        cand = (float(medf[kf]), side, float(fine[kf]))
        if best is None or cand[0] < best[0] - 1e-15:
            best = cand
    med, side, thr = best
    return {"c_star": side * thr, "side": side, "thr": thr, "train_median_abs": med}


def summarize(ln_err, boot_idx=None):
    """ln_err: signed ln(s_pred/s_ref) (numpy, +inf allowed). Returns metrics and optional bootstrap CIs
    (all quantiles inf-safe, see qinf)."""
    e = np.asarray(ln_err, dtype=np.float64)
    a = np.abs(e)

    def stats(ee, aa):
        return float(np.median(aa)), qinf(aa, 0.9), float(np.median(ee))

    med, q90, sgn = stats(e, a)
    out = {"n": int(len(e)), "median_abs": med, "q90_abs": q90, "signed_median": sgn,
           "frac_no_crossing": float(np.mean(np.isposinf(e))), "mean_abs_finite": float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else None}
    if boot_idx is not None:
        bs = np.array([stats(e[i], a[i]) for i in boot_idx])
        for k, name in enumerate(["median_abs", "q90_abs", "signed_median"]):
            lo, hi = qinf(bs[:, k], [0.025, 0.975])
            out[name + "_ci"] = [float(lo), float(hi)]
    return out


def paired_diff(err_a, err_b, boot_idx):
    """Bootstrap CI of median|a| - median|b| and q90|a| - q90|b| over the same resamples (inf-safe; inf - inf := 0).
    The components (median and q90 of |a| and |b|, all >= 0, so a JSON null there means +inf) are stored as well,
    so that the sign of an infinite difference is recoverable from results.json."""
    a, b = np.abs(np.asarray(err_a, dtype=np.float64)), np.abs(np.asarray(err_b, dtype=np.float64))

    def comp(i):
        return np.median(a[i]), np.median(b[i]), qinf(a[i], 0.9), qinf(b[i], 0.9)

    def d(i):
        ma, mb, qa, qb = comp(i)
        return _dinf(ma, mb), _dinf(qa, qb)

    full = np.arange(len(a))
    ma, mb, qa, qb = comp(full)
    pt = d(full)
    bs = np.array([d(i) for i in boot_idx])
    out = {"median_abs_a": float(ma), "median_abs_b": float(mb), "q90_abs_a": float(qa), "q90_abs_b": float(qb)}
    for k, name in enumerate(["median_abs_diff", "q90_abs_diff"]):
        col = bs[:, k]
        lo, hi = qinf(col, [0.025, 0.975])
        out[name] = float(pt[k])
        out[name + "_ci"] = [float(lo), float(hi)]
        out[name + "_p_le0"] = float(np.mean(col <= 0))
        out[name + "_frac_minus_inf"] = float(np.mean(col == -np.inf))
        out[name + "_frac_plus_inf"] = float(np.mean(col == np.inf))
    return out
