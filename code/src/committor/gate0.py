"""Gate 0 check: Monte Carlo committor labels against the exact 1D committor."""

import numpy as np
from scipy.stats import binom
from tqdm.auto import tqdm

from .ranking import binom_ll, feature_skills_under_labels, sat_ll


def analytic_label_check(qa, n, M, crit, seed, progress=True):
    """Compare counts n ~ Binomial(M, q) with the exact q (qa).

    deviance D = 2 sum [LL_sat - LL(qa)], null distribution by parametric bootstrap from Binomial(M, qa);
    pooled z = sum (n - M qa) / sqrt(sum M qa (1 - qa));  per-q-bin z likewise;
    randomised PIT (uniform under the null) and 95% central-interval coverage for display.
    """
    qa = np.asarray(qa, float)
    n = np.asarray(n, float)
    M = np.asarray(M, float)
    Mi = M.astype(np.int64)
    rng = np.random.default_rng(seed)

    def dev(nn):
        return 2.0 * float(np.sum(sat_ll(nn, M) - binom_ll(qa, nn, M)))

    lo = binom.ppf(0.025, Mi, qa)
    hi = binom.ppf(0.975, Mi, qa)
    covered = (n >= lo) & (n <= hi)

    D = dev(n)
    B = int(crit["n_parametric_bootstrap"])
    Dpb = np.empty(B)
    cov_pb = np.empty(B)
    for b in tqdm(range(B), desc="parametric bootstrap", disable=not progress, mininterval=2.0):
        nb = rng.binomial(Mi, qa).astype(float)
        Dpb[b] = dev(nb)
        cov_pb[b] = np.mean((nb >= lo) & (nb <= hi))
    p_dev = (1.0 + np.sum(Dpb >= D)) / (1.0 + B)

    var = M * qa * (1.0 - qa)
    z_pool = float(np.sum(n - M * qa) / np.sqrt(np.sum(var)))

    nbins = int(crit["n_qbins"])
    edges = np.linspace(0.0, 1.0, nbins + 1)
    bidx = np.clip(np.digitize(qa, edges) - 1, 0, nbins - 1)
    bins = []
    for k in range(nbins):
        m = bidx == k
        cnt = int(m.sum())
        rec = {"q_lo": float(edges[k]), "q_hi": float(edges[k + 1]), "n_states": cnt}
        if cnt:
            rec["mean_q_analytic"] = float(qa[m].mean())
            rec["mean_q_mc"] = float((n[m] / M[m]).mean())
            rec["z"] = float(np.sum(n[m] - M[m] * qa[m]) / np.sqrt(np.sum(var[m])))
            rec["used_in_test"] = cnt >= int(crit["min_states_per_bin"])
        bins.append(rec)
    zb = [b["z"] for b in bins if b.get("used_in_test")]
    max_bin_z = float(np.max(np.abs(zb))) if zb else float("nan")

    u = rng.random(len(n))
    pit = binom.cdf(n - 1, Mi, qa) + u * binom.pmf(n, Mi, qa)

    passed = {
        "deviance_pb_p": bool(p_dev >= crit["deviance_pb_p_min"]),
        "pooled_bias_z": bool(abs(z_pool) <= crit["pooled_bias_abs_z_max"]),
        "max_qbin_z": bool(max_bin_z <= crit["qbin_abs_z_max"]),
    }
    evaluable = len(zb) > 0
    return {
        "n_states": int(len(n)),
        "deviance": D,
        "deviance_per_state": D / len(n),
        "deviance_pb_mean": float(Dpb.mean()),
        "deviance_pb_p": float(p_dev),
        "pooled_bias_z": z_pool,
        "pooled_mean_residual": float(np.mean(n / M - qa)),
        "max_abs_residual": float(np.max(np.abs(n / M - qa))),
        "max_abs_qbin_z": max_bin_z,
        "qbins": bins,
        "coverage95": float(covered.mean()),
        "coverage95_pb_p01_p99": [float(np.quantile(cov_pb, 0.01)), float(np.quantile(cov_pb, 0.99))],
        "pit_hist": np.histogram(pit, bins=20, range=(0, 1))[0].tolist(),
        "passed": passed,
        "n_qbins_evaluable": len(zb),
        "evaluable": bool(evaluable),
        "pass": bool(evaluable and all(passed.values())),
    }, pit


def exact_label_null(system, X, split, qa, M, feature_names, R, seed, iso_pseudo, progress=True):
    """Skill of each feature (and of the exact q itself) when labels are drawn from the exact committor.

    n* ~ Binomial(M, q_exact) on the same states and split, R replicates. This needs no MC labels, so
    it is computed before labeling and serves as the pre-registered prediction for the MC-label skills.
    """
    rng = np.random.default_rng(seed)
    Mi = np.asarray(M, np.int64)
    tr, te = split == 0, split == 2
    out = {nm: [] for nm in feature_names}
    out["exact q [exact]"] = []
    for _ in tqdm(range(int(R)), desc="exact-label null", disable=not progress, mininterval=2.0):
        nb = rng.binomial(Mi, qa).astype(float)
        sk = feature_skills_under_labels(system, X, split, nb, Mi.astype(float), feature_names, iso_pseudo)
        for k, v in sk.items():
            out[k].append(v)
        p0 = nb[tr].sum() / Mi[tr].sum()
        llc = binom_ll(np.full(len(nb), p0), nb, Mi)
        lls = sat_ll(nb, Mi)
        out["exact q [exact]"].append(float((binom_ll(qa, nb, Mi)[te] - llc[te]).sum() / (lls[te] - llc[te]).sum()))
    summ = {}
    for k, v in out.items():
        v = np.asarray(v)
        summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                   "q005": float(np.quantile(v, 0.005)), "q025": float(np.quantile(v, 0.025)),
                   "q975": float(np.quantile(v, 0.975)), "q995": float(np.quantile(v, 0.995)),
                   "values": v.tolist()}
    return summ


def edge_bin_pattern(qa, n, M, nbins=10):
    """Per-q-bin z of MC counts against the exact q (used to display discrete-monitoring bias)."""
    qa, n, M = (np.asarray(a, float) for a in (qa, n, M))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    bidx = np.clip(np.digitize(qa, edges) - 1, 0, nbins - 1)
    var = M * qa * (1.0 - qa)
    zs = []
    for k in range(nbins):
        m = bidx == k
        zs.append(float(np.sum(n[m] - M[m] * qa[m]) / np.sqrt(np.sum(var[m]))) if m.any() else None)
    return {"edges": edges.tolist(), "z": zs,
            "pooled_z": float(np.sum(n - M * qa) / np.sqrt(np.sum(var))),
            "mean_residual_q_lt_0.1": float(np.mean((n / M - qa)[qa < 0.1])) if np.any(qa < 0.1) else None,
            "mean_residual_q_gt_0.9": float(np.mean((n / M - qa)[qa > 0.9])) if np.any(qa > 0.9) else None}
