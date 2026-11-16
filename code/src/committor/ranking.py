"""Reaction-coordinate ranking by out-of-sample binomial likelihood.

For a candidate coordinate c(x):
  1. learned candidates (logistic / MLP) produce c(x) from train labels (val for model selection);
  2. a monotone map c -> q is fitted on train by isotonic regression (PAVA; this is the binomial
     maximum-likelihood monotone fit), direction chosen by train likelihood, Jeffreys pseudo-counts
     inside each block, linear interpolation between block centres;
  3. test binomial log-likelihood LL_c = sum_i n_i log p_i + (M_i - n_i) log(1 - p_i).

skill = (LL_c - LL_const) / (LL_sat - LL_const), with LL_const the constant train-mean prediction and
LL_sat each state's own n_i / M_i. Bootstrap CIs resample test groups (rays) with replacement; the
same resamples are shared by every candidate.
"""

import copy
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import xlogy
from scipy.stats import binom
from tqdm.auto import tqdm

from .utils import derive_seed

P_CLIP = 1e-12


# ---- likelihoods ------------------------------------------------------------------------
def binom_ll(p, n, M):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
    return n * np.log(p) + (M - n) * np.log1p(-p)


def sat_ll(n, M):
    q = n / M
    return xlogy(n, q) + xlogy(M - n, 1.0 - q)


# ---- splits and bootstrap -----------------------------------------------------------------
def group_split(groups, fracs, seed):
    """Assign whole groups to train/val/test (0/1/2) so that state fractions approximate fracs."""
    rng = np.random.default_rng(seed)
    ug, counts = np.unique(groups, return_counts=True)
    perm = rng.permutation(len(ug))
    ug, counts = ug[perm], counts[perm]
    tot = counts.sum()
    mid = (np.cumsum(counts) - 0.5 * counts) / tot
    c1, c2 = fracs[0], fracs[0] + fracs[1]
    gsplit = np.where(mid < c1, 0, np.where(mid < c2, 1, 2)).astype(np.int8)
    o = np.argsort(ug)
    return gsplit[o][np.searchsorted(ug[o], groups)]


class GroupBootstrap:
    def __init__(self, groups, B, seed):
        self.ug, self.inv = np.unique(groups, return_inverse=True)
        G = len(self.ug)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, G, size=(B, G))
        self.counts = np.stack([np.bincount(r, minlength=G) for r in draws]).astype(np.float64)

    def skills(self, num, den):
        G = len(self.ug)
        ng = np.bincount(self.inv, weights=num, minlength=G)
        dg = np.bincount(self.inv, weights=den, minlength=G)
        return (self.counts @ ng) / (self.counts @ dg)

    def ci(self, num, den, level=0.95):
        s = self.skills(num, den)
        a = (1.0 - level) / 2.0
        return float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))


# ---- isotonic map c -> q --------------------------------------------------------------------
def pava_blocks(c, n, M):
    """Increasing binomial PAVA. Returns block (sum_n, sum_M, centre) and each input point's block."""
    c = np.asarray(c, dtype=np.float64)
    order = np.argsort(c, kind="mergesort")
    cs, ns, Ms = c[order], np.asarray(n, float)[order], np.asarray(M, float)[order]
    uc, start, inv_u = np.unique(cs, return_index=True, return_inverse=True)
    sn = np.add.reduceat(ns, start)
    sM = np.add.reduceat(Ms, start)
    scM = np.add.reduceat(cs * Ms, start)
    bn, bM, bcM, bfirst = [], [], [], []
    for i in range(len(uc)):
        bn.append(sn[i]); bM.append(sM[i]); bcM.append(scM[i]); bfirst.append(i)
        while len(bn) > 1 and bn[-2] * bM[-1] >= bn[-1] * bM[-2]:
            n_, M_, cm_ = bn.pop(), bM.pop(), bcM.pop()
            bfirst.pop()
            bn[-1] += n_; bM[-1] += M_; bcM[-1] += cm_
    bn, bM, bcM = np.array(bn), np.array(bM), np.array(bcM)
    ublock = np.searchsorted(np.array(bfirst), np.arange(len(uc)), side="right") - 1
    point_block = np.empty(len(c), np.int64)
    point_block[order] = ublock[inv_u]
    return bn, bM, bcM / bM, point_block


class IsotonicBinomial:
    def __init__(self, pseudo=0.5):
        self.pseudo = float(pseudo)

    def _fit_inc(self, c, n, M):
        bn, bM, centres, _ = pava_blocks(c, n, M)
        p = (bn + self.pseudo) / (bM + 2.0 * self.pseudo)
        p = np.maximum.accumulate(p)
        return centres, p

    def fit(self, c, n, M):
        best = None
        for sign in (1.0, -1.0):
            centres, p = self._fit_inc(sign * np.asarray(c, float), n, M)
            ll = binom_ll(np.interp(sign * np.asarray(c, float), centres, p), n, M).sum()
            if best is None or ll > best[0]:
                best = (ll, sign, centres, p)
        self.train_ll, self.sign, self.centres, self.p = best
        return self

    def predict(self, c):
        return np.interp(self.sign * np.asarray(c, float), self.centres, self.p)


# ---- learned coordinates ------------------------------------------------------------------
def poly_features(Fm, degree):
    cols = [Fm[:, i] for i in range(Fm.shape[1])]
    names = [f"f{i}" for i in range(Fm.shape[1])]
    if degree >= 2:
        for i, j in combinations_with_replacement(range(Fm.shape[1]), 2):
            cols.append(Fm[:, i] * Fm[:, j])
            names.append(f"f{i}*f{j}")
    if degree > 2:
        raise ValueError("degree > 2 not supported")
    return np.stack(cols, axis=1), names


def _nll(z, n, M):
    return (n * F.softplus(-z) + (M - n) * F.softplus(z)).sum() / M.sum()


def _standardise(A, mask):
    mu = A[mask].mean(0)
    sd = A[mask].std(0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, mu, sd


def _tensors(device, *arrs):
    return [torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device) for a in arrs]


def fit_logistic(Phi, n, M, split, l2_grid, device):
    tr = split == 0
    va = split == 1 if np.any(split == 1) else tr
    Z, mu, sd = _standardise(Phi, tr)
    Zt, nt, Mt = _tensors(device, Z, n, M)
    trm, vam = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    p0 = n[tr].sum() / M[tr].sum()
    b0 = float(np.log(p0 / (1.0 - p0)))
    best = None
    for lam in l2_grid:
        w = torch.zeros(Zt.shape[1], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([b0], dtype=torch.float64, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=2000, tolerance_grad=1e-12,
                                tolerance_change=1e-14, history_size=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = _nll(Zt[trm] @ w + b, nt[trm], Mt[trm]) + lam * (w * w).sum()
            loss.backward()
            return loss

        for _ in range(5):
            opt.step(closure)
        with torch.no_grad():
            v = _nll(Zt[vam] @ w + b, nt[vam], Mt[vam]).item()
            z = (Zt @ w + b).cpu().numpy()
        if best is None or v < best[0]:
            best = (v, lam, z, w.detach().cpu().numpy(), float(b.item()))
    v, lam, z, w, b = best
    return z, {"l2": lam, "val_nll": v, "coef_standardised": w, "intercept": b,
               "feature_mean": mu, "feature_std": sd}


def _make_mlp(d_in, hidden):
    dims = [d_in] + list(hidden)
    layers = []
    for i in range(len(dims) - 1):
        layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.SiLU()]
    layers.append(torch.nn.Linear(dims[-1], 1))
    return torch.nn.Sequential(*layers).to(torch.float64)


def fit_mlp(Fm, n, M, split, mcfg, seed, device, progress=True):
    tr = split == 0
    va = split == 1 if np.any(split == 1) else tr
    Z, mu, sd = _standardise(Fm, tr)
    Zt, nt, Mt = _tensors(device, Z, n, M)
    trm, vam = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    best = None
    for s in range(int(mcfg["n_seeds"])):
        g = torch.Generator().manual_seed(derive_seed(seed, "init", s))
        net = _make_mlp(Fm.shape[1], mcfg["hidden"])
        with torch.no_grad():
            for m in net:
                if isinstance(m, torch.nn.Linear):
                    bound = 1.0 / np.sqrt(m.in_features)
                    m.weight.uniform_(-bound, bound, generator=g)
                    m.bias.uniform_(-bound, bound, generator=g)
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"]))
        bv, bstate, bep, wait = np.inf, None, -1, 0
        for ep in tqdm(range(int(mcfg["max_epochs"])), desc=f"MLP seed {s}", leave=False,
                       disable=not progress, mininterval=2.0):
            opt.zero_grad()
            loss = _nll(net(Zt[trm]).squeeze(1), nt[trm], Mt[trm])
            loss.backward()
            opt.step()
            with torch.no_grad():
                v = _nll(net(Zt[vam]).squeeze(1), nt[vam], Mt[vam]).item()
            if v < bv - 1e-12:
                bv, bep, wait = v, ep, 0
                bstate = copy.deepcopy(net.state_dict())
            else:
                wait += 1
                if wait >= int(mcfg["patience"]):
                    break
        if best is None or bv < best[0]:
            best = (bv, s, bstate, bep)
    net = _make_mlp(Fm.shape[1], mcfg["hidden"]).to(device)
    net.load_state_dict(best[2])
    with torch.no_grad():
        z = net(Zt).squeeze(1).cpu().numpy()
    return z, {"val_nll": best[0], "best_seed": best[1], "best_epoch": best[3],
               "net": net, "feature_mean": mu, "feature_std": sd}   # net/mean/std: to score new states (Gate 3)


# ---- candidates -------------------------------------------------------------------------
def feature_matrix(system, X, names):
    if names == "all":
        return np.asarray(X, dtype=np.float64)
    feats = system.features()
    return np.stack([feats[nm](X) for nm in names], axis=1)


def candidate_coordinate(system, X, cand, n_fit, M_fit, split, rcfg, seed, device, progress=True):
    kind = cand["type"]
    if kind == "feature":
        return system.features()[cand["feature"]](X), {}
    Fm = feature_matrix(system, X, cand["features"])
    if kind == "logistic":
        Phi, _ = poly_features(Fm, int(cand.get("degree", 1)))
        return fit_logistic(Phi, n_fit, M_fit, split, rcfg["logistic_l2_grid"], device)
    if kind == "mlp":
        return fit_mlp(Fm, n_fit, M_fit, split, rcfg["mlp"], derive_seed(seed, "mlp", cand["name"]),
                       device, progress)
    raise ValueError(f"unknown candidate type {kind!r}")


def expected_window_hist(p, M, bins):
    """Histogram of n/M expected if each state's true q equalled its prediction p (binomial noise only)."""
    out = np.zeros(len(bins) - 1)
    for pi, Mi in zip(p, M):
        k = np.arange(int(Mi) + 1)
        pmf = binom.pmf(k, int(Mi), pi)
        out += np.histogram(k / Mi, bins=bins, weights=pmf)[0]
    return out


def rank(system, data, rcfg, seed, device, references=(), progress=True):
    """Rank candidate coordinates. references: list of (name, callable X -> q) used as given, no fit."""
    X = data["X"]
    n = data["n_B"].astype(np.float64)
    M = data["M_eff"].astype(np.float64)
    groups = data["group"]
    split = group_split(groups, rcfg["split"], derive_seed(seed, "split"))
    tr, va, te = split == 0, split == 1, split == 2
    if va.sum() == 0:
        va = tr
    qhat = n / M
    p0 = n[tr].sum() / M[tr].sum()
    ll_const = binom_ll(np.full(len(n), p0), n, M)
    ll_sat = sat_ll(n, M)
    boot = GroupBootstrap(groups[te], int(rcfg["bootstrap"]), derive_seed(seed, "bootstrap"))
    hw = rcfg["hist_window"]
    bins = np.linspace(0.0, 1.0, int(rcfg.get("hist_bins", 20)) + 1)
    iso_pseudo = float(rcfg["iso_pseudo"])

    def sk(ll, m):
        return float((ll[m] - ll_const[m]).sum() / (ll_sat[m] - ll_const[m]).sum())

    rows, preds = [], {}

    origin = data.get("origin")
    te_ray = te & (origin == 0) if origin is not None else None
    te_traj = te & (origin == 1) if origin is not None else None

    def fit_map(c, n_fit, M_fit):
        iso = IsotonicBinomial(iso_pseudo).fit(c[tr], n_fit[tr], M_fit[tr])
        return iso.predict(c), ("+" if iso.sign > 0 else "-")

    def score(name, kind, c, n_fit=n, M_fit=M, extra=None, fixed_p=None, light=False):
        if fixed_p is None:
            p, direction = fit_map(c, n_fit, M_fit)
        else:
            p, direction = np.asarray(fixed_p, float), None
        ll = binom_ll(p, n, M)
        row = {"name": name, "kind": kind, "skill": sk(ll, te)}
        if light:
            return row
        row["ci95"] = list(boot.ci(ll[te] - ll_const[te], ll_sat[te] - ll_const[te]))
        row["interval"] = "test-group bootstrap"
        row["skill_train"] = sk(ll, tr)
        row["skill_val"] = sk(ll, va)
        if te_ray is not None and te_ray.any() and te_traj.any():
            row["skill_test_ray_states"] = sk(ll, te_ray)
            row["skill_test_traj_states"] = sk(ll, te_traj)
        row["direction"] = direction
        w = te & (p >= hw[0]) & (p <= hw[1])
        row["hist_n"] = int(w.sum())
        row["hist_counts"] = np.histogram(qhat[w], bins=bins)[0].tolist()
        row["hist_expected_if_exact"] = expected_window_hist(p[w], M[w], bins).tolist()
        row["hist_mean_q"] = float(qhat[w].mean()) if w.any() else None
        row["hist_std_q"] = float(qhat[w].std()) if w.any() else None
        # spread expected if every window state's true q equalled its prediction:
        # spread of predictions inside the window + binomial noise
        row["hist_std_if_exact"] = (float(np.sqrt(np.var(p[w]) + np.mean(p[w] * (1 - p[w]) / M[w])))
                                    if w.any() else None)
        if extra:
            row.update(extra)
        rows.append(row)
        preds[name] = {"c": np.asarray(c, float), "p": p}
        return row

    for cand in tqdm(rcfg["candidates"], desc="candidates", disable=not progress):
        c, info = candidate_coordinate(system, X, cand, n, M, split, rcfg, seed, device, progress)
        extra = {k: v for k, v in info.items() if k in ("l2", "val_nll", "best_seed", "best_epoch")}
        score(cand["name"], cand["type"], c, extra=extra)

    for name, fn in references:
        q_ref = fn(X)
        score(f"{name} [exact]", "reference", q_ref, fixed_p=q_ref)
        score(f"{name} [isotonic]", "reference", q_ref)

    # ---- controls ----
    # shuffled labels: (n, M) pairs permuted jointly within train and within val, the candidate is
    # refitted on the permuted labels and scored on the real test labels; repeated K times and
    # summarised by the permutation distribution (not by a test bootstrap of a single draw).
    ctl = rcfg["controls"]
    by_name = {c["name"]: c for c in rcfg["candidates"]}
    for nm in ctl.get("shuffled", []):
        cand = by_name[nm]
        learned = cand["type"] in ("logistic", "mlp")
        K = int(ctl["shuffle_permutations_learned"] if learned else ctl["shuffle_permutations_feature"])
        rng = np.random.default_rng(derive_seed(seed, "shuffle", nm))
        c_fixed = None if learned else candidate_coordinate(system, X, cand, n, M, split, rcfg, seed, device)[0]
        vals = []
        for k in tqdm(range(K), desc=f"shuffled: {nm}", disable=not progress, leave=False):
            n_s, M_s = n.copy(), M.copy()
            for m in (tr, va):
                ix = np.flatnonzero(m)
                pi = rng.permutation(ix)
                n_s[ix], M_s[ix] = n[pi], M[pi]
            if learned:
                c, _ = candidate_coordinate(system, X, cand, n_s, M_s, split, rcfg,
                                            derive_seed(seed, "shuf", nm, k), device, False)
            else:
                c = c_fixed
            p, _ = fit_map(c, n_s, M_s)
            vals.append(sk(binom_ll(p, n, M), te))
        vals = np.array(vals)
        rows.append({"name": f"shuffled labels: {nm}", "kind": "control", "skill": float(vals.mean()),
                     "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                     "interval": f"permutation distribution (K={K}), mean and 2.5-97.5%",
                     "skill_train": None, "n_permutations": K, "permutation_skills": vals.tolist()})

    K = int(ctl.get("n_random_projections", 0))
    rp = []
    if K:
        Z, _, _ = _standardise(np.asarray(X, float), tr)
        Wr = np.random.default_rng(derive_seed(seed, "randproj")).standard_normal((K, X.shape[1]))
        Wr /= np.linalg.norm(Wr, axis=1, keepdims=True)
        for k in tqdm(range(K), desc="random projections", disable=not progress):
            rp.append(score(f"randproj{k}", "control", Z @ Wr[k], light=True)["skill"])
    rp = np.array(rp)

    out = {
        "split_counts": {s: int((split == i).sum()) for i, s in enumerate(("train", "val", "test"))},
        "split_groups": {s: int(len(np.unique(groups[split == i]))) for i, s in enumerate(("train", "val", "test"))},
        "p0_train": float(p0),
        "test_den_per_state": float((ll_sat[te] - ll_const[te]).mean()),
        "rows": rows,
        "random_projections": {
            "n": int(len(rp)),
            "skills": rp.tolist(),
            "mean": float(rp.mean()) if len(rp) else None,
            "median": float(np.median(rp)) if len(rp) else None,
            "p05": float(np.quantile(rp, 0.05)) if len(rp) else None,
            "p95": float(np.quantile(rp, 0.95)) if len(rp) else None,
            "max": float(rp.max()) if len(rp) else None,
        },
        "hist_window": list(hw),
        "hist_bins": bins.tolist(),
    }
    return out, preds, split


def ranking_table(res):
    rows = sorted(res["rows"], key=lambda r: -r["skill"])
    lines = ["95% intervals: test-group bootstrap (fitted maps held fixed); control rows: permutation distribution.",
             "",
             "| rank | candidate | kind | skill (test) | 95% interval | skill (train) | test: ray / traj states |",
             "|---:|---|---|---:|---|---:|---|"]
    for i, r in enumerate(rows, 1):
        interval = f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
        if r["kind"] == "control":
            interval += f" (permutations, K={r.get('n_permutations')})"
        tr_s = f"{r['skill_train']:.4f}" if r.get("skill_train") is not None else "–"
        sub = (f"{r['skill_test_ray_states']:.3f} / {r['skill_test_traj_states']:.3f}"
               if r.get("skill_test_ray_states") is not None else "–")
        name = r["name"].replace("|", "\\|")
        lines.append(f"| {i} | {name} | {r['kind']} | {r['skill']:.4f} | {interval} | {tr_s} | {sub} |")
    rp = res["random_projections"]
    if rp["n"]:
        lines.append(f"| – | random linear projections (n={rp['n']}) | control | median {rp['median']:.4f} | "
                     f"5–95%: [{rp['p05']:.4f}, {rp['p95']:.4f}], max {rp['max']:.4f} | – | – |")
    return "\n".join(lines)


def feature_skills_under_labels(system, X, split, n, M, feature_names, iso_pseudo):
    """Test skill of single-feature candidates (isotonic on train) for an arbitrary label set.

    Same estimator as rank(); used to build null distributions (e.g. labels drawn from the exact q)."""
    n = np.asarray(n, float)
    M = np.asarray(M, float)
    tr, te = split == 0, split == 2
    p0 = n[tr].sum() / M[tr].sum()
    llc = binom_ll(np.full(len(n), p0), n, M)
    lls = sat_ll(n, M)
    den = (lls[te] - llc[te]).sum()
    feats = system.features()
    out = {}
    for nm in feature_names:
        c = feats[nm](X)
        p = IsotonicBinomial(iso_pseudo).fit(c[tr], n[tr], M[tr]).predict(c)
        out[nm] = float((binom_ll(p, n, M)[te] - llc[te]).sum() / den)
    return out
