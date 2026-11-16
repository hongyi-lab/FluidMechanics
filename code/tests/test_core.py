import math

import numpy as np
import pytest
import torch
from scipy.integrate import cumulative_trapezoid
from sklearn.isotonic import IsotonicRegression

from committor import labeling
from committor.labeling import label_states
from committor.ranking import (GroupBootstrap, IsotonicBinomial, binom_ll, group_split, pava_blocks,
                               sat_ll)
from committor.sde import simulate_absorbing
from committor.systems.base import System
from committor.systems.langevin2d import Langevin2D
from committor.utils import torch_generator

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class BM1D(System):
    """Driftless Brownian motion between A = {x <= 0} and B = {x >= 1}: q(x) = x."""
    name, dim, coord_names = "bm1d", 1, ("x",)

    def __init__(self, s=1.0):
        self.s = s

    def drift(self, X):
        return torch.zeros_like(X)

    def noise_std(self, device, dtype):
        return torch.full((1,), self.s, device=device, dtype=dtype)

    def in_A(self, X):
        return X[:, 0] <= 0.0

    def in_B(self, X):
        return X[:, 0] >= 1.0

    def boundary_levels(self, X):
        v = torch.full_like(X[:, 0], self.s ** 2)
        return X[:, 0], 1.0 - X[:, 0], v, v

    def reference_state(self):
        return np.zeros(1)

    def features(self):
        return {"x": lambda X: X[:, 0]}

    def params(self):
        return {"name": self.name, "s": self.s}


def test_analytic_q_matches_dense_quadrature():
    sysm = Langevin2D(eps=0.25)
    xs = np.linspace(-1, 1, 201)
    q = sysm.analytic_q(xs)
    grid = np.linspace(-1, 1, 2_000_001)
    g = np.exp((sysm.V(grid) - 1.0) / 0.25)
    cum = cumulative_trapezoid(g, grid, initial=0.0)
    qg = np.interp(xs, grid, cum / cum[-1])
    assert np.max(np.abs(q - qg)) < 1e-9
    assert q[0] == 0.0 and abs(q[-1] - 1.0) < 1e-12
    assert np.all(np.diff(q) > 0)
    assert np.max(np.abs(q + q[::-1] - 1.0)) < 1e-10      # symmetric potential
    assert sysm.analytic_q(np.array([-3.0, 3.0])).tolist() == [0.0, 1.0]


def test_drift_is_minus_gradient():
    sysm = Langevin2D(eps=0.25, k_y=1.7)
    X = torch.randn(1000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    Xr = X.clone().requires_grad_(True)
    W = (Xr[:, 0] ** 2 - 1) ** 2 + 0.5 * 1.7 * Xr[:, 1] ** 2
    (gW,) = torch.autograd.grad(W.sum(), Xr)
    assert torch.allclose(sysm.drift(X), -gW, atol=1e-12)


def test_brownian_committor_is_linear():
    sysm = BM1D()
    M = 20000
    x0 = 0.3
    X0 = torch.full((M, 1), x0, dtype=torch.float64, device=DEV)
    res = simulate_absorbing(sysm, X0, 1e-4, 200000, "em", torch_generator(1, DEV), progress=False)
    assert np.all(res["outcome"] >= 0)
    q = (res["outcome"] == 1).mean()
    se = math.sqrt(x0 * (1 - x0) / M)
    assert abs(q - x0) < 5 * se + 0.005


def test_simulation_is_deterministic_and_initial_absorption():
    sysm = Langevin2D(eps=0.25)
    X0 = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.1, -0.2]], dtype=torch.float64,
                      device=DEV).repeat_interleave(50, 0)
    r1 = simulate_absorbing(sysm, X0, 1e-3, 50000, "heun", torch_generator(7, DEV), progress=False)
    r2 = simulate_absorbing(sysm, X0, 1e-3, 50000, "heun", torch_generator(7, DEV), progress=False)
    assert np.array_equal(r1["outcome"], r2["outcome"]) and np.array_equal(r1["steps"], r2["steps"])
    assert np.all(r1["outcome"][:50] == 0) and np.all(r1["steps"][:50] == 0)
    assert np.all(r1["outcome"][50:100] == 1) and np.all(r1["steps"][50:100] == 0)


def test_pava_matches_sklearn():
    rng = np.random.default_rng(0)
    c = np.round(rng.normal(size=500), 2)          # includes ties
    M = rng.integers(1, 300, size=500).astype(float)
    n = rng.binomial(M.astype(int), 1 / (1 + np.exp(-2 * c))).astype(float)
    bn, bM, centres, pb = pava_blocks(c, n, M)
    fitted = (bn / bM)[pb]
    ref = IsotonicRegression().fit(c, n / M, sample_weight=M).predict(c)
    assert np.max(np.abs(fitted - ref)) < 1e-10
    assert np.all(np.diff(centres) > 0)


def test_isotonic_direction_and_monotone():
    rng = np.random.default_rng(1)
    c = rng.normal(size=400)
    M = np.full(400, 200.0)
    n = rng.binomial(200, 1 / (1 + np.exp(3 * c))).astype(float)     # decreasing in c
    iso = IsotonicBinomial(0.5).fit(c, n, M)
    assert iso.sign == -1.0
    grid = np.linspace(-3, 3, 101)
    p = iso.predict(grid)
    assert np.all(np.diff(p) <= 1e-15) and p.min() > 0 and p.max() < 1


def test_skill_endpoints():
    rng = np.random.default_rng(2)
    M = np.full(300, 200.0)
    n = rng.binomial(200, rng.random(300)).astype(float)
    p0 = n.sum() / M.sum()
    llc = binom_ll(np.full(300, p0), n, M)
    lls = sat_ll(n, M)
    assert abs(((binom_ll(n / M, n, M) - llc).sum()) / ((lls - llc).sum()) - 1.0) < 1e-6
    assert abs((llc - llc).sum()) == 0.0
    assert np.all(lls >= binom_ll(np.clip(n / M + 0.01, 0, 1), n, M) - 1e-9)


def test_group_split_keeps_groups_together():
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 500, size=6000)
    split = group_split(groups, (0.7, 0.15, 0.15), 11)
    for g in np.unique(groups):
        assert len(np.unique(split[groups == g])) == 1
    frac = np.bincount(split, minlength=3) / len(split)
    assert np.allclose(frac, (0.7, 0.15, 0.15), atol=0.03)


def test_group_bootstrap_ci_contains_point():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(100), 10)
    num = rng.random(1000)
    den = num + rng.random(1000)
    b = GroupBootstrap(groups, 500, 5)
    lo, hi = b.ci(num, den)
    assert lo < num.sum() / den.sum() < hi


def test_label_resume_skips_done_chunks(tmp_path, monkeypatch):
    sysm = Langevin2D(eps=0.25)
    sim = {"dt": 1e-3, "max_steps": 50000, "scheme": "heun", "compact_every": 100, "bridge": True}
    states = np.array([[0.0, 0.0], [0.2, 1.0], [-0.3, -0.5], [0.5, 0.1], [-0.6, 0.0]])
    r1 = label_states(sysm, states, 64, sim, 9, tmp_path, 2, DEV, progress=False)

    def boom(*a, **k):
        raise AssertionError("should not re-simulate finished chunks")

    monkeypatch.setattr(labeling, "simulate_absorbing", boom)
    r2 = label_states(sysm, states, 64, sim, 9, tmp_path, 2, DEV, progress=False)
    assert np.array_equal(r1["n_B"], r2["n_B"])
    with pytest.raises(RuntimeError):
        label_states(sysm, states, 64, sim, 10, tmp_path, 2, DEV, progress=False)


def test_bridge_removes_discrete_monitoring_bias():
    """Driftless BM with a coarse step: grid-only checks overshoot (q biased up by ~0.58 s sqrt(dt)/(1+...)),
    the Brownian-bridge test is exact for flat boundaries and constant noise."""
    sysm = BM1D()
    M, x0, dt = 40000, 0.3, 1e-2
    X0 = torch.full((M, 1), x0, dtype=torch.float64, device=DEV)
    se = math.sqrt(x0 * (1 - x0) / M)
    q_plain = (simulate_absorbing(sysm, X0, dt, 100000, "em", torch_generator(3, DEV), progress=False,
                                  bridge=False)["outcome"] == 1).mean()
    q_bridge = (simulate_absorbing(sysm, X0, dt, 100000, "em", torch_generator(4, DEV), progress=False,
                                   bridge=True)["outcome"] == 1).mean()
    shift = 0.5826 * math.sqrt(dt)
    expected_plain = (x0 + shift) / (1 + 2 * shift)
    assert abs(q_plain - expected_plain) < 5 * se
    assert abs(q_plain - x0) > 4 * se
    assert abs(q_bridge - x0) < 4 * se


def test_bridge_requires_boundary_levels():
    class NoLevels(BM1D):
        boundary_levels = System.boundary_levels
    X0 = torch.zeros((4, 1), dtype=torch.float64, device=DEV) + 0.5
    with pytest.raises(ValueError):
        simulate_absorbing(NoLevels(), X0, 1e-3, 10, "em", torch_generator(0, DEV), progress=False, bridge=True)


def test_json_is_strict(tmp_path):
    import json
    from committor.utils import atomic_write_json
    atomic_write_json(tmp_path / "a.json", {"a": float("nan"), "b": np.float64(np.inf), "c": [np.int64(3)]})
    assert json.loads((tmp_path / "a.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))) \
        == {"a": None, "b": None, "c": [3]}


@pytest.mark.parametrize("bridge", [False, True])
def test_compiled_step_matches_eager(bridge):
    """compile_step=True runs the same per-step math with the same random stream: identical outcomes."""
    if DEV != "cuda":
        pytest.skip("compiled path is used on GPU")
    sysm = Langevin2D(eps=0.25)
    X0 = torch.tensor([[0.0, 0.0], [0.3, 0.5], [-0.4, -0.2]], dtype=torch.float64, device=DEV).repeat_interleave(300, 0)
    r1 = simulate_absorbing(sysm, X0, 1e-3, 20000, "heun", torch_generator(5, DEV), progress=False, bridge=bridge)
    r2 = simulate_absorbing(sysm, X0, 1e-3, 20000, "heun", torch_generator(5, DEV), progress=False, bridge=bridge,
                            compile_step=True)
    assert np.array_equal(r1["outcome"], r2["outcome"]) and np.array_equal(r1["steps"], r2["steps"])


def test_return_final_states_are_absorption_states():
    sysm = Langevin2D(eps=0.25)
    X0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=DEV).repeat(500, 1)
    r = simulate_absorbing(sysm, X0, 1e-3, 50000, "heun", torch_generator(2, DEV), progress=False, bridge=False,
                           return_final=True, compact_every=7)
    fx = r["final_X"]
    assert np.all(fx[r["outcome"] == 1, 0] >= 1.0) and np.all(fx[r["outcome"] == 0, 0] <= -1.0)
    r2 = simulate_absorbing(sysm, X0, 1e-3, 50000, "heun", torch_generator(2, DEV), progress=False, bridge=False,
                            compact_every=7)
    assert np.array_equal(r["outcome"], r2["outcome"]) and np.array_equal(r["steps"], r2["steps"])


def test_threshold_quantiles_and_paired_diff_are_inf_safe():
    """Gate 3 review: np.quantile returns NaN next to inf; paired_diff must give p(diff <= 0) = 1 when a is finite and
    b has many no-crossings, and 0 in the swapped order."""
    import numpy as np
    from committor.threshold import paired_diff, qinf
    inf = np.inf
    assert qinf([1] * 8 + [inf] * 2, 0.9) == inf
    assert qinf([1, 2, 3, 4], 0.5) == 2.5
    assert qinf(list(range(11)), 0.9) == 9.0
    rng = np.random.default_rng(0)
    a = 0.1 * rng.normal(size=400)
    b = a.copy()
    b[rng.random(400) < 0.5] = inf
    boot = rng.integers(0, 400, (200, 400))
    assert paired_diff(a, b, boot)["q90_abs_diff_p_le0"] == 1.0
    assert paired_diff(b, a, boot)["q90_abs_diff_p_le0"] == 0.0
    assert paired_diff(b, b, boot)["q90_abs_diff_p_le0"] == 1.0          # inf - inf := 0


def test_quadmech_physical_warm_start_and_resume(tmp_path):
    """Gate 3b mechanism models: physical Q/b reproduce u; nested warm start equals the (k-1)-model; a checkpointed
    run resumed after an interruption ends in the same state as an uninterrupted run (CPU, deterministic)."""
    from committor import mechanism as mech
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 9)) * np.geomspace(1e-3, 1e-1, 9)[None, :]
    y = (X[:, 2] ** 2 + (X[:, 3:8] ** 2).sum(1) > np.median(X[:, 2] ** 2 + (X[:, 3:8] ** 2).sum(1))).astype(float)
    xs = np.sqrt((X ** 2).mean(0))
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    m1 = mech.QuadMech(1, [8, 8], xs)
    gen = torch.Generator().manual_seed(1)
    m1.reset_g(gen)
    m1.init_random(gen, 1.0, Xt, 1e-3)
    Q, b = m1.physical()
    u_phys = np.einsum("nj,jl,nl->n", X, Q[0], X) + X @ b[0]
    assert np.allclose(u_phys, m1.u(Xt)[:, 0].detach().numpy(), rtol=1e-10, atol=1e-14)
    m2 = mech.QuadMech(2, [8, 8], xs)
    m2.init_random(gen, 1.0, Xt, 1e-3, slots=[1])
    m2.warm_from(m1)
    assert torch.allclose(m2(Xt), m1(Xt), rtol=0, atol=1e-13)
    # set_quadratic round trip
    m3 = mech.QuadMech(1, [8, 8], xs)
    f = m3.set_quadratic(0, Q[0], b[0])
    assert np.allclose(m3.u(Xt)[:, 0].detach().numpy() * f, u_phys, rtol=1e-10, atol=1e-14)
    assert abs(float(torch.sqrt((m3.A[0].detach() ** 2).sum() + (m3.b[0].detach() ** 2).sum())) - 1.0) < 1e-12
    # phys2 forms
    p2 = mech.QuadMech(2, [8, 8], xs, kind="phys2")
    up = p2.u(Xt).detach().numpy()
    assert np.allclose(up[:, 0], X[:, 2] ** 2) and np.allclose(up[:, 1], (X[:, 3:8] ** 2).sum(1))
    p2.reset_g(gen)
    p2.init_random(gen, 0.0, Xt, 1e-3)
    f2 = mech.QuadMech(2, [8, 8], xs)
    f2.copy_from_phys2(p2)
    assert torch.allclose(f2(Xt), p2(Xt), rtol=0, atol=1e-12)
    # a warm start is never worse than its start on the val objective
    ip = mech.train(f2, Xt[:300], yt[:300], Xt[300:], yt[300:], {"lr": 0.5, "max_epochs": 5, "patience": 100, "checkpoint_every": 100},
                    tmp_path / "w.ck", tmp_path / "w.pt", progress=False)
    with torch.no_grad():
        v_start = float(torch.nn.functional.binary_cross_entropy_with_logits(p2(Xt[300:]), yt[300:]))
    assert ip["val_nll"] <= v_start + 1e-15
    # resume
    cfg = {"lr": 3e-3, "max_epochs": 60, "patience": 1000, "checkpoint_every": 20, "lr_warmup_epochs": 30}

    def fresh():
        m = mech.QuadMech(2, [8, 8], xs)
        g = torch.Generator().manual_seed(5)
        m.reset_g(g)
        m.init_random(g, 1.0, Xt, 1e-3)
        return m
    ma = fresh()
    ia = mech.train(ma, Xt[:300], yt[:300], Xt[300:], yt[300:], cfg, tmp_path / "a.ck", tmp_path / "a.pt", progress=False)
    ia["state"] = {k: v.clone() for k, v in ma.state_dict().items()}

    class Boom(Exception):
        pass

    mb = fresh()
    calls = {"n": 0}
    orig = mb.forward

    def bomb(x):
        calls["n"] += 1
        if calls["n"] > 2 * 47:                                # interrupted during epoch 47 (checkpoint at 40 exists)
            raise Boom()
        return orig(x)
    mb.forward = bomb
    with pytest.raises(Boom):
        mech.train(mb, Xt[:300], yt[:300], Xt[300:], yt[300:], cfg, tmp_path / "b.ck", tmp_path / "b.pt", progress=False)
    assert (tmp_path / "b.ck").exists() and not (tmp_path / "b.pt").exists()
    mc = fresh()                                              # restart: state, optimiser and counters from the checkpoint
    ib = mech.train(mc, Xt[:300], yt[:300], Xt[300:], yt[300:], cfg, tmp_path / "b.ck", tmp_path / "b.pt", progress=False)
    assert ib["epochs_run"] == ia["epochs_run"] == 60 and ib["best_epoch"] == ia["best_epoch"]
    for k, v in mc.state_dict().items():
        assert torch.equal(v, ia["state"][k]), k
    assert not (tmp_path / "b.ck").exists()
    # a finished model is loaded, not retrained
    md = fresh()
    ic = mech.train(md, Xt[:300], yt[:300], Xt[300:], yt[300:], cfg, tmp_path / "b.ck", tmp_path / "b.pt", progress=False)
    assert ic["val_nll"] == ib["val_nll"]
