"""Gate 4a solver-correctness items 1, 2, 4, 6 (+ divergence / symmetry residual along a short run) at 128x64x64, dt 0.03125.
Not in the preset plan: a cheap check that the new resolution is a correct discretisation (only solver correctness stops)."""
import math, sys, json
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, "/srv/hgao0864/project_main/code/scripts")
from gate4_dns import make_grid, SHARED
from committor.dns import lawson_rk4, step_nl
from committor.systems.mfe import LAMINAR, rhs_paper
from committor.utils import atomic_write_json
cfg = yaml.safe_load(open("/srv/hgao0864/project_main/code/configs/gate4_dns.yaml"))
C = cfg["checks_4a"]
out = {}
for N, dt in (([128, 64, 64], 0.03125), ([96, 48, 48], 0.0625)):
    g = make_grid(cfg, N)
    o = {}
    sym = {k: float((g.sym_apply(g.modes, R, d, D) - g.modes).abs().max()) for k, (R, d, D) in g.candidate_symmetries().items()}
    o["mode_symmetry_max_abs_gu_minus_u"] = {k: v for k, v in sym.items() if k in SHARED}
    zero = torch.zeros_like(g.U)
    Fh = g.fwd(torch.stack([torch.as_tensor(math.sqrt(2) * math.pi ** 2 / (4 * g.Re) * np.sin(np.pi * np.broadcast_to(g.ys[:, None, None], (g.Ny, g.Nz, g.Nx)) / 2), device=g.dev),
                            torch.zeros(g.Ny, g.Nz, g.Nx, device=g.dev, dtype=torch.float64), torch.zeros(g.Ny, g.Nz, g.Nx, device=g.dev, dtype=torch.float64)])[None])
    full_lam = g.proj(g.fwd(g.cross(g.U_phys, g.wU_phys)) * g.mask) - g.nu * g.K2 * g.U + Fh
    o["1_laminar"] = max(float(g.inv(g.nl_full(zero)).abs().max()), float(g.inv(full_lam).abs().max()))
    rng = np.random.default_rng(1)
    d = rng.standard_normal((100, 9)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    A = LAMINAR[None, :] + rng.uniform(0, 0.3, 100)[:, None] * d
    diffs = []
    for i in range(0, 100, 20):
        uh = g.from_coeffs(A[i:i + 20])
        dudt = g.proj(g.fwd(g.cross(g.inv(uh), g.inv(g.curl(uh)))) * g.mask) - g.nu * g.K2 * uh + Fh
        diffs.append(np.max(np.abs(g.coeffs(dudt).cpu().numpy() - rhs_paper(A[i:i + 20], g.Re, g.Lx, g.Lz))))
    o["2_galerkin_max_abs_diff"] = float(max(diffs))
    gen = torch.Generator(device=g.dev).manual_seed(7)
    shape = (8, 3, g.Ny, g.Nz, g.Nxh)
    amp = torch.where(g.K2 > 0, 1.0 / torch.where(g.K2 > 0, g.K2, torch.ones_like(g.K2)), torch.zeros_like(g.K2))
    rr = torch.randn(shape, generator=gen, device=g.dev, dtype=torch.float64) + 1j * torch.randn(shape, generator=gen, device=g.dev, dtype=torch.float64)
    uw = g.proj(g.fwd(g.inv(rr * amp * g.mask)) * g.mask)
    Nw = g.selfcross(uw)
    o["4_work_max_ratio"] = float((g.inner(uw, Nw).abs() / (g.energy(uw).sqrt() * g.energy(Nw).sqrt())).max())
    dd = rng.standard_normal((4, 9)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    u = g.clean(g.from_coeffs(math.sqrt(1e-3) * dd))
    E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
    smax = dmax = 0.0
    for k in range(int(round(30 / dt))):
        (un,) = lawson_rk4(g, (u,), lambda s_: (g.nl_full(s_[0]),), dt, E1, E2)
        un = un * g.mask
        if k % 16 == 0:
            smax = max(smax, float(g.sym_residual(un).max()))
            dmax = max(dmax, float(g.max_div(un).max()))
        u = g.symmetrize(un)
    o["1_sym_residual_max_t<=30"] = smax
    o["1_div_max_t<=30"] = max(dmax, float(g.max_div(uw).max()))
    o["E_t30"] = g.energy(u).cpu().numpy().tolist()
    ua = g.clean(g.from_coeffs(math.sqrt(1e-3) * dd[:1]))
    ub = g.clean(g.from_coeffs(math.sqrt(1e-3) * dd))
    for k in range(int(round(20 / dt))):
        ua, ub = step_nl(g, ua, dt, E1, E2), step_nl(g, ub, dt, E1, E2)
    o["6_alone_vs_batch_max_abs_du"] = float((g.inv(ua) - g.inv(ub[:1])).abs().max())
    o["pass"] = {"1 laminar < 1e-12": o["1_laminar"] < float(C["laminar_rhs_max"]), "1 div < 1e-10": o["1_div_max_t<=30"] < float(C["div_max"]),
                 "1 sym residual < 1e-12": o["1_sym_residual_max_t<=30"] < float(C["sym_residual_max"]),
                 "2 Galerkin < 1e-10": o["2_galerkin_max_abs_diff"] < float(C["galerkin"]["tol"]), "4 work < 1e-10": o["4_work_max_ratio"] < float(C["work"]["tol"]),
                 "6 batch < 1e-12": o["6_alone_vs_batch_max_abs_du"] < float(C["batch"]["tol"])}
    out[f"{N[0]}x{N[1]}x{N[2]} dt {dt}"] = o
    print(N, json.dumps(o, indent=1))
    del g; torch.cuda.empty_cache()
atomic_write_json(Path("/srv/hgao0864/project_main/experiments/20261118_gate7_dns_v1/check_4a_128.json"), out)
