"""Moehlis-Faisst-Eckhardt nine-mode model of sinusoidal shear flow.

Source: J. Moehlis, H. Faisst, B. Eckhardt, "A low-dimensional model for turbulent shear flows",
New J. Phys. 6, 56 (2004); file literature/papers/Moehlis_Faisst_Eckhardt_2004_NJP_6_56.pdf.
`rhs_paper` is a term-by-term transcription of the amplitude equations (21)-(28) with the
constants (29)-(31). `mode_fields` transcribes the modes (7)-(16) and the body force (5); it is
used only to cross-check the transcription by Galerkin projection (eq. (20)).
Symmetries: Moehlis, Faisst, Eckhardt, SIAM J. Appl. Dyn. Syst. 4, 352 (2005), eqs. (3.24)-(3.25).

The fast drift used for integration is the same polynomial in tensor form,
    f(a) = c + L a + Q(a, a),     Q symmetric in its last two indices,
extracted from `rhs_paper` by exact polarisation (f is a quadratic polynomial).
Units: lengths d/2, velocity U0, time (d/2)/U0 (paper section 2). Laminar state a = (1, 0, ..., 0).
"""

import math

import numpy as np
import torch

from .base import System

NMODES = 9
LAMINAR = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])

# Sign actions of the discrete symmetries on (a1..a9), SIADS 2005 eqs. (3.24)-(3.25).
# R (reflection z -> -z) has the same action as T_{Lx/2} (stated there).
SYMMETRIES = {
    "T_Lx/2": np.array([1, 1, 1, -1, -1, -1, -1, -1, 1], dtype=float),
    "T_Lz/2": np.array([1, -1, -1, 1, 1, -1, -1, -1, 1], dtype=float),
}
SYMMETRIES["T_Lx/2*T_Lz/2"] = SYMMETRIES["T_Lx/2"] * SYMMETRIES["T_Lz/2"]
SYMMETRIES["R"] = SYMMETRIES["T_Lx/2"].copy()


def wavenumbers(Lx, Lz):
    """alpha = 2 pi / Lx, beta = pi / 2, gamma = 2 pi / Lz (paper section 2)."""
    return 2.0 * math.pi / Lx, math.pi / 2.0, 2.0 * math.pi / Lz


def _stack(cols, like):
    if isinstance(like, torch.Tensor):
        return torch.stack(cols, dim=-1)
    return np.stack(cols, axis=-1)


def rhs_paper(a, Re, Lx, Lz):
    """da/dt from equations (21)-(28) of the paper, written term by term.

    a: (..., 9) numpy array or torch tensor. Returns the same type and shape.
    """
    al, be, ga = wavenumbers(Lx, Lz)
    k_ag = math.sqrt(al ** 2 + ga ** 2)                      # (29)
    k_bg = math.sqrt(be ** 2 + ga ** 2)                      # (30)
    k_abg = math.sqrt(al ** 2 + be ** 2 + ga ** 2)           # (31)
    s32 = math.sqrt(3.0 / 2.0)
    s6 = math.sqrt(6.0)
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = (a[..., i] for i in range(9))

    # (21)
    d1 = (be ** 2 / Re - be ** 2 / Re * a1
          - s32 * be * ga / k_abg * a6 * a8
          + s32 * be * ga / k_bg * a2 * a3)
    # (22)
    d2 = (-(4.0 * be ** 2 / 3.0 + ga ** 2) * a2 / Re
          + 5.0 * math.sqrt(2.0) / (3.0 * math.sqrt(3.0)) * ga ** 2 / k_ag * a4 * a6
          - ga ** 2 / (s6 * k_ag) * a5 * a7
          - al * be * ga / (s6 * k_ag * k_abg) * a5 * a8
          - s32 * be * ga / k_bg * a1 * a3
          - s32 * be * ga / k_bg * a3 * a9)
    # (23)
    d3 = (-(be ** 2 + ga ** 2) / Re * a3
          + 2.0 / s6 * al * be * ga / (k_ag * k_bg) * (a4 * a7 + a5 * a6)
          + (be ** 2 * (3.0 * al ** 2 + ga ** 2) - 3.0 * ga ** 2 * (al ** 2 + ga ** 2))
          / (s6 * k_ag * k_bg * k_abg) * a4 * a8)
    # (24)
    d4 = (-(3.0 * al ** 2 + 4.0 * be ** 2) / (3.0 * Re) * a4
          - al / s6 * a1 * a5
          - 10.0 / (3.0 * s6) * al ** 2 / k_ag * a2 * a6
          - s32 * al * be * ga / (k_ag * k_bg) * a3 * a7
          - s32 * al ** 2 * be ** 2 / (k_ag * k_bg * k_abg) * a3 * a8
          - al / s6 * a5 * a9)
    # (25)
    d5 = (-(al ** 2 + be ** 2) / Re * a5
          + al / s6 * a1 * a4
          + al ** 2 / (s6 * k_ag) * a2 * a7
          - al * be * ga / (s6 * k_ag * k_abg) * a2 * a8
          + al / s6 * a4 * a9
          + 2.0 / s6 * al * be * ga / (k_ag * k_bg) * a3 * a6)
    # (26)
    d6 = (-(3.0 * al ** 2 + 4.0 * be ** 2 + 3.0 * ga ** 2) / (3.0 * Re) * a6
          + al / s6 * a1 * a7
          + s32 * be * ga / k_abg * a1 * a8
          + 10.0 / (3.0 * s6) * (al ** 2 - ga ** 2) / k_ag * a2 * a4
          - 2.0 * math.sqrt(2.0 / 3.0) * al * be * ga / (k_ag * k_bg) * a3 * a5
          + al / s6 * a7 * a9
          + s32 * be * ga / k_abg * a8 * a9)
    # (27), first line
    d7 = (-(al ** 2 + be ** 2 + ga ** 2) / Re * a7
          - al / s6 * (a1 * a6 + a6 * a9)
          + 1.0 / s6 * (ga ** 2 - al ** 2) / k_ag * a2 * a5
          + 1.0 / s6 * al * be * ga / (k_ag * k_bg) * a3 * a4)
    # (27), second line
    d8 = (-(al ** 2 + be ** 2 + ga ** 2) / Re * a8
          + 2.0 / s6 * al * be * ga / (k_ag * k_abg) * a2 * a5
          + ga ** 2 * (3.0 * al ** 2 - be ** 2 + 3.0 * ga ** 2) / (s6 * k_ag * k_bg * k_abg) * a3 * a4)
    # (28)
    d9 = (-9.0 * be ** 2 / Re * a9
          + s32 * be * ga / k_bg * a2 * a3
          - s32 * be * ga / k_abg * a6 * a8)
    return _stack([d1, d2, d3, d4, d5, d6, d7, d8, d9], a)


def polynomial_coefficients(Re, Lx, Lz):
    """(c, L, Q) with rhs_paper(a) = c + L a + Q(a, a), Q[i, j, k] = Q[i, k, j], by polarisation."""
    f = lambda v: rhs_paper(np.asarray(v, dtype=np.float64), Re, Lx, Lz)
    E = np.eye(NMODES)
    c = f(np.zeros(NMODES))
    fp = np.array([f(E[j]) for j in range(NMODES)])          # c + L e_j + Q_jj
    fm = np.array([f(-E[j]) for j in range(NMODES)])         # c - L e_j + Q_jj
    L = ((fp - fm) / 2.0).T
    Q = np.zeros((NMODES, NMODES, NMODES))
    for j in range(NMODES):
        Q[:, j, j] = (fp[j] + fm[j]) / 2.0 - c
    for j in range(NMODES):
        for k in range(j + 1, NMODES):
            s = f(E[j] + E[k]) - fp[j] - fp[k] + c            # Q_jk + Q_kj
            Q[:, j, k] = Q[:, k, j] = s / 2.0
    return c, L, Q


def perturbation_energy(a):
    """E = sum_i (a_i - a_i^lam)^2 (paper convention: figure 6 plots E - 1 = sum_{i>=2} a_i^2 at a1 = 0).

    Mode normalisation (17) makes this proportional to the volume-integrated kinetic energy of
    the deviation from the laminar flow."""
    if isinstance(a, torch.Tensor):
        d = a.clone()
        d[..., 0] = d[..., 0] - 1.0
        return (d * d).sum(-1)
    a = np.asarray(a, dtype=np.float64)
    d = a.copy()
    d[..., 0] -= 1.0
    return (d * d).sum(-1)


class MFE(System):
    """MFE dynamics with additive isotropic noise: da = f(a) dt + sigma dW.

    A = {E < r_A} (near laminar), B = {E >= E_B} (turbulent region), E = sum_i (a_i - a_i^lam)^2.
    r_A = None or 0 means A is empty; E_B = None means B is empty (Gate 1 uses neither).
    """

    name = "mfe"
    dim = NMODES
    coord_names = tuple(f"a{i}" for i in range(1, NMODES + 1))

    def __init__(self, Re=400.0, Lx_over_pi=4.0, Lz_over_pi=2.0, sigma=0.0, r_A=None, E_B=None):
        self.Re = float(Re)
        self.Lx_over_pi = float(Lx_over_pi)
        self.Lz_over_pi = float(Lz_over_pi)
        self.Lx = self.Lx_over_pi * math.pi
        self.Lz = self.Lz_over_pi * math.pi
        self.sigma = float(sigma)
        self.r_A = float(r_A) if r_A else None
        self.E_B = float(E_B) if E_B is not None else None
        self.c, self.L, self.Q = polynomial_coefficients(self.Re, self.Lx, self.Lz)
        self._dev_cache = {}

    # --- numpy -------------------------------------------------------------------------------
    def rhs(self, a):
        """Tensor-form RHS (numpy), a: (..., 9)."""
        a = np.asarray(a, dtype=np.float64)
        return self.c + a @ self.L.T + np.einsum("ijk,...j,...k->...i", self.Q, a, a)

    def nonlinear(self, a):
        a = np.asarray(a, dtype=np.float64)
        return np.einsum("ijk,...j,...k->...i", self.Q, a, a)

    def jacobian(self, a):
        """df/da at a (9,): L + 2 Q(a, .)."""
        return self.L + 2.0 * np.einsum("ijk,k->ij", self.Q, np.asarray(a, dtype=np.float64))

    # --- torch -------------------------------------------------------------------------------
    def _tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._dev_cache:
            self._dev_cache[key] = tuple(torch.as_tensor(x, device=device, dtype=dtype)
                                         for x in (self.c, self.L.T.copy(), self.Q))
        return self._dev_cache[key]

    def drift(self, X):
        if torch.compiler.is_compiling():
            return rhs_paper(X, self.Re, self.Lx, self.Lz)
        if X.is_cuda and X.shape[0] >= 2:                 # inductor mis-launches batch-size-1 kernels
            if getattr(self, "_drift_c", None) is None:
                Re, Lx, Lz = self.Re, self.Lx, self.Lz
                self._drift_c = torch.compile(lambda Y: rhs_paper(Y, Re, Lx, Lz), dynamic=True, fullgraph=True)
            return self._drift_c(X)
        return rhs_paper(X, self.Re, self.Lx, self.Lz)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_drift_c"] = None
        st["_dev_cache"] = {}
        return st

    def noise_std(self, device, dtype):
        return torch.full((NMODES,), self.sigma, device=device, dtype=dtype)

    @staticmethod
    def energy(X):
        return (X[:, 0] - 1.0) ** 2 + (X[:, 1:] ** 2).sum(1)

    def in_A(self, X):
        if self.r_A is None:
            return torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        return self.energy(X) < self.r_A

    def in_B(self, X):
        if self.E_B is None:
            return torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        return self.energy(X) >= self.E_B

    def boundary_levels(self, X):
        """l_A = E - r_A, l_B = E_B - E; both have gradient norm^2 = 4E, so var = 4 E sigma^2."""
        E = self.energy(X)
        inf = torch.full_like(E, math.inf)
        lA = E - self.r_A if self.r_A is not None else inf
        lB = self.E_B - E if self.E_B is not None else inf
        v = (4.0 * self.sigma ** 2) * E.clamp(min=1e-300)
        return lA, lB, v, v

    def reference_state(self):
        return LAMINAR.copy()

    def features(self):
        feats = {"E": perturbation_energy}
        feats.update({f"a{i + 1}": (lambda X, i=i: X[:, i]) for i in range(NMODES)})
        feats.update({f"abs_a{i + 1}": (lambda X, i=i: np.abs(X[:, i])) for i in range(NMODES)})
        feats["one_minus_a1"] = lambda X: 1.0 - X[:, 0]
        return feats

    def params(self):
        return {"name": self.name, "Re": self.Re, "Lx_over_pi": self.Lx_over_pi,
                "Lz_over_pi": self.Lz_over_pi, "sigma": self.sigma, "r_A": self.r_A, "E_B": self.E_B}


@torch.no_grad()
def rk4_step(f, X, dt):
    k1 = f(X)
    k2 = f(X + 0.5 * dt * k1)
    k3 = f(X + 0.5 * dt * k2)
    k4 = f(X + dt * k3)
    return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ------------------------------------------------------------------------------------------------
# Transcription cross-check: modes (7)-(16), force (5) and Galerkin projection (20).
# ------------------------------------------------------------------------------------------------

def mode_fields(Lx, Lz, Re):
    """Sympy expressions for the nine modes (7)-(16), the force (5) and the coordinates."""
    import sympy as sp
    x, y, z = sp.symbols("x y z", real=True)
    al = sp.Float(2 * math.pi / Lx, 30)
    be = sp.pi / 2
    ga = sp.Float(2 * math.pi / Lz, 30)
    s2, s3 = sp.sqrt(2), sp.sqrt(3)
    cy = sp.cos(sp.pi * y / 2)
    sy = sp.sin(sp.pi * y / 2)
    N8 = 2 * s2 / sp.sqrt((al ** 2 + ga ** 2) * (4 * al ** 2 + 4 * ga ** 2 + sp.pi ** 2))   # (15)
    u = [
        (s2 * sy, 0, 0),                                                                     # (7)
        (4 / s3 * cy ** 2 * sp.cos(ga * z), 0, 0),                                           # (8)
        tuple(2 / sp.sqrt(4 * ga ** 2 + sp.pi ** 2) * e for e in
              (0, 2 * ga * cy * sp.cos(ga * z), sp.pi * sy * sp.sin(ga * z))),               # (9)
        (0, 0, 4 / s3 * sp.cos(al * x) * cy ** 2),                                           # (10)
        (0, 0, 2 * sp.sin(al * x) * sy),                                                     # (11)
        tuple(4 * s2 / sp.sqrt(3 * (al ** 2 + ga ** 2)) * e for e in
              (-ga * sp.cos(al * x) * cy ** 2 * sp.sin(ga * z), 0,
               al * sp.sin(al * x) * cy ** 2 * sp.cos(ga * z))),                             # (12)
        tuple(2 * s2 / sp.sqrt(al ** 2 + ga ** 2) * e for e in
              (ga * sp.sin(al * x) * sy * sp.sin(ga * z), 0,
               al * sp.cos(al * x) * sy * sp.cos(ga * z))),                                  # (13)
        tuple(N8 * e for e in
              (sp.pi * al * sp.sin(al * x) * sy * sp.sin(ga * z),
               2 * (al ** 2 + ga ** 2) * sp.cos(al * x) * cy * sp.sin(ga * z),
               -sp.pi * ga * sp.cos(al * x) * sy * sp.cos(ga * z))),                         # (14)
        (s2 * sp.sin(3 * sp.pi * y / 2), 0, 0),                                              # (16)
    ]
    u = [tuple(sp.sympify(e) for e in m) for m in u]
    F = (s2 * sp.pi ** 2 / (4 * sp.Float(Re, 30)) * sy, sp.Integer(0), sp.Integer(0))       # (5)
    return (x, y, z), u, F


def galerkin_projection(Lx, Lz, Re, nx=16, nz=16, ny=64):
    """Coefficients of eq. (20) by quadrature: returns dict with c, L, Q (Q symmetrised in its
    last two indices), the Gram matrix of the modes, max |div u_m| and boundary residuals.

    x, z: uniform periodic grids (exact for the trigonometric polynomials involved);
    y: Gauss-Legendre on [-1, 1] (spectrally accurate for these entire functions)."""
    import sympy as sp
    (x, y, z), u, F = mode_fields(Lx, Lz, Re)
    X = (x, y, z)
    xs = np.arange(nx) * Lx / nx
    zs = np.arange(nz) * Lz / nz
    yg, wy = np.polynomial.legendre.leggauss(ny)
    XX, YY, ZZ = np.meshgrid(xs, yg, zs, indexing="ij")
    W = (Lx / nx) * (Lz / nz) * wy[None, :, None] * np.ones_like(XX)

    def ev(expr):
        fn = sp.lambdify((x, y, z), expr, "numpy")
        return np.broadcast_to(np.asarray(fn(XX, YY, ZZ), dtype=np.float64), XX.shape)

    U = np.array([[ev(c) for c in m] for m in u])                                   # (9, 3, grid)
    dU = np.array([[[ev(sp.diff(c, X[d])) for d in range(3)] for c in m] for m in u])  # (9, 3c, 3d, grid)
    lapU = np.array([[ev(sum(sp.diff(c, X[d], 2) for d in range(3))) for c in m] for m in u])
    Fv = np.array([ev(c) for c in F])
    inner = lambda A, B: float(np.sum(A * B * W))
    gram = np.array([[sum(inner(U[m, c], U[n, c]) for c in range(3)) for n in range(9)] for m in range(9)])
    norm = 2.0 * Lx * Lz                     # (17): 2 (2 pi / alpha)(2 pi / gamma)
    # advection (u_m . grad) u_l, component c: sum_d U[m,d] dU[l,c,d]
    adv = np.einsum("mdg,lcdg->mlcg", U.reshape(9, 3, -1), dU.reshape(9, 3, 3, -1))
    Uf = U.reshape(9, 3, -1)
    Wf = W.reshape(-1)
    Qraw = -np.einsum("mlcg,ncg,g->nml", adv, Uf, Wf) / norm
    Q = 0.5 * (Qraw + Qraw.transpose(0, 2, 1))
    L = np.einsum("mcg,ncg,g->nm", lapU.reshape(9, 3, -1), Uf, Wf) / (Re * norm)
    c = np.einsum("cg,ncg,g->n", Fv.reshape(3, -1), Uf, Wf) / norm
    div = np.array([np.max(np.abs(sum(dU[m, d, d] for d in range(3)))) for m in range(9)])
    # free-slip walls: u_y = 0 and du_x/dy = du_z/dy = 0 at y = +-1
    walls = []
    for yw in (-1.0, 1.0):
        XW, ZW = np.meshgrid(xs, zs, indexing="ij")
        for m in range(9):
            vals = [sp.lambdify((x, z), u[m][1].subs(y, yw), "numpy")(XW, ZW),
                    sp.lambdify((x, z), sp.diff(u[m][0], y).subs(y, yw), "numpy")(XW, ZW),
                    sp.lambdify((x, z), sp.diff(u[m][2], y).subs(y, yw), "numpy")(XW, ZW)]
            walls.append(max(float(np.max(np.abs(np.broadcast_to(v, XW.shape)))) for v in vals))
    return {"c": c, "L": L, "Q": Q, "gram": gram, "norm": norm, "max_div": div,
            "max_wall_residual": float(max(walls))}


def mode_symmetry_signs(Lx, Lz, n=7, seed=0):
    """Sign s_m with g u_m = s_m u_m for the velocity-field actions of T_{Lx/2}, T_{Lz/2} and R
    (SIADS 2005 eqs. (2.7)-(2.8)), evaluated at random points; NaN if u_m is not mapped to +-u_m."""
    import sympy as sp
    (x, y, z), u, _ = mode_fields(Lx, Lz, 400.0)
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(0, Lx, n), rng.uniform(-1, 1, n), rng.uniform(0, Lz, n)]
    actions = {
        "T_Lx/2": lambda P: (P + [Lx / 2, 0, 0], np.array([1.0, 1.0, 1.0])),
        "T_Lz/2": lambda P: (P + [0, 0, Lz / 2], np.array([1.0, 1.0, 1.0])),
        "R": lambda P: (P * [1, 1, -1], np.array([1.0, 1.0, -1.0])),
    }
    out = {}
    for name, act in actions.items():
        signs = []
        for m in range(9):
            fn = sp.lambdify((x, y, z), list(u[m]), "numpy")
            Pm, flip = act(P)
            v0 = np.array([np.broadcast_to(np.asarray(c, dtype=float), (n,)) for c in fn(*P.T)])
            v1 = flip[:, None] * np.array([np.broadcast_to(np.asarray(c, dtype=float), (n,)) for c in fn(*Pm.T)])
            if np.allclose(v1, v0, atol=1e-12):
                signs.append(1.0)
            elif np.allclose(v1, -v0, atol=1e-12):
                signs.append(-1.0)
            else:
                signs.append(float("nan"))
        out[name] = np.array(signs)
    return out
