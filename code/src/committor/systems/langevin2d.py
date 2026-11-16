"""Gate 0 positive control: 2D overdamped Langevin dynamics with a separable potential.

    dX = -grad W(X) dt + sqrt(2 eps) dW,   W(x, y) = V(x) + U(y)
    V(x) = (x^2 - 1)^2   (double well, minima at x = -1, +1, barrier height 1 at x = 0)
    U(y) = k_y y^2 / 2   (single well)

A = {x <= a}, B = {x >= b}. Because the x-dynamics is autonomous and A, B depend only on x,
the committor depends only on x and solves eps q'' - V'(x) q' = 0 with q(a) = 0, q(b) = 1:

    q(x) = int_a^x exp(V(s)/eps) ds / int_a^b exp(V(s)/eps) ds
"""

import math

import numpy as np
import torch
from scipy.integrate import quad

from .base import System


class Langevin2D(System):
    name = "langevin2d"
    dim = 2
    coord_names = ("x", "y")

    def __init__(self, eps, k_y=1.0, a=-1.0, b=1.0):
        if not a < b:
            raise ValueError("need a < b")
        self.eps = float(eps)
        self.k_y = float(k_y)
        self.a = float(a)
        self.b = float(b)

    @staticmethod
    def V(x):
        return (x * x - 1.0) ** 2

    @staticmethod
    def dV(x):
        return 4.0 * x * (x * x - 1.0)

    def drift(self, X):
        x, y = X[:, 0], X[:, 1]
        return torch.stack((-self.dV(x), -self.k_y * y), dim=1)

    def noise_std(self, device, dtype):
        return torch.full((2,), math.sqrt(2.0 * self.eps), device=device, dtype=dtype)

    def in_A(self, X):
        return X[:, 0] <= self.a

    def in_B(self, X):
        return X[:, 0] >= self.b

    def boundary_levels(self, X):
        v = torch.full_like(X[:, 0], 2.0 * self.eps)
        return X[:, 0] - self.a, self.b - X[:, 0], v, v

    def reference_state(self):
        return np.array([self.a, 0.0])

    def features(self):
        a = self.a
        return {
            "x": lambda X: np.asarray(X)[:, 0].copy(),
            "y": lambda X: np.asarray(X)[:, 1].copy(),
            "absy": lambda X: np.abs(np.asarray(X)[:, 1]),
            # analogue of the perturbation energy: squared distance from the A-side reference state
            "E": lambda X: 0.5 * ((np.asarray(X)[:, 0] - a) ** 2 + np.asarray(X)[:, 1] ** 2),
        }

    def params(self):
        return {"name": self.name, "eps": self.eps, "k_y": self.k_y, "a": self.a, "b": self.b}

    # ---- exact committor -------------------------------------------------------------
    def _integrand(self):
        grid = np.linspace(self.a, self.b, 20001)
        vmax = float(np.max(self.V(grid)))
        if self.a < 0.0 < self.b:
            vmax = max(vmax, float(self.V(0.0)))
        eps = self.eps
        return lambda s: math.exp((self.V(s) - vmax) / eps)

    def analytic_q(self, x):
        """Exact committor q(x) by adaptive quadrature (cumulative over the sorted points)."""
        x = np.asarray(x, dtype=np.float64)
        xs = np.clip(x.ravel(), self.a, self.b)
        g = self._integrand()
        breaks = [p for p in (-1.0, 0.0, 1.0) if self.a < p < self.b]
        Z = quad(g, self.a, self.b, points=breaks or None, epsabs=0.0, epsrel=1e-13, limit=500)[0]
        order = np.argsort(xs, kind="mergesort")
        u = xs[order]
        knots = np.concatenate(([self.a], u))
        pieces = np.empty(len(u))
        for i in range(len(u)):
            lo, hi = knots[i], knots[i + 1]
            pieces[i] = 0.0 if hi <= lo else quad(g, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
        q = np.empty(len(u))
        q[order] = np.cumsum(pieces) / Z
        return np.clip(q, 0.0, 1.0).reshape(x.shape)
