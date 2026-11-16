"""Interface every system must provide.

SDE convention: dX = drift(X) dt + noise_std * dW, with W a standard Wiener process
and noise_std a per-coordinate vector (additive, diagonal noise).
"""

import numpy as np


class System:
    name = "base"
    dim = 0
    coord_names = ()

    def drift(self, X):
        """Batched drift f(X). X: torch tensor (n, dim) -> (n, dim)."""
        raise NotImplementedError

    def noise_std(self, device, dtype):
        """Per-coordinate noise amplitude, torch tensor (dim,)."""
        raise NotImplementedError

    def in_A(self, X):
        """Boolean torch tensor (n,): state is inside the reactant set A."""
        raise NotImplementedError

    def in_B(self, X):
        """Boolean torch tensor (n,): state is inside the product set B."""
        raise NotImplementedError

    def boundary_levels(self, X):
        """Optional, for the Brownian-bridge crossing test. Returns (l_A, l_B, var_A, var_B):
        level functions with A = {l_A <= 0}, B = {l_B <= 0} (consistent with in_A / in_B) and
        var_* = sum_i (d l / d x_i)^2 noise_std_i^2, the variance rate of the level's increment."""
        raise NotImplementedError

    def has_boundary_levels(self):
        return type(self).boundary_levels is not System.boundary_levels

    def reference_state(self):
        """Origin of the sampling rays (the A-side reference state), numpy (dim,)."""
        raise NotImplementedError

    def features(self):
        """Named scalar candidate coordinates: dict name -> f(numpy (N, dim)) -> numpy (N,)."""
        raise NotImplementedError

    def params(self):
        """JSON-serialisable parameters (used in cache keys)."""
        raise NotImplementedError

    def coordinates(self, X):
        return np.asarray(X, dtype=np.float64)
