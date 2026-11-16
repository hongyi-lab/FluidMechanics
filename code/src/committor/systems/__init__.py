"""Dynamical systems with a common interface for the committor pipeline."""

from .base import System
from .langevin2d import Langevin2D
from .mfe import MFE


def build_system(cfg):
    kind = cfg["kind"]
    kwargs = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "langevin2d":
        return Langevin2D(**kwargs)
    if kind == "mfe":
        return MFE(**kwargs)
    raise ValueError(f"unknown system kind: {kind}")


__all__ = ["System", "Langevin2D", "MFE", "build_system"]
