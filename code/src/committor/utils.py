"""Seeds, atomic file I/O, hashing and run status bookkeeping."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch


def derive_seed(base, *keys):
    """Deterministic 63-bit seed from a base seed and any number of hashable keys."""
    words = [int(base) & 0xFFFFFFFF, (int(base) >> 32) & 0xFFFFFFFF]
    for k in keys:
        h = hashlib.sha256(repr(k).encode()).digest()
        words.extend(int.from_bytes(h[i:i + 4], "little") for i in (0, 4, 8, 12))
    s = np.random.SeedSequence(words).generate_state(1, dtype=np.uint64)[0]
    return int(s >> np.uint64(1))


def torch_generator(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def json_safe(o):
    """Recursively convert numpy types and non-finite floats (-> None) so the output is strict JSON."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def atomic_save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def stable_hash(obj, n=16):
    s = json.dumps(obj, sort_keys=True, default=_json_default)
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def array_hash(a, n=16):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:n]


def tree_hash(paths, n=16):
    """Hash of the contents of a set of files (used to fingerprint the code a run used).

    File names enter relative to their common parent directory, so the hash does not depend on
    where the repository is checked out."""
    paths = [Path(p).resolve() for p in paths]
    root = Path(os.path.commonpath([str(p) for p in paths])) if len(paths) > 1 else paths[0].parent
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:n]


def git_info(path):
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


class Status:
    """status.json with one record per pipeline stage, rewritten atomically after each update."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(self.path, default={"stages": {}})

    def done(self, stage):
        return bool(self.data["stages"].get(stage, {}).get("done", False))

    def get(self, stage):
        return self.data["stages"].get(stage, {})

    def update(self, stage, **info):
        self.data["stages"].setdefault(stage, {}).update(info)
        atomic_write_json(self.path, self.data)
