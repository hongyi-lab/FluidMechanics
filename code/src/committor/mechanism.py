"""Quadratic mechanism-quantity models for Gate 3b (code/configs/gate3b_mfe.yaml).

    c(x) = g(u_1, ..., u_k),   u_i = x^T Q_i x + b_i^T x,   x = a - a_lam,

Q_i symmetric (not forced definite), g a small MLP without monotonicity constraint. Implementation details that do
not change the model class (config choice h3):
  - the quadratic forms act on xs = x / x_scale (fixed per-mode scale); physical Q_i, b_i are returned by physical();
  - g sees z_i = asinh(u_i / tau_i) / Z_SCALE with a learnable tau_i > 0 (a strictly monotone bijection of u_i).
kind = "phys2": u_1 = a3^2 = x_3^2, u_2 = |y|^2 = sum_{i=4..8} x_i^2 (fixed, user item 2), g and tau learnable.

Training: full-batch Adam on binary cross-entropy, early stopping on the val states, tqdm, resumable from a checkpoint
written every `checkpoint_every` epochs.
"""
import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

Z_SCALE = 5.0


def make_g(k, hidden):
    dims = [k] + list(hidden)
    layers = []
    for i in range(len(dims) - 1):
        layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.SiLU()]
    layers.append(torch.nn.Linear(dims[-1], 1))
    return torch.nn.Sequential(*layers).to(torch.float64)


class QuadMech(torch.nn.Module):
    def __init__(self, k, hidden, x_scale, kind="free"):
        super().__init__()
        if kind not in ("free", "phys2"):
            raise ValueError(kind)
        if kind == "phys2" and k != 2:
            raise ValueError("phys2 has k = 2")
        self.k, self.kind, self.hidden = int(k), kind, list(hidden)
        self.register_buffer("x_scale", torch.as_tensor(np.asarray(x_scale, dtype=np.float64)))
        if kind == "free":
            self.A = torch.nn.Parameter(torch.zeros(self.k, 9, 9, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.zeros(self.k, 9, dtype=torch.float64))
        self.log_tau = torch.nn.Parameter(torch.zeros(self.k, dtype=torch.float64))
        self.g = make_g(self.k, self.hidden)

    # ---- quadratic forms ----
    def Qs(self):
        """Q~_i in scaled units (symmetric part of A)."""
        return 0.5 * (self.A + self.A.transpose(1, 2))

    def u(self, x):
        if self.kind == "phys2":
            return torch.stack([x[:, 2] ** 2, (x[:, 3:8] ** 2).sum(1)], dim=1)
        xs = x / self.x_scale
        XQ = torch.einsum("nj,kjl->nkl", xs, self.Qs())
        return (XQ * xs[:, None, :]).sum(-1) + xs @ self.b.T

    def z(self, u):
        return torch.asinh(u * torch.exp(-self.log_tau)) / Z_SCALE

    def from_u(self, u):
        return self.g(self.z(u)).squeeze(1)

    def forward(self, x):
        return self.from_u(self.u(x))

    @torch.no_grad()
    def physical(self):
        """(Q (k, 9, 9), b (k, 9)) acting on x = a - lam (numpy)."""
        if self.kind == "phys2":
            Q = np.zeros((2, 9, 9))
            Q[0, 2, 2] = 1.0
            for i in range(3, 8):
                Q[1, i, i] = 1.0
            return Q, np.zeros((2, 9))
        D = (1.0 / self.x_scale).cpu().numpy()
        Q = self.Qs().cpu().numpy() * D[None, :, None] * D[None, None, :]
        b = self.b.detach().cpu().numpy() * D[None, :]
        return Q, b

    # ---- initialisation ----
    def init_random(self, gen, init_scale, X_tr, tau_rel, slots=None):
        """Random Q~, b~ (slots: which u_i; default all), g reset (PyTorch default ranges, drawn from gen), and tau_i of
        those slots set to tau_rel x RMS of u_i over X_tr (torch, already on the model's device)."""
        slots = list(range(self.k)) if slots is None else list(slots)
        dev = self.log_tau.device
        with torch.no_grad():
            if self.kind == "free":
                for i in slots:
                    self.A[i] = (torch.randn(9, 9, generator=gen, dtype=torch.float64) * (init_scale / 9.0)).to(dev)
                    self.b[i] = (torch.randn(9, generator=gen, dtype=torch.float64) * (init_scale / 3.0)).to(dev)
            self.set_tau(X_tr, tau_rel, slots)

    @torch.no_grad()
    def set_tau(self, X_tr, tau_rel, slots):
        U = torch.cat([self.u(X_tr[i:i + 500_000]) for i in range(0, len(X_tr), 500_000)])
        rms = torch.sqrt((U ** 2).mean(0)).clamp_min(1e-300)
        for i in slots:
            self.log_tau[i] = torch.log(tau_rel * rms[i])

    @torch.no_grad()
    def reset_g(self, gen):
        for m in self.g:
            if isinstance(m, torch.nn.Linear):
                bound = 1.0 / np.sqrt(m.in_features)
                m.weight.copy_((torch.rand(m.weight.shape, generator=gen, dtype=torch.float64) * 2 - 1) * bound)
                m.bias.copy_((torch.rand(m.bias.shape, generator=gen, dtype=torch.float64) * 2 - 1) * bound)

    @torch.no_grad()
    def warm_from(self, old):
        """Copy u_1..u_{k-1}, their tau and g from a (k-1)-model; the new input's first-layer weights are set to 0, so the
        function equals old exactly (the new slot's Q, b, tau must be initialised before or after; they do not matter
        at the start)."""
        if old.k != self.k - 1 or old.hidden != self.hidden or old.kind != "free" or self.kind != "free":
            raise ValueError("warm start needs a free (k-1)-model with the same g")
        j = old.k
        self.A[:j] = old.A
        self.b[:j] = old.b
        self.log_tau[:j] = old.log_tau
        for mn, mo in zip(self.g, old.g):
            if isinstance(mn, torch.nn.Linear):
                if mn.weight.shape == mo.weight.shape:
                    mn.weight.copy_(mo.weight)
                else:                                                   # first layer: (h, k) <- (h, k-1) + zero column
                    mn.weight.zero_()
                    mn.weight[:, :j] = mo.weight
                mn.bias.copy_(mo.bias)

    @torch.no_grad()
    def copy_from_phys2(self, ph):
        """Free k = 2 model set to exactly the phys2 model ph: u1 = a3^2, u2 = |y|^2 as quadratic forms, tau and g copied."""
        if self.kind != "free" or self.k != 2 or ph.kind != "phys2" or ph.hidden != self.hidden:
            raise ValueError("copy_from_phys2 needs a free k = 2 model and a phys2 model with the same g")
        Q, b = ph.physical()
        for i in range(2):
            f = self.set_quadratic(i, Q[i], b[i])
            self.log_tau[i] = ph.log_tau[i] - np.log(f)          # u_i was divided by f: tau_i too (u_i / tau_i unchanged)
        for mn, mo in zip(self.g, ph.g):
            if isinstance(mn, torch.nn.Linear):
                mn.weight.copy_(mo.weight)
                mn.bias.copy_(mo.bias)

    @torch.no_grad()
    def set_quadratic(self, i, Q_phys, b_phys):
        """Set u_i to (x^T Q x + b^T x) / f, Q and b given in physical units, f = norm of (Q~, b~) in scaled units, so that
        the stored parameters are O(1) like a random start (an Adam step of size lr would otherwise destroy a form whose
        scaled entries are ~1e-3; found in the 1% test). Returns f; the caller rescales tau_i by 1/f if it keeps an old
        tau (g sees u_i / tau_i only)."""
        S = self.x_scale.cpu().numpy()
        Qs = np.asarray(Q_phys) * S[:, None] * S[None, :]
        bs = np.asarray(b_phys) * S
        f = float(np.sqrt(np.sum(Qs ** 2) + np.sum(bs ** 2)))
        self.A[i] = torch.as_tensor(Qs / f, device=self.A.device)
        self.b[i] = torch.as_tensor(bs / f, device=self.b.device)
        return f

    def spec(self):
        return {"k": self.k, "kind": self.kind, "hidden": self.hidden}


def build(spec, x_scale, device):
    return QuadMech(spec["k"], spec["hidden"], x_scale, spec["kind"]).to(device)


@torch.no_grad()
def score(model, X, chunk=1_000_000):
    """c on states X (numpy (..., 9), x = a - lam) -> numpy (...)."""
    shp = X.shape[:-1]
    Xf = np.asarray(X, dtype=np.float64).reshape(-1, 9)
    dev = model.log_tau.device
    out = np.empty(len(Xf))
    for i in range(0, len(Xf), chunk):
        out[i:i + chunk] = model(torch.as_tensor(Xf[i:i + chunk], device=dev)).cpu().numpy()
    return out.reshape(shp)


def _bce(z, y):
    return F.binary_cross_entropy_with_logits(z, y)


def _atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(model, Xtr_t, ytr_t, Xva_t, yva_t, tcfg, ckpt_path, final_path, desc="train", progress=True):
    """Full-batch Adam on BCE; early stopping on val BCE. Resumes from ckpt_path; returns info dict (the model holds the
    best state on return). If final_path exists, its state is loaded and training is skipped."""
    final_path, ckpt_path = Path(final_path), Path(ckpt_path)
    if final_path.exists():
        ck = torch.load(final_path, weights_only=False, map_location=model.log_tau.device)
        model.load_state_dict(ck["state"])
        return ck["info"]
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    max_ep, patience, every = int(tcfg["max_epochs"]), int(tcfg["patience"]), int(tcfg["checkpoint_every"])
    lr0, warm_ep = float(tcfg["lr"]), int(tcfg.get("lr_warmup_epochs", 0))
    ep0, best_v, best_state, best_ep, wait, hist = 0, np.inf, None, -2, 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False, map_location=model.log_tau.device)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        ep0, best_v, best_state, best_ep, wait, hist = ck["epoch"], ck["best_v"], ck["best_state"], ck["best_ep"], ck["wait"], ck["hist"]
    if not ckpt_path.exists():
        # the starting state is a candidate too (a warm start is then never worse than its start on the val objective)
        with torch.no_grad():
            v0 = _bce(model(Xva_t), yva_t).item()
        if np.isfinite(v0):
            best_v, best_state, best_ep = v0, copy.deepcopy(model.state_dict()), -1
    stopped = wait >= patience
    done = ep0
    bar = tqdm(range(ep0, max_ep), initial=ep0, total=max_ep, desc=desc, leave=False, disable=not progress, mininterval=2.0)
    for ep in bar:
        if stopped:
            break
        if warm_ep > 0:                                           # linear lr warm-up (stateless in ep: resume-safe)
            for pg in opt.param_groups:
                pg["lr"] = lr0 * min(1.0, (ep + 1) / warm_ep)
        opt.zero_grad()
        loss = _bce(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        done = ep + 1
        with torch.no_grad():
            v = _bce(model(Xva_t), yva_t).item()
        if not np.isfinite(v):
            raise FloatingPointError(f"{desc}: non-finite val loss at epoch {ep}")
        if v < best_v - 1e-12:
            best_v, best_ep, wait = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if (ep + 1) % 50 == 0:
            hist.append([ep, float(loss.item()), v])
        if wait >= patience:
            stopped = True
        if done % every == 0 or stopped:
            _atomic_torch_save({"state": model.state_dict(), "opt": opt.state_dict(), "epoch": done, "best_v": best_v,
                                "best_state": best_state, "best_ep": best_ep, "wait": wait, "hist": hist}, ckpt_path)
        if stopped:
            break
    bar.close()
    model.load_state_dict(best_state)
    with torch.no_grad():
        tr_loss = _bce(model(Xtr_t), ytr_t).item()
    info = {"val_nll": float(best_v), "train_nll": float(tr_loss), "best_epoch": int(best_ep),   # -1: the starting state
            "epochs_run": int(done), "early_stopped": bool(stopped), "history_every_50": hist}
    _atomic_torch_save({"state": model.state_dict(), "info": info, "spec": model.spec()}, final_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    return info


def load(final_path, x_scale, device):
    ck = torch.load(final_path, weights_only=False, map_location=device)
    m = build(ck["spec"], x_scale, device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["info"]
