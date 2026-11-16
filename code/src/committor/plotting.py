"""White-background figures (PNG + PDF)."""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
C_PTS, C_FIT, C_REF, C_ALT = "#4C72B0", "#C44E52", "#222222", "#DD8452"


def _save(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def q_vs_coordinate(panels, path):
    """panels: dicts with c, qhat, p, title, xlabel and optional analytic=(xgrid, qgrid)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6), squeeze=False)
        for ax, pn in zip(axes[0], panels):
            ax.scatter(pn["c"], pn["qhat"], s=4, alpha=0.35, color=C_PTS, lw=0, label=r"test states, $n_B/M$")
            o = np.argsort(pn["c"])
            ax.plot(pn["c"][o], pn["p"][o], color=C_FIT, lw=1.6, label="isotonic map (fit on train)")
            if pn.get("analytic") is not None:
                ax.plot(*pn["analytic"], color=C_REF, ls="--", lw=1.1, label="exact q")
            ax.set_xlabel(pn["xlabel"])
            ax.set_ylabel("committor q")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(pn["title"], fontsize=10)
            ax.legend(loc="upper left", fontsize=8, markerscale=3)
        _save(fig, path)


def histogram_test(panels, bins, path, window=None):
    """panels: dicts with title, counts, expected (reference if the prediction were exact), n, std, std_exact."""
    bins = np.asarray(bins)
    centres = 0.5 * (bins[1:] + bins[:-1])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.1), squeeze=False, sharey=False)
        for ax, pn in zip(axes[0], panels):
            ax.bar(centres, pn["counts"], width=np.diff(bins), color=C_PTS, alpha=0.75, edgecolor="white",
                   label=f"MC q (n={pn['n']})")
            ax.step(bins, np.r_[pn["expected"], pn["expected"][-1]], where="post", color=C_REF, lw=1.2,
                    label="expected if prediction exact")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.45 * max(np.max(pn["counts"]), np.max(pn["expected"])))
            ax.set_xlabel(r"MC committor $n_B/M$")
            ax.set_ylabel("states")
            ax.set_title(f"{pn['title']}\nstd {pn['std']:.3f} (if exact {pn['std_exact']:.3f})", fontsize=9)
            ax.legend(fontsize=7, loc="upper center", ncol=1)
        if window is not None:
            fig.suptitle(f"committor histogram test: test states with predicted q in [{window[0]}, {window[1]}]",
                         fontsize=10, y=1.04)
        _save(fig, path)


def q_vs_E(E, qhat, origin, path, xlabel="perturbation energy E"):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for o, lab, col in ((0, "ray states (i)", C_PTS), (1, "trajectory states (ii)", C_ALT)):
            m = origin == o
            ax.scatter(E[m], qhat[m], s=4, alpha=0.35, lw=0, color=col, label=lab)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"MC committor $n_B/M$")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, markerscale=3)
        _save(fig, path)


def gate0_analytic(x, qa, qhat, M, pit, path, xgrid, qgrid):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
        ax = axes[0]
        g = np.linspace(0, 1, 401)
        Mm = float(np.median(M))
        band = 1.96 * np.sqrt(g * (1 - g) / Mm)
        ax.fill_between(g, g - band, g + band, color="0.85", lw=0, zorder=0, label=f"95% binomial band (M={Mm:.0f})")
        ax.scatter(qa, qhat, s=4, alpha=0.3, lw=0, color=C_PTS, zorder=1)
        ax.plot(g, g, color=C_REF, lw=1, zorder=2)
        ax.set_xlabel("exact q(x)")
        ax.set_ylabel(r"MC estimate $n_B/M$")
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.scatter(x, qhat, s=4, alpha=0.3, lw=0, color=C_PTS, label="MC estimate")
        ax.plot(xgrid, qgrid, color=C_FIT, lw=1.4, label="exact q(x)")
        ax.set_xlabel("x")
        ax.set_ylabel("committor q")
        ax.legend(fontsize=8, markerscale=3)
        ax = axes[2]
        ax.hist(pit, bins=20, range=(0, 1), color=C_PTS, alpha=0.75, edgecolor="white")
        ax.axhline(len(pit) / 20, color=C_REF, lw=1, ls="--", label="uniform (null)")
        ax.set_xlabel("randomised PIT of $n_B$ under Binomial(M, q exact)")
        ax.set_ylabel("states")
        ax.legend(fontsize=8)
        _save(fig, path)


# ------------------------------------------------------------------------------------------------
# Gate 1 (MFE model)
# ------------------------------------------------------------------------------------------------

def gate1_lifetimes(T, t_max, fit, paper, t0, path, T_paper=None):
    """(a) lifetime histogram on semi-log axes; (b) survival P(t) with the exponential fit and the
    fit line digitised from NJP figure 10 (Re = 400)."""
    T = np.asarray(T, dtype=float)
    fin = T[np.isfinite(T)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
        ax = axes[0]
        bins = np.linspace(0, t_max, 41)
        cnt, _ = np.histogram(fin, bins=bins)
        cen = 0.5 * (bins[1:] + bins[:-1])
        ax.bar(cen, np.where(cnt > 0, cnt, np.nan), width=np.diff(bins), color=C_PTS, alpha=0.75,
               edgecolor="white", label=f"decayed by t = {t_max:.0f}: {len(fin)} of {len(T)}")
        w = np.diff(bins)[0]
        tt = np.linspace(t0, t_max, 200)
        ax.plot(tt, len(T) * fit["P_t0"] * np.exp(-(tt - t0) / fit["tau"]) * (1 - np.exp(-w / fit["tau"])),
                color=C_FIT, lw=1.4, label=rf"exponential fit, $\tau$ = {fit['tau']:.0f}")
        ax.set_yscale("log")
        ax.set_xlabel("lifetime t")
        ax.set_ylabel(f"trajectories per bin (width {w:.0f})")
        ax.set_xlim(0, t_max)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("(a) lifetime histogram", fontsize=10)
        ax = axes[1]
        grid = np.linspace(0, t_max, 2001)
        Ts = np.sort(np.where(np.isfinite(T), T, np.inf))
        P = 1.0 - np.searchsorted(Ts, grid, side="right") / len(T)
        ax.step(grid, P, where="post", color=C_PTS, lw=1.4, label=f"this work ({len(T)} ICs)")
        ax.plot(tt, fit["P_t0"] * np.exp(-(tt - t0) / fit["tau"]), color=C_FIT, lw=1.2, ls="--",
                label=rf"fit, $\tau$ = {fit['tau']:.0f} $\pm$ {fit['tau_se']:.0f}")
        tp = np.linspace(paper["t_onset"], t_max, 200)
        ax.plot(tp, np.exp(-(tp - paper["t_onset"]) / paper["tau"]), color=C_REF, lw=1.2, ls=":",
                label=rf"NJP fig. 10 fit line, $\tau$ = {paper['tau']:.0f}")
        if T_paper is not None:
            Tp = np.sort(np.where(np.isfinite(T_paper), T_paper, np.inf))
            Pp = 1.0 - np.searchsorted(Tp, grid, side="right") / len(Tp)
            ax.step(grid, Pp, where="post", color=C_ALT, lw=1.2,
                    label=f"NJP fig. 10 curve (from PDF vectors, {len(Tp)} ICs)")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 1.05)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(0, t_max)
        ax.set_xlabel("t")
        ax.set_ylabel("P(lifetime > t)")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("(b) survival probability, Re = 400", fontsize=10)
        fig.tight_layout(w_pad=2.0)
        _save(fig, path)


def gate1_jacobian(Re, max_real, path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        ax.plot(Re, -np.asarray(max_real), color=C_PTS, lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("Re")
        ax.set_ylabel(r"$-\max_i \mathrm{Re}\,\lambda_i$ (laminar state)")
        ax.set_title("slowest laminar decay rate (all > 0 = stable)", fontsize=10)
        _save(fig, path)


def gate1_timeseries(t, rec_a, lifetimes, path):
    """Amplitudes a1..a9 of the recorded trajectory whose lifetime is closest to the median of the
    recorded ones (cf. NJP figure 5)."""
    lt = np.asarray(lifetimes, dtype=float)
    finite = np.where(np.isfinite(lt))[0]
    j = int(finite[np.argmin(np.abs(lt[finite] - np.median(lt[finite])))]) if len(finite) else 0
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(5, 2, figsize=(9.0, 7.6), sharex=True)
        for i in range(9):
            ax = axes[i % 5, i // 5]
            ax.plot(t, rec_a[:, j, i], color=C_REF, lw=0.5)
            ax.set_ylabel(f"$a_{i + 1}$")
        axes[4, 1].axis("off")
        for ax in (axes[4, 0], axes[3, 1]):
            ax.set_xlabel("t")
            ax.xaxis.set_tick_params(labelbottom=True)
        fig.suptitle(f"recorded trajectory {j}, lifetime {lt[j]:.0f}" if np.isfinite(lt[j]) else
                     f"recorded trajectory {j} (did not decay)", fontsize=10)
        fig.tight_layout()
        _save(fig, path)


def gate1_energy_hist(edges, counts, thresholds, path):
    """Time-occupation histogram of log10 E over all trajectories (with under/overflow bins dropped)."""
    edges = np.asarray(edges)
    c = np.asarray(counts)[1:-1].astype(float)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        cen = 0.5 * (edges[1:] + edges[:-1])
        ax.bar(cen, np.where(c > 0, c / c.sum(), np.nan), width=np.diff(edges), color=C_PTS, alpha=0.75,
               edgecolor="white")
        for th in thresholds:
            ax.axvline(np.log10(th), color=C_FIT, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10} E$,  $E = \sum_i (a_i - a_i^{lam})^2$")
        ax.set_ylabel("fraction of (trajectory, time) samples")
        ax.set_title("occupation of perturbation energy (dashed: lifetime thresholds)", fontsize=9)
        _save(fig, path)
