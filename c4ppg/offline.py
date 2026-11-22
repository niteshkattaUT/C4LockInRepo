"""Exports and offline reprocessing of a saved HDF5 session.

Every live session ends with ``<session>_traces.csv`` (ROI amplitude per frame and
wavelength), ``<session>_bursts.csv`` (per-burst checks and R) and ``<session>_ppg.png``.
``reprocess`` rebuilds the same three files from the raw HDF5, optionally with new ROIs
or PPG settings, using exactly the same code as the live pipeline.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

from .config import Config
from .ppg import PPGTracker
from .processing import BurstProcessor, OffsetMap


def write_traces_csv(path: str, tracker: PPGTracker, cfg: Config) -> str:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "wavelength_nm", "t_s", "amplitude"])
        for name, (ta, va, tb, vb) in tracker.traces().items():
            for nm_, t, v in ((cfg.led_a_nm, ta, va), (cfg.led_b_nm, tb, vb)):
                for x, y in zip(t, v):
                    w.writerow([name, nm_, f"{x:.6f}", f"{y:.5f}"])
    return path


def write_bursts_csv(path: str, rows: list[dict]) -> str:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    return path


def summary_plot(path: str, tracker: PPGTracker, cfg: Config, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from .ppg import bandpass_sos
    from scipy import signal

    tr = tracker.traces()
    fig, axs = plt.subplots(len(tr) or 1, 2, figsize=(13, 3.2 * max(1, len(tr))), squeeze=False)
    sos = bandpass_sos(cfg, tracker.fs)
    for i, (name, (ta, va, tb, vb)) in enumerate(tr.items()):
        ax = axs[i, 0]
        for t, v, nm_, c in ((ta, va, cfg.led_a_nm, "#d62728"), (tb, vb, cfg.led_b_nm, "#1f77b4")):
            if v.size > 3 * (2 * sos.shape[0] + 1):
                tu = t[0] + np.arange(int((t[-1] - t[0]) * tracker.fs) + 1) / tracker.fs
                vu = np.interp(tu, t, v)
                ax.plot(tu, signal.sosfiltfilt(sos, vu - vu.mean()) / vu.mean() * 100, color=c, lw=0.9,
                        label=f"{nm_} nm")
        ax.set_title(f"{name}: band-passed AC/DC", fontsize=10)
        ax.set_xlabel("time (s)"), ax.set_ylabel("%"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
        ax = axs[i, 1]
        h = tracker.R_hist.get(name)
        if h and h[0]:
            ax.plot(h[0], h[1], "o-", ms=3, color="k")
            ax.set_title(f"{name}: R (rolling {cfg.ppg_window_s:g} s)  median {np.nanmedian(h[1]):.3f}", fontsize=10)
        ax.set_xlabel("time (s)"), ax.set_ylabel("R"), ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_session(base: str, tracker: PPGTracker, rows: list[dict], cfg: Config, title: str = "") -> list[str]:
    return [write_traces_csv(base + "_traces.csv", tracker, cfg),
            write_bursts_csv(base + "_bursts.csv", rows),
            summary_plot(base + "_ppg.png", tracker, cfg, title)]


def reprocess(path: str, cfg_overrides: Optional[dict] = None, out_base: Optional[str] = None,
              plot: bool = True) -> dict:
    """Run the live pipeline's processing on a saved session; returns {'tracker', 'rows', 'files'}."""
    from .pipeline import Processor
    from .storage import SessionReader

    with SessionReader(path) as rd:
        cfg = rd.cfg if not cfg_overrides else Config.from_dict({**rd.cfg.to_dict(), **cfg_overrides})
        offset = rd.offset()
        T = None
        a = rd.actual
        if a.get("ref_freq_hz"):
            T = (a["n_periods"] + a["n_blank"]) / a["ref_freq_hz"]
        tracker = PPGTracker(cfg, T)
        proc = None
        rows = []
        for iq, meta in rd.bursts():
            if proc is None:
                proc = BurstProcessor(cfg, offset or OffsetMap.mid_scale(iq.shape[2:]))
            res = proc.process(iq, meta)
            tracker.add(res)
            state = tracker.compute()
            rows.append(Processor._summary(res, state))
    files = []
    if out_base is None:
        out_base = os.path.splitext(path)[0] + "_reprocessed"
    if plot:
        files = export_session(out_base, tracker, rows, cfg, os.path.basename(path))
    return {"tracker": tracker, "rows": rows, "files": files, "cfg": cfg}
