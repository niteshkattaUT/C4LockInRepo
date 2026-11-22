"""Live view on the main thread (matplotlib is not thread-safe).

Top: mean amplitude images for LED A and LED B with the ROIs.
Bottom: band-passed relative pulsation (AC/DC) for both wavelengths in the selected
ROI, and the ratio R over time.  Status line: burst count, drops, gaps, saturation,
alternation state.

Mouse: drag a rectangle on the LED A image to set the "vessel" ROI, on the LED B
image to set the "background" ROI.  Keys: 1-9 pick which ROI the traces show, q quits.
If the view ever becomes the bottleneck, swap this module for pyqtgraph; the
worker threads do not change.
"""
from __future__ import annotations

import queue
import threading

import numpy as np

ROI_COLORS = ["#00e5ff", "#ffd400", "#ff4fd8", "#7CFC00", "#ff8c00", "#ffffff"]


def run_viewer(session, stop_evt: threading.Event, title: str = "heliCam C4 two-LED PPG", snapshot: str = None,
               carriers=(), carrier: float = None):
    """Blocks until the window closes / q / stop_evt.  Returns a newly picked carrier (Hz) or None."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RadioButtons, RectangleSelector

    cfg = session.cfg
    proc = session.processor
    view_q: queue.Queue = session.view_q
    la, lb = cfg.led_a_nm, cfg.led_b_nm

    plt.ion()
    fig = plt.figure(figsize=(13, 8.5))
    if carrier:
        title = f"{title}  -  carrier {carrier / 1e3:g} kHz"
    fig.canvas.manager.set_window_title(title)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.28, wspace=0.22, right=0.80, left=0.06)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_tr, ax_r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    status = fig.text(0.01, 0.985, "waiting for the first burst ...", va="top", family="monospace", fontsize=9)

    im = {}
    for ax, key, nm in ((ax_a, "a", la), (ax_b, "b", lb)):
        im[key] = ax.imshow(np.zeros((8, 8)), cmap="inferno", interpolation="nearest")
        ax.set_title(f"LED A  {nm} nm  (drag = vessel ROI)" if key == "a" else
                     f"LED B  {nm} nm  (drag = background ROI)", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    cb = {k: fig.colorbar(im[k], ax=ax, fraction=0.046, pad=0.02) for k, ax in (("a", ax_a), ("b", ax_b))}

    (l_a,) = ax_tr.plot([], [], color="#d62728", lw=1.2, label=f"{la} nm")
    (l_b,) = ax_tr.plot([], [], color="#1f77b4", lw=1.2, label=f"{lb} nm")
    ax_tr.set_xlabel("time (s)")
    ax_tr.set_ylabel("AC / DC  (band-passed)")
    ax_tr.legend(loc="upper left", fontsize=8)
    ax_tr.grid(alpha=0.3)
    r_lines = {}
    ax_r.set_xlabel("time (s)")
    ax_r.set_ylabel(f"R = (AC/DC)$_{{{la}}}$ / (AC/DC)$_{{{lb}}}$")
    ax_r.grid(alpha=0.3)

    roi_patches: dict = {}
    sel = {"roi": next(iter(proc.proc.rois), None)}

    def draw_rois(rois):
        for p in roi_patches.values():
            for q in p:
                q.remove()
        roi_patches.clear()
        for i, (name, (y0, y1, x0, x1)) in enumerate(rois.items()):
            c = ROI_COLORS[i % len(ROI_COLORS)]
            roi_patches[name] = [ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                                        ec=c, lw=1.5 if name == sel["roi"] else 0.8))
                                 for ax in (ax_a, ax_b)]
            roi_patches[name].append(ax_a.text(x0, y0 - 3, name, color=c, fontsize=8))

    def on_select(name):
        def cb_(e0, e1):
            x0, x1 = sorted((int(round(e0.xdata)), int(round(e1.xdata))))
            y0, y1 = sorted((int(round(e0.ydata)), int(round(e1.ydata))))
            if x1 - x0 < 2 or y1 - y0 < 2:
                return
            proc.set_roi(name, [y0, y1 + 1, x0, x1 + 1])
            rois = dict(proc.proc.rois)
            rois[name] = [y0, y1 + 1, x0, x1 + 1]
            sel["roi"] = name
            draw_rois(rois)
        return cb_

    selectors = [RectangleSelector(ax_a, on_select("vessel"), useblit=False, interactive=False,
                                   props=dict(ec="#00e5ff", fill=False)),
                 RectangleSelector(ax_b, on_select("background"), useblit=False, interactive=False,
                                   props=dict(ec="#ffd400", fill=False))]

    def on_key(ev):
        if ev.key == "q":
            stop_evt.set()
        elif ev.key and ev.key.isdigit():
            names = list(proc.proc.rois)
            i = int(ev.key) - 1
            if 0 <= i < len(names):
                sel["roi"] = names[i]
                draw_rois(proc.proc.rois)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # carrier selector: picking another one ends this run and restarts at the new carrier
    picked = {"carrier": None}
    radio = None
    if carriers and carrier:
        freqs = sorted(set(float(f) for f in carriers) | {float(carrier)})
        labels = [f"{f / 1e3:g} kHz" for f in freqs]
        rax = fig.add_axes([0.885, 0.62, 0.11, 0.035 * len(freqs) + 0.03])
        rax.set_title("carrier", fontsize=9)
        radio = RadioButtons(rax, labels, active=freqs.index(float(carrier)))

        def on_carrier(label):
            f = freqs[labels.index(label)]
            if f != float(carrier):
                picked["carrier"] = f
                status.set_text(f"switching to {label}: stopping, reconfiguring, new offset map ...")
                status.set_color("navy")
                fig.canvas.draw_idle()
                stop_evt.set()
        radio.on_clicked(on_carrier)
        fig.text(0.885, 0.60, "restarts acquisition\n(new file + offset map)", fontsize=7, va="top", color="0.4")
    plt.show(block=False)
    first = True
    while not stop_evt.is_set() and plt.fignum_exists(fig.number):
        try:
            res, state = view_q.get(timeout=0.1)
        except queue.Empty:
            plt.pause(0.05)
            if not session.alive:
                break
            continue

        for key, img in (("a", res.img_a), ("b", res.img_b)):
            im[key].set_data(img)
            lo, hi = np.percentile(img, [1, 99.5])
            im[key].set_clim(lo, hi if hi > lo else lo + 1)
            if first:
                im[key].set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        if first:
            draw_rois(proc.proc.rois)
            first = False

        m = state.rois.get(sel["roi"])
        if m and m.a and m.b:
            l_a.set_data(m.a.t, m.a.filtered * 100)
            l_b.set_data(m.b.t, m.b.filtered * 100)
            ax_tr.set_ylabel("AC / DC  (%, band-passed)")
            ax_tr.relim(), ax_tr.autoscale_view()
            ax_tr.set_title(f"{sel['roi']}:  AC/DC {la} = {m.a.ac_dc * 100:.2f} %   {lb} = {m.b.ac_dc * 100:.2f} %"
                            f"   HR {m.hr_bpm:.0f} bpm", fontsize=9)
        for i, (name, (t, r)) in enumerate(state.R_hist.items()):
            if name not in r_lines:
                (r_lines[name],) = ax_r.plot([], [], "o-", ms=3, lw=1, color=ROI_COLORS[i % len(ROI_COLORS)],
                                             label=name)
                ax_r.legend(loc="upper left", fontsize=8)
            keep = t >= t[-1] - cfg.history_s if t.size else slice(None)
            r_lines[name].set_data(t[keep], r[keep])
        ax_r.relim(), ax_r.autoscale_view()
        Rs = "  ".join(f"R[{n}]={mm.R:.3f}" for n, mm in state.rois.items() if np.isfinite(mm.R))
        ax_r.set_title(Rs or f"R needs {cfg.ppg_min_window_s:g} s of data", fontsize=9)

        s = session.stats.summary()
        alt = res.alternation
        status.set_text(
            f"burst {s['bursts']}  dropped {s['dropped_raw']}  timeouts {s['timeouts']}  "
            f"written {s['written']}  gap {s['gap_ms_mean']:.1f}/{s['gap_ms_max']:.1f} ms (mean/max)  "
            f"proc {res.proc_s * 1e3:.0f} ms\n"
            f"alternation {alt.state} (contrast {alt.contrast:.1f}, match {alt.match:.2f})"
            f"{'  SWAPPED-FIXED' if res.swapped else ''}   "
            f"clipped px {res.sat_clipped}   flat px {res.sat_flat}"
            + ("   !! " + "; ".join(res.flags + res.meta.flags) if (res.flags or res.meta.flags) else ""))
        status.set_color("crimson" if (res.flags or res.meta.flags or alt.state != "ok") else "black")
        fig.canvas.draw_idle()
        plt.pause(0.01)
    stop_evt.set()
    if snapshot and plt.fignum_exists(fig.number):
        fig.savefig(snapshot, dpi=110)
    del selectors, radio
    plt.close(fig)
    return picked["carrier"]
