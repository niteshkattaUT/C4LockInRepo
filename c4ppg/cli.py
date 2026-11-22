"""Command line.  Every command takes --sim to run against the simulated camera.

    python -m c4ppg config                     print the full configuration (YAML)
    python -m c4ppg check     [--sim]          build step 1: configure, print actual values, pass/fail
    python -m c4ppg beat      [--sim]          build step 2: internal LED module beat test (25 Hz in I/Q)
    python -m c4ppg live      [--sim]          offset map, then live two-LED PPG (+ raw HDF5)
    python -m c4ppg reprocess FILE.h5          offline: rebuild traces / R from the raw data

Common options: --config my.yaml, --set key=value (repeatable).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

from .config import Config, load_config

log = logging.getLogger("c4ppg")


def _common(p: argparse.ArgumentParser):
    p.add_argument("--sim", action="store_true", help="use the simulated camera and LED controller")
    p.add_argument("--config", help="YAML file with config overrides")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config value")


def _cfg(args) -> Config:
    return load_config(args.config, args.set)


def _print_actual(actual: dict):
    print("Actual camera settings (read back):")
    for k, v in actual.items():
        print(f"  {k:<18} {v}")


# ---------------------------------------------------------------------------------------
def cmd_config(args) -> int:
    cfg = _cfg(args)
    print(cfg.dump_yaml(), end="")
    problems = cfg.validate()
    for p in problems:
        print(f"# INVALID: {p}")
    return 1 if problems else 0


def check_actuals(cfg: Config, a: dict) -> list[tuple[str, bool, str]]:
    exp_fr = 1.0 / cfg.frame_period_s
    return [
        ("reference frequency", abs(a["ref_freq_hz"] - cfg.ref_freq_hz) <= 0.01 * cfg.ref_freq_hz,
         f"{a['ref_freq_hz']:.3f} Hz (target {cfg.ref_freq_hz:g})"),
        ("periods + blanks", (a["n_periods"], a["n_blank"]) == (cfg.n_periods, cfg.n_blank),
         f"{a['n_periods']} + {a['n_blank']} (target {cfg.n_periods} + {cfg.n_blank})"),
        ("frame rate", abs(a["frame_rate"] - exp_fr) <= 0.02 * exp_fr,
         f"{a['frame_rate']:.2f} fps (expected {exp_fr:.2f})"),
        ("burst frames", a["burst_frames"] == cfg.burst_frames, f"{a['burst_frames']}"),
        ("image size", (a["height"], a["width"]) == (542, 512) or bool(cfg.sim_shape),
         f"{a['height']} x {a['width']}"),
    ]


def cmd_check(args) -> int:
    from .camera import open_camera
    cfg = _cfg(args)
    cam = open_camera(cfg, args.sim)
    try:
        a = cam.configure()
        _print_actual(a)
        from .feasibility import plan
        p = plan(cfg, (a["height"], a["width"]))
        print(f"  link               {p.wire_mbit_s:.0f} Mbit/s of 800 ({p.link_util * 100:.0f} % busy), "
              f"host {p.host_mb_s:.0f} MB/s unpacked")
        ok = True
        print("\nBuild step 1 - connect and configure:")
        for name, passed, txt in check_actuals(cfg, a):
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {txt}")
        print(f"  routed: {cfg.frame_sync_line}=LockInFrameSync, {cfg.reference_line}=LockInReference"
              + (f", {cfg.recording_active_line}=RecordingActive" if cfg.recording_active_line
                 else "  (RecordingActive: no DIO line chosen yet)"))
        print()
        print("Streaming (Heliotis calculator):")
        for ch in p.checks:
            tag = ("PASS" if ch.ok else "FAIL") if ch.hard else ("ok" if ch.ok else "WARN")
            print(f"  [{tag}] {ch.name}: {ch.detail}")
        return 0 if ok and p.ok else 1
    finally:
        cam.close()


def cmd_plan(args) -> int:
    """Offline: is this configuration real-time capable?  No camera needed."""
    from .feasibility import calculator, format_plan, plan
    cfg = _cfg(args)
    if args.spreadsheet:
        b = calculator()
        print("Spreadsheet defaults (fd 10 kHz, Np 100, Nf 900, bs 1, 45 mW, 700 nm, S40U):")
        for k in ("tf_us", "tv_s", "tt_rawiq_s", "tt_amp_s", "Npe", "fullwell_fill_pct", "noise_cnt",
                  "racdc_frame", "racdc_volume", "snr_frame", "snr_volume"):
            print(f"  {k:<18} {getattr(b, k):.6g}")
        print()
    shape = tuple(cfg.sim_shape) if (args.sim and cfg.sim_shape) else (542, 512)
    p = plan(cfg, shape, args.proc_ms / 1e3 if args.proc_ms else None)
    print(format_plan(cfg, p))
    print()
    print(f"VERDICT: {'real-time capable' if p.ok else 'NOT real-time capable with these settings'}")
    return 0 if p.ok else 1


def cmd_beat(args) -> int:
    """Reproduce c4DemodSimple.py: internal LED module 25 Hz off the reference -> 25 Hz beat in I/Q."""
    from .camera import open_camera
    cfg = _cfg(args).replace(ref_freq_hz=args.ref, n_periods=10, n_blank=0, burst_frames=60,
                             sensitivity=args.sensitivity, coupling="DC", switch_after_periods=5)
    cam = open_camera(cfg, args.sim)
    try:
        a = cam.configure()
        cam.set_internal_ldm(True, freq_hz=a["ref_freq_hz"] - args.beat, amplitude_pct=10.0, offset_pct=20.0)
        print(f"reference {a['ref_freq_hz']:.3f} Hz, LED module at {a['ref_freq_hz'] - args.beat:.3f} Hz, "
              f"{a['frame_rate']:.1f} fps")
        time.sleep(0 if args.sim else 2.5)        # let the illumination settle (as in the example)
        cam.start()
        try:
            iq, _ = cam.fetch_burst()
        finally:
            cam.stop()
        H, W = iq.shape[2:]
        I = iq[0, 2:, H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10].astype(float).mean(axis=(1, 2))
        Q = iq[1, 2:, H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10].astype(float).mean(axis=(1, 2))
        z = (I - I.mean()) + 1j * (Q - Q.mean())
        # beat = rotation rate of the I/Q phasor (more precise than an FFT over ~2 beat cycles)
        f_peak = abs(float(np.angle(np.sum(z[1:] * np.conj(z[:-1])))) * a["frame_rate"] / (2 * np.pi))
        amp = np.ptp(I) / 2
        ok = abs(f_peak - args.beat) < 2.0 and amp > 5
        print(f"[{'PASS' if ok else 'FAIL'}] beat {f_peak:.1f} Hz (expected {args.beat:g}), I swing +/-{amp:.0f} counts")
        if args.plot:
            import matplotlib.pyplot as plt
            t = np.arange(I.size) / a["frame_rate"] * 1e3
            plt.plot(t, I, "o:", label="I"), plt.plot(t, Q, "o:", label="Q")
            plt.xlabel("ms"), plt.ylabel("raw (centre 20x20 mean)"), plt.legend(), plt.title("beat test")
            plt.show()
        return 0 if ok else 1
    finally:
        try:
            cam.set_internal_ldm(False)
        except Exception:   # noqa: BLE001
            pass
        cam.close()


def cmd_live(args) -> int:
    from .camera import open_camera
    from .controller import make_controller
    from .offline import export_session
    from .pipeline import Session, record_offset_map
    from .processing import OffsetMap
    from .storage import session_path

    cfg = _cfg(args)
    if args.sim:
        cfg = cfg.replace(controller="sim")
    if args.no_save:
        cfg = cfg.replace(save_raw=False)
    os.makedirs(cfg.save_dir, exist_ok=True)
    h5_path = session_path(cfg.save_dir) if cfg.save_raw else None
    base = os.path.splitext(h5_path or session_path(cfg.save_dir))[0]
    fh = logging.FileHandler(base + ".log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s"))
    log.addHandler(fh)

    cam = open_camera(cfg, args.sim)
    controller = None
    session = None
    summary = {}
    try:
        actual = cam.configure()
        _print_actual(actual)
        log.info("actual %s", actual)
        from .feasibility import plan
        pl = plan(cfg, (actual["height"], actual["width"]))
        bad = [c for c in pl.checks if not c.ok]
        for c in bad:
            msg = f"{'NOT REAL-TIME' if c.hard else 'note'}: {c.name}: {c.detail}"
            print(msg)
            log.warning(msg)
        if not pl.ok and not args.force:
            print("These settings cannot stream in real time (see `python -m c4ppg plan`). Use --force to run anyway.")
            return 2
        controller = make_controller(cfg.controller, cfg.controller_baud,
                                     cam.ia.sim.controller if args.sim else None)
        if not args.sim:
            print("\nSAFETY: before any human retinal use, check the combined LED irradiance against "
                  "ANSI Z136 / ISO 15004-2 for both wavelengths.\n")
        if args.no_offset:
            offset = OffsetMap.mid_scale((actual["height"], actual["width"]))
            print("offset map: nominal 512 (no recording)")
        else:
            print(f"recording offset map ({cfg.offset_bursts} bursts, controller in OFFSET mode) ...")
            offset = record_offset_map(cam, controller)
            print(f"  I0 mean {offset.I0.mean():.1f}, Q0 mean {offset.Q0.mean():.1f}")
        session = Session(cfg, cam, controller, offset, h5_path, args.bursts)
        session.start()
        print(f"streaming ({cfg.burst_frames} frames per burst, {cfg.burst_duration_s:.2f} s) - "
              + ("close the window or press q to stop" if not args.no_view else "Ctrl+C to stop"))
        try:
            if args.no_view:
                session.wait(args.duration)
            else:
                from .viewer import run_viewer
                if args.duration:
                    import threading
                    threading.Timer(args.duration, session.stop_evt.set).start()
                run_viewer(session, session.stop_evt, snapshot=base + "_view.png")
        except KeyboardInterrupt:
            print("\nstopping ...")
    finally:
        if session is not None:
            summary = session.stop()
        elif controller is not None:
            try:
                controller.set_mode("off")
            except Exception:   # noqa: BLE001
                pass
        cam.close()
        if controller is not None:
            controller.close()
    if session is not None:
        files = export_session(base, session.processor.ppg, session.processor.results, cfg,
                               os.path.basename(base))
        print("\nsession summary:")
        for k, v in summary.items():
            print(f"  {k:<15} {v}")
        for name, m in session.processor.ppg.compute().rois.items():
            print(f"  R[{name}] = {m.R:.3f}   HR {m.hr_bpm:.0f} bpm")
        print("files:")
        for f in ([h5_path] if h5_path else []) + files + [base + ".log"]:
            print(f"  {f}")
    log.removeHandler(fh)
    fh.close()
    return 0 if not summary.get("errors") else 1


def cmd_reprocess(args) -> int:
    from .offline import reprocess
    over = load_config(None, args.set).to_dict() if args.set else None
    if over:
        default = Config().to_dict()
        over = {k: v for k, v in over.items() if v != default[k]}
    out = reprocess(args.file, over)
    for name, m in out["tracker"].compute().rois.items():
        print(f"R[{name}] = {m.R:.3f}   HR {m.hr_bpm:.0f} bpm")
    for f in out["files"]:
        print(f)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(threadName)s: %(message)s")
    log.setLevel(logging.INFO)
    ap = argparse.ArgumentParser(prog="python -m c4ppg", description="heliCam C4 two-LED real-time PPG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("config", help="print the full configuration as YAML")
    _common(p)
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("check", help="build step 1: configure and verify the actual values")
    _common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("plan", help="real-time feasibility of the config (Heliotis calculator), no camera needed")
    _common(p)
    p.add_argument("--proc-ms", type=float, help="measured host processing time per burst (from a live summary)")
    p.add_argument("--spreadsheet", action="store_true", help="also print the spreadsheet's own example numbers")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("beat", help="build step 2: internal LED module beat test")
    _common(p)
    p.add_argument("--ref", type=float, default=10000.0, help="reference frequency (Hz)")
    p.add_argument("--beat", type=float, default=25.0, help="LED module offset from the reference (Hz)")
    p.add_argument("--sensitivity", type=float, default=0.2)
    p.add_argument("--plot", action="store_true")
    p.set_defaults(fn=cmd_beat)

    p = sub.add_parser("live", help="offset map + live two-LED PPG")
    _common(p)
    p.add_argument("--bursts", type=int, help="stop after this many bursts")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--no-view", action="store_true", help="headless (e.g. 5-minute streaming test)")
    p.add_argument("--no-save", action="store_true", help="do not write raw HDF5")
    p.add_argument("--no-offset", action="store_true", help="skip the offset map (use nominal 512)")
    p.add_argument("--force", action="store_true", help="run even if the settings fail the real-time checks")
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("reprocess", help="rebuild traces and R from a saved HDF5 session")
    p.add_argument("file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override processing settings, e.g. rois=\"{'vessel':[250,270,240,260]}\"")
    p.set_defaults(fn=cmd_reprocess)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
