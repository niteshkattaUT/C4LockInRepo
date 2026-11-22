"""PPG metrics, the threaded pipeline, HDF5 round trip and the CLI, all on the simulator."""
import os
import time

import numpy as np
import pytest

from c4ppg.camera import open_camera
from c4ppg.config import Config
from c4ppg.controller import SimController
from c4ppg.ppg import PPGTracker, heart_rate, resample_uniform, wave_metrics, bandpass_sos
from c4ppg.processing import BurstResult, Alternation


def test_ac_dc_and_ratio_on_synthetic_traces():
    cfg = Config()
    fs = 1 / 0.044
    t = np.arange(0, 10, 1 / fs)
    sos = bandpass_sos(cfg, fs)
    a = wave_metrics(t, 200 * (1 + 0.010 * np.sin(2 * np.pi * 1.2 * t)), cfg, fs, sos)
    b = wave_metrics(t, 300 * (1 + 0.005 * np.sin(2 * np.pi * 1.2 * t)), cfg, fs, sos)
    assert a.dc == pytest.approx(200, rel=1e-3)
    assert a.ac_dc == pytest.approx(0.020, rel=0.03)       # RMS * 2 sqrt 2 = peak-to-peak of a sine
    assert a.ac_dc / b.ac_dc == pytest.approx(2.0, rel=0.01)
    assert heart_rate(a.filtered, fs) == pytest.approx(72, abs=2)
    assert wave_metrics(t[:30], np.ones(30), cfg, fs, sos) is None       # < ppg_min_window_s


def test_resample_bridges_gap():
    t = np.r_[np.arange(0, 1, 0.1), np.arange(1.3, 2, 0.1)]
    tu, vu = resample_uniform(t, 2 * t, 10)
    assert np.allclose(np.diff(tu), 0.1) and np.allclose(vu, 2 * tu)


def _expected_R(cam, roi):
    """Amplitude-weighted pulsation depth of the phantom in an ROI, LED A over LED B."""
    sc = cam.ia.sim.scene
    y0, y1, x0, x1 = roi
    r = []
    for led in ("A", "B"):
        w = (sc.refl[led] * sc.led_power[led])[y0:y1, x0:x1]
        r.append((w * sc.depth[led][y0:y1, x0:x1]).sum() / w.sum())
    return r[0] / r[1]


def run_session(cfg, n_bursts, save=True):
    from c4ppg.pipeline import Session, record_offset_map
    from c4ppg.storage import session_path
    cam = open_camera(cfg, sim=True)
    cam.configure()
    ctl = SimController(cam.ia.sim.controller)
    offset = record_offset_map(cam, ctl)
    path = session_path(cfg.save_dir) if save else None
    s = Session(cfg, cam, ctl, offset, path, n_bursts)
    s.start()
    s.wait(duration_s=120)
    summary = s.stop()
    return cam, s, summary, path


def test_live_session_recovers_ratio_and_saves(small_cfg):
    cfg = small_cfg.replace(burst_frames=60)
    cam, s, summary, path = run_session(cfg, 10)
    try:
        assert summary["bursts"] == 10 and summary["written"] == 10
        assert summary["dropped_raw"] == 0 and summary["timeouts"] == 0 and not summary["errors"]
        assert summary["alternation"] == {"ok": 10}
        assert summary["gap_ms_max"] == pytest.approx(cfg.sim_gap_s * 1e3, abs=0.01)
        st = s.processor.ppg.compute()
        # vessel: strong pulsation -> tight; background: weak pulsation, noise-limited in 8 s
        for name, tol in (("vessel", 0.08), ("background", 0.15)):
            exp = _expected_R(cam, cfg.rois[name])
            assert st.rois[name].R == pytest.approx(exp, rel=tol), name
            assert st.rois[name].hr_bpm == pytest.approx(cfg.sim_heart_rate_bpm, abs=4), name
        # vessel pulsates more at 760 nm than the background does
        assert st.rois["vessel"].R > st.rois["background"].R
    finally:
        cam.close()

    from c4ppg.offline import reprocess
    from c4ppg.storage import SessionReader
    with SessionReader(path) as rd:
        assert len(rd) == 10 and rd.actual["n_periods"] == 20
        assert rd.offset().I0.shape == (2, 96, 96)
        assert rd.stats()["bursts"] == 10
    out = reprocess(path)
    live_R = [r["R_vessel"] for r in s.processor.results]
    off_R = [r["R_vessel"] for r in out["rows"]]
    np.testing.assert_allclose(off_R, live_R, equal_nan=True)
    assert all(os.path.isfile(f) for f in out["files"])
    # reprocess with a new ROI
    out2 = reprocess(path, {"rois": {"edge": [0, 8, 0, 8]}}, plot=False)
    assert list(out2["tracker"].compute().rois) == ["edge"]


def test_fault_bursts_are_flagged(small_cfg):
    cfg = small_cfg.replace(sim_sync_miss_prob=0.02, sim_seed=3)
    cam, s, summary, _ = run_session(cfg, 8, save=False)
    cam.close()
    alt = summary["alternation"]
    assert alt.get("broken", 0) + alt.get("swapped", 0) + alt.get("indeterminate", 0) >= 1
    assert alt.get("ok", 0) >= 1


def test_slow_processor_drops_instead_of_blocking(small_cfg, monkeypatch):
    from c4ppg import processing
    cfg = small_cfg.replace(raw_q_max=1, sim_time_scale=0.2)
    orig = processing.BurstProcessor.process

    def slow(self, iq, meta):
        time.sleep(0.6)
        return orig(self, iq, meta)
    monkeypatch.setattr(processing.BurstProcessor, "process", slow)
    cam, s, summary, _ = run_session(cfg, 8, save=False)
    cam.close()
    assert summary["bursts"] == 8
    assert summary["dropped_raw"] >= 1
    assert not summary["errors"]


def test_roi_change_from_viewer(small_cfg):
    from c4ppg.pipeline import Processor, Stats
    import queue
    import threading
    p = Processor(small_cfg, queue.Queue(), None, None, None, threading.Event(), Stats())
    p.set_roi("vessel", [1, 5, 1, 5])
    assert p._pending_rois["vessel"] == [1, 5, 1, 5]


def test_cli_commands(tmp_path, capsys):
    from c4ppg.cli import main
    common = ["--sim", "--set", "sim_shape=(96, 96)", "--set", "sim_time_scale=0",
              "--set", f"save_dir={str(tmp_path)!r}", "--set", "burst_frames=40",
              "--set", "rois={'vessel': [44, 52, 30, 60]}"]
    assert main(["check"] + common) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["beat"] + common) == 0
    assert "beat 25.0 Hz" in capsys.readouterr().out
    assert main(["live"] + common + ["--no-view", "--bursts", "8"]) == 0
    out = capsys.readouterr().out
    assert "R[vessel]" in out
    h5 = [f for f in os.listdir(tmp_path) if f.endswith(".h5")]
    assert len(h5) == 1
    for suffix in ("_traces.csv", "_bursts.csv", "_ppg.png", ".log"):
        assert os.path.isfile(os.path.join(tmp_path, h5[0][:-3] + suffix))
    assert main(["reprocess", os.path.join(tmp_path, h5[0])]) == 0
    assert main(["config"]) == 0
