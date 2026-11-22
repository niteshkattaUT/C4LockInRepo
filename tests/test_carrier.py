"""Selectable carrier frequency: derived frame timing, feasibility, simulator link, live switching."""
import os

import pytest

from c4ppg.config import Config, load_config, parse_frequency
from c4ppg.feasibility import carrier_table, plan, transfer_time_s


@pytest.mark.parametrize("f, n, b, sw, frame_ms", [
    (1000, 20, 2, 11, 22.0),      # the original design, unchanged
    (2000, 40, 4, 21, 22.0),
    (5000, 100, 10, 51, 22.0),
    (10000, 100, 100, 51, 20.0),  # 100-period cap: integration 10 ms
    (20000, 100, 100, 51, 10.0),
    (50000, 100, 100, 51, 4.0),
])
def test_timing_follows_carrier(f, n, b, sw, frame_ms):
    c = Config().with_carrier(f)
    assert (c.n_periods, c.n_blank, c.switch_after_periods) == (n, b, sw)
    assert c.frame_period_s * 1e3 == pytest.approx(frame_ms)
    assert c.validate() == [] or f == 50000   # 50 kHz still valid for the camera; feasibility decides streaming
    # switch lands in the blank periods: FrameSync at total//2, toggle after sw edges
    toggle_period = (n + b) // 2 + sw - 1
    assert n <= toggle_period < n + b


def test_config_sources_keep_timing_derived():
    assert load_config(None, ["ref_freq_hz=20000"]).n_periods == 100
    c = Config().replace(burst_frames=40)            # unrelated change keeps derivation
    assert c.replace(ref_freq_hz=5000).n_blank == 10
    pinned = Config(n_periods=8, n_blank=2, switch_after_periods=5)
    assert pinned.replace(ref_freq_hz=2000).n_periods == 8          # pinned stays pinned
    assert pinned.with_carrier(2000).n_periods == 40                # with_carrier always re-derives
    d = Config().with_carrier(20000).to_dict()
    assert d["n_periods"] is None and Config.from_dict(d).n_periods == 100
    assert Config().with_carrier(20000).to_dict(resolved=True)["n_periods"] == 100
    assert "ref_freq_hz must be" in " ".join(Config(ref_freq_hz=300000).validate())


def test_parse_frequency():
    assert parse_frequency("20k") == 20000 and parse_frequency("1.5 kHz") == 1500
    assert parse_frequency("20000") == 20000 and parse_frequency(5000.0) == 5000


def test_feasibility_by_carrier():
    assert plan(Config().with_carrier(20000)).ok
    p50 = plan(Config().with_carrier(50000))
    assert not p50.ok and p50.mode == "link-limited" and p50.max_burst_link_limited == 18
    ok50 = plan(Config(allow_link_limited=True, burst_frames=18).with_carrier(50000))
    assert ok50.ok and ok50.gap_s * 1e3 <= 100
    rows = {r["freq_hz"]: r for r in carrier_table(Config())}
    assert rows[1000.0]["verdict"] == "streams" and rows[20000.0]["verdict"] == "streams"
    assert "bursts <= 18" in rows[50000.0]["verdict"]
    # light allowed on the sensor scales with the carrier (shorter quarter periods)
    assert rows[20000.0]["pmax_a_mw"] == pytest.approx(20 * rows[1000.0]["pmax_a_mw"], rel=1e-6)


def test_sim_link_limited_full_sensor():
    from c4ppg.camera import open_camera
    cfg = Config(sim_time_scale=0.0, controller="sim", allow_link_limited=True, burst_frames=16).with_carrier(50000)
    cam = open_camera(cfg, sim=True)
    cam.configure()
    cam.ia.sim.controller.mode = "run"
    cam.start()
    metas = [cam.fetch_burst()[1] for _ in range(5)]
    cam.stop()
    cam.close()
    tv = 16 * 0.004
    tt = transfer_time_s(16)
    starts = [m.t0_ns * 1e-9 for m in metas]
    period = [b - a for a, b in zip(starts, starts[1:])]
    # steady state: one burst per link transfer (camera waits for the link)
    assert period[-1] == pytest.approx(tt, rel=0.02)
    assert period[-1] - tv == pytest.approx(tt - tv, abs=0.003)


def test_live_switches_carrier(tmp_path, monkeypatch, capsys):
    """Picking 20 kHz in the window ends the 1 kHz segment and continues at 20 kHz (new file, new offset)."""
    from c4ppg import viewer
    from c4ppg.cli import main
    picks = iter([20000.0, None])
    seen = []

    def fake_viewer(session, stop_evt, snapshot=None, carriers=(), carrier=None, **kw):
        import time
        t0 = time.time()
        while session.stats.bursts < 6 and session.alive and time.time() - t0 < 60:
            time.sleep(0.02)
        session.stop_evt.set()
        seen.append((carrier, session.cfg.n_periods, session.cfg.n_blank,
                     session.controller.model.switch_after))
        return next(picks)
    monkeypatch.setattr(viewer, "run_viewer", fake_viewer)
    rc = main(["live", "--sim", "--bursts", "12", "--set", "sim_shape=(96, 96)", "--set", "sim_time_scale=0",
               "--set", f"save_dir={str(tmp_path)!r}", "--set", "burst_frames=40",
               "--set", "rois={'vessel': [44, 52, 30, 60]}"])
    assert rc == 0
    assert seen[0] == (1000.0, 20, 2, 11)
    assert seen[1] == (20000.0, 100, 100, 51)
    h5 = sorted(f for f in os.listdir(tmp_path) if f.endswith(".h5"))
    assert len(h5) == 2
    from c4ppg.storage import SessionReader
    refs = []
    for f in h5:
        with SessionReader(os.path.join(tmp_path, f)) as rd:
            refs.append(rd.actual["ref_freq_hz"])
    assert sorted(refs) == [1000.0, 20000.0]


def test_cli_carrier_flag(capsys):
    from c4ppg.cli import main
    assert main(["check", "--sim", "--carrier", "20k", "--set", "sim_shape=(96, 96)"]) == 0
    out = capsys.readouterr().out
    assert "20000.000 Hz" in out and "100 + 100" in out
    assert main(["plan", "--carrier", "50k", "--no-carriers"]) == 1
