"""The Heliotis C4 calculator in code must reproduce the spreadsheet, and drive the streaming checks."""
import pytest

from c4ppg.config import Config
from c4ppg.feasibility import (burst_options, calculator, max_streaming_fps, plan, qe_at,
                               transfer_time_s)


def test_reproduces_spreadsheet_values():
    """Numbers saved in C4_calculator_20240712_S4_S3.xlsx (fd 10 kHz, Np 100, Nf 900, bs 1, 45 mW, 700 nm)."""
    b = calculator()
    assert b.tqp_us == pytest.approx(25)
    assert b.tf_us == pytest.approx(10100)
    assert b.tv_s == pytest.approx(9.09)
    assert b.tt_amp_s == pytest.approx(10.065144, rel=1e-9)
    assert b.tt_rawiq_s == pytest.approx(6.660096, rel=1e-9)
    assert b.Pd == pytest.approx(0.28125)
    assert b.Pp_uW == pytest.approx(0.08918790359778599, rel=1e-12)
    assert b.Nph == pytest.approx(7851837.77523521, rel=1e-12)
    assert b.Npe == pytest.approx(5449175.416013236, rel=1e-12)
    assert b.Nper == pytest.approx(181639.18053377455, rel=1e-12)
    assert b.fullwell_fill_pct == pytest.approx(36.32783610675491, rel=1e-12) and b.fits
    assert b.Npef == pytest.approx(1089835083.2026472, rel=1e-12)
    assert b.psn_e == pytest.approx(46686.937856377925, rel=1e-12)
    assert b.psn_cnt == pytest.approx(101.49334316603897, rel=1e-12)
    assert b.rn_cnt == pytest.approx(5.728001396647874, rel=1e-12)
    assert b.rn_e == pytest.approx(2634.880642458022, rel=1e-12)
    assert b.noise_e == pytest.approx(46761.23140385948, rel=1e-12)
    assert b.noise_cnt == pytest.approx(101.6548508779554, rel=1e-12)
    assert b.racdc_frame == pytest.approx(0.00008581340814693621, rel=1e-12)
    assert b.racdc_volume == pytest.approx(0.000002860446938231207, rel=1e-12)
    assert b.racdc_10x10 == pytest.approx(0.00000858134081469362, rel=1e-12)
    assert b.snr_frame == pytest.approx(466.12762345377274, rel=1e-12)
    assert b.snr_volume == pytest.approx(13983.828703613182, rel=1e-12)


def test_qe_interpolation():
    assert qe_at(700) == (pytest.approx(69.4), False)
    assert qe_at(760)[0] == pytest.approx(58.4 + (45.5 - 58.4) * 0.2) and not qe_at(760)[1]
    q850, ext = qe_at(850)
    assert ext and q850 == pytest.approx(32.6)


def test_link_model():
    assert transfer_time_s(1) == pytest.approx(64 / 3 * 542 * 512 / 8e8)      # 7.40 ms per frame
    assert transfer_time_s(1) * 1e3 == pytest.approx(7.4, abs=0.01)
    assert max_streaming_fps() == pytest.approx(1 / (1.25 * transfer_time_s(1)))


def test_default_config_streams_in_real_time():
    p = plan(Config())
    assert p.ok
    assert p.tf_s == pytest.approx(0.022) and p.tv_s == pytest.approx(2.2)
    assert p.link_util == pytest.approx(transfer_time_s(1) / 0.022)          # ~34 %
    assert p.link_util_worst < 0.5


def test_settings_that_cannot_stream():
    # the beat-test settings: 1 ms frames need 7.4 ms of link each
    p = plan(Config(ref_freq_hz=10000, n_periods=10, n_blank=0, switch_after_periods=5, ppg_band_hz=(0.5, 5.0)))
    link = [c for c in p.checks if c.name.startswith("link")][0]
    assert not link.ok and not p.ok
    # 100 fps fits nominally but not with the +25 % margin? 10 ms frames: 74 % / 93 %
    p = plan(Config(ref_freq_hz=1000, n_periods=8, n_blank=2, switch_after_periods=5))
    assert p.link_util == pytest.approx(0.74, abs=0.01) and p.ok
    # no pipelining is flagged
    assert not plan(Config(queue_blocks=1)).ok
    # PPG band above the per-wavelength Nyquist: 60 ms frames -> 8.3 Hz per wavelength
    assert not plan(Config(n_periods=58, n_blank=2, ppg_band_hz=(0.5, 5.0))).ok


def test_ac_coupling_adds_a_period():
    assert plan(Config(coupling="AC")).tf_s == pytest.approx(0.023)


def test_led_power_window_and_full_well():
    from c4ppg.feasibility import power_limits
    cfg = Config()
    lim = power_limits(cfg, 760, 400)
    # 1 kHz carrier: 250 us quarter periods -> well under 1 mW on the sensor before 50 % full well (DC coupling)
    assert lim["p_max_mw"] == pytest.approx(45 * 50 / calculator(1000, 20, 100, 0, 45, 760).fullwell_fill_pct)
    assert lim["p_max_mw"] < 1.0
    assert 0 < lim["p_min_mw"] < lim["p_max_mw"]
    assert calculator(1000, 20, 100, 0, lim["p_max_mw"], 760).fullwell_fill_pct == pytest.approx(50)
    # AC coupling (background suppression /30) allows 30x more light
    assert power_limits(cfg.replace(coupling="AC"), 760, 400)["p_max_mw"] == pytest.approx(30 * lim["p_max_mw"],
                                                                                          rel=1e-9)
    # measured power above the limit is a hard failure
    p = plan(cfg.replace(led_power_mw=(45.0, 0.1)))
    fw_a = [c for c in p.checks if c.name.startswith("LED A") and "full well" in c.name][0]
    fw_b = [c for c in p.checks if c.name.startswith("LED B") and "full well" in c.name][0]
    assert not fw_a.ok and fw_b.ok and not p.ok


def test_burst_options_table():
    rows = burst_options(Config())
    r100 = [r for r in rows if r["frames"] == 100][0]
    assert r100["burst_s"] == pytest.approx(2.2) and r100["transfer_s"] == pytest.approx(100 * transfer_time_s(1))
    assert all(r["ok"] for r in rows)


def test_cli_plan(capsys):
    from c4ppg.cli import main
    assert main(["plan", "--spreadsheet"]) == 0
    out = capsys.readouterr().out
    assert "6.6601" in out and "real-time capable" in out
    assert main(["plan", "--set", "queue_blocks=1"]) == 1


def test_live_refuses_non_realtime_settings(tmp_path):
    from c4ppg.cli import main
    rc = main(["live", "--sim", "--no-view", "--bursts", "2", "--set", "queue_blocks=1",
               "--set", "sim_shape=(96, 96)", "--set", "sim_time_scale=0", "--set", f"save_dir={str(tmp_path)!r}"])
    assert rc == 2
