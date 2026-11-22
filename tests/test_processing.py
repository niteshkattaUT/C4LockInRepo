import numpy as np
import pytest

from c4ppg.config import Config
from c4ppg.controller import LedControllerModel
from c4ppg.processing import (BurstProcessor, OffsetMap, amplitude, check_alternation,
                              saturation_counts)


def test_controller_model_alternates_and_resets():
    m = LedControllerModel(switch_after=11)
    m.mode = "run"
    m.on_recording_active()
    leds = [m.simulate_frame(20, 2)[0] for _ in range(6)]
    assert leds == ["A", "B", "A", "B", "A", "B"]
    m.on_recording_active()                     # new burst always starts on A
    assert m.simulate_frame(20, 2)[0] == "A"
    m.miss_next_sync = True                     # fault: one FrameSync edge missed -> no toggle
    assert [m.simulate_frame(20, 2)[0] for _ in range(3)] == ["B", "B", "A"]
    m.miss_next_reset = True                    # ... and the next burst-start reset is missed too
    m.on_recording_active()
    assert m.simulate_frame(20, 2)[0] == "B"    # -> the whole burst is swapped
    m.mode = "A"
    assert m.enabled() == "A" and m.modulated
    m.mode = "offset"
    assert not m.modulated


def test_switch_lands_in_blank_periods():
    """Toggle happens on the last reference edge of the frame: after period 20 of 22."""
    m = LedControllerModel(switch_after=11)
    m.mode = "run"
    m.on_recording_active()
    seen = []
    for p in range(22):
        if p == 11:
            m.on_frame_sync()
        m.on_reference_edge()
        seen.append(m.led)
    assert seen[:21] == ["A"] * 21 and seen[21] == "B"


def _frames(pattern, base=100.0, d=20.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([base + (d if p == "A" else 0.0) for p in pattern])
    return m + np.linspace(0, 5, m.size) + rng.normal(0, noise, m.size)


def test_alternation_states():
    cfg = Config()
    good = _frames("AB" * 20)
    a = check_alternation(good, cfg)
    assert a.state == "ok" and a.sign == 1 and a.match == 1.0
    assert check_alternation(_frames("BA" * 20), cfg, expected_sign=1).state == "swapped"
    broken = np.concatenate([_frames("AB" * 10), _frames("AABB" * 5)])
    assert check_alternation(broken, cfg, expected_sign=1).state in ("broken", "indeterminate")
    assert check_alternation(_frames("AB" * 20, d=0.0), cfg).state == "indeterminate"


def test_amplitude_is_phase_independent():
    H = W = 4
    A_true = 100.0
    offs = OffsetMap.mid_scale((H, W))
    out = []
    for phi in (0.0, 0.7, 2.0):
        I = np.full((4, H, W), 512 + A_true * np.cos(phi))
        Q = np.full((4, H, W), 512 + A_true * np.sin(phi))
        out.append(amplitude(np.round(np.stack([I, Q])).astype(np.uint16), offs).mean())
    assert np.allclose(out, A_true, atol=1.0)


def test_offset_map_per_parity():
    b = np.zeros((2, 4, 3, 3), np.uint16)
    b[:, 0::2] = 500
    b[:, 1::2] = 520
    om = OffsetMap.from_bursts([b, b])
    assert om.I0.shape == (2, 3, 3)
    assert np.allclose(om.I0[0], 500) and np.allclose(om.I0[1], 520)
    A = amplitude(b, om)
    assert np.allclose(A, 0)
    A_shift = amplitude(b, om, parity_shift=1)
    assert np.allclose(A_shift, np.hypot(20, 20))


def test_saturation_counts():
    cfg = Config()
    iq = np.full((2, 4, 5, 5), 600, np.uint16)
    iq[0, 2, 0, 0] = 1023
    iq[1, 1, 0, 1] = 0
    iq[:, :, 4, 4] = 512
    iq[:, 0, 4, 3] = 513
    iq[:, 1:, 4, 3] = 512
    assert saturation_counts(iq, cfg) == (2, 2)


def test_sim_burst_processing(sim_cam, small_cfg):
    sim = sim_cam.ia.sim
    sim.controller.mode = "offset"
    sim_cam.start()
    sim_cam.fetch_burst()
    om = OffsetMap.from_bursts([sim_cam.fetch_burst()[0] for _ in range(2)])
    sim_cam.stop()
    sim.controller.mode = "run"
    sim_cam.start()
    proc = BurstProcessor(small_cfg, om)
    res = [proc.process(*sim_cam.fetch_burst()) for _ in range(3)]
    sim_cam.stop()
    assert all(r.alternation.state == "ok" for r in res)
    assert proc.expected_sign == -1          # 850 nm LED B is brighter in the phantom
    r = res[0]
    assert r.roi_a["vessel"].shape == (20,) and r.t_b[0] - r.t_a[0] == pytest.approx(0.022)
    # 760 nm is absorbed more by the vessel: lower vessel/background contrast ratio at A
    ca = r.roi_a["vessel"].mean() / r.roi_a["background"].mean()
    cb = r.roi_b["vessel"].mean() / r.roi_b["background"].mean()
    assert ca < cb
    assert r.sat_clipped == 0


def test_swapped_burst_is_detected_and_fixed(small_cfg):
    from c4ppg.camera import open_camera
    cam = open_camera(small_cfg, sim=True)
    cam.configure()
    sim = cam.ia.sim
    sim.controller.mode = "run"
    cam.start()
    proc = BurstProcessor(small_cfg, OffsetMap.mid_scale((96, 96)))
    ok = proc.process(*cam.fetch_burst())
    # fault: the controller is left on B (e.g. a missed FrameSync) and misses the reset,
    # so the next burst starts on B
    sim.controller.led = "B"
    sim.controller.miss_next_reset = True
    bad = proc.process(*cam.fetch_burst())
    cam.stop()
    cam.close()
    assert ok.alternation.state == "ok"
    assert bad.alternation.state == "swapped" and bad.swapped
    # after the fix LED A still means the 760 nm image (dimmer than B in the phantom)
    assert bad.img_a.mean() < bad.img_b.mean()
    assert abs(bad.img_a.mean() - ok.img_a.mean()) / ok.img_a.mean() < 0.05


def test_high_sensitivity_saturates(small_cfg):
    from c4ppg.camera import open_camera
    cam = open_camera(small_cfg.replace(sensitivity=0.5), sim=True)
    cam.configure()
    cam.ia.sim.controller.mode = "run"
    cam.start()
    iq, _ = cam.fetch_burst()
    cam.stop()
    cam.close()
    clipped, flat = saturation_counts(iq, small_cfg)
    assert clipped > 0 and flat > 0
