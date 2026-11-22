"""Camera wrapper against the spec-checked simulated node map."""
import numpy as np
import pytest

from c4ppg.camera import decode, open_camera
from c4ppg.config import Config
from c4ppg.sim import AccessException


def test_configure_actuals(sim_cam, small_cfg):
    a = sim_cam.actual
    assert a["ref_freq_hz"] == pytest.approx(1000.0, rel=1e-6)
    assert (a["n_periods"], a["n_blank"]) == (20, 2)
    assert a["frame_rate"] == pytest.approx(1000 / 22, rel=1e-6)
    assert sim_cam.frame_period_s == pytest.approx(0.022)
    nm = sim_cam.nm
    assert nm.selected("LineSource", "RTIO2") == "LockInFrameSync"
    assert nm.selected("LineSource", "RTIO3") == "LockInReference"
    assert nm.selected("ChunkEnable", "Timestamp") and nm.selected("ChunkEnable", "FrameID")
    for trig in ("RecordingStart", "FrameStart", "AcquisitionStart"):
        assert nm.selected("TriggerMode", trig) == "Off"
    assert nm.selected("TransferQueueMaxBlockCount", "All") == 2
    assert nm.DeviceOperationMode.value == "LockInCam" and nm.Scan3dExtractionMethod.value == "rawIQ"


def test_reference_rounding_to_80mhz_clock(small_cfg):
    cam = open_camera(small_cfg.replace(ref_freq_hz=1234.567), sim=True)
    a = cam.configure()
    assert a["ref_freq_hz"] == pytest.approx(80e6 / round(80e6 / 1234.567))
    assert a["ref_freq_hz"] != 1234.567
    cam.close()


def test_recording_active_routing_needs_output_mode(small_cfg):
    cam = open_camera(small_cfg.replace(recording_active_line="DIO1"), sim=True)
    cam.configure()                   # route() sets LineMode=Output first
    assert cam.nm.selected("LineMode", "DIO1") == "Output"
    assert cam.nm.selected("LineSource", "DIO1") == "RecordingActive"
    cam.nm.LineSelector.value = "DIO2"
    with pytest.raises(AccessException):
        cam.nm.LineSource.value = "RecordingActive"     # still an input
    cam.close()


def test_illegal_settings_rejected(sim_cam):
    nm = sim_cam.nm
    with pytest.raises(ValueError):
        nm.AcquisitionBurstFrameCount.value = 101       # must be even
    with pytest.raises(ValueError):
        nm.AcquisitionBurstFrameCount.value = 902
    with pytest.raises(ValueError):
        nm.LockInCoupling.value = "ACDC"
    with pytest.raises(AttributeError):
        nm.LockInTimeConstant                           # not a feature
    sim_cam.start()
    with pytest.raises(AccessException):
        nm.LockInSensitivity.value = 0.2                # configuration mode only
    sim_cam.stop()
    with pytest.raises(RuntimeError):
        sim_cam.running = True
        sim_cam.configure()
    sim_cam.running = False


def test_config_validation_blocks_configure(small_cfg):
    cam = open_camera(small_cfg.replace(burst_frames=7), sim=True)
    with pytest.raises(ValueError, match="burst_frames"):
        cam.configure()
    cam.close()


def test_fetch_burst_shape_decode_and_chunks(sim_cam, small_cfg):
    sim_cam.ia.sim.controller.mode = "run"
    sim_cam.start()
    bursts = [sim_cam.fetch_burst() for _ in range(3)]
    sim_cam.stop()
    iq, m0 = bursts[0]
    assert iq.shape == (2, 40, 96, 96) and iq.dtype == np.uint16
    assert iq.min() >= 0 and iq.max() <= 1023
    assert 400 < np.median(iq) < 800            # centred near 512 plus signal
    m1, m2 = bursts[1][1], bursts[2][1]
    assert (m0.frame_id, m1.frame_id, m2.frame_id) == (0, 40, 80)
    gap = (m1.t0_ns - m0.t0_ns) * 1e-9 - 40 * 0.022
    assert gap == pytest.approx(small_cfg.sim_gap_s, abs=1e-6)
    t = m1.frame_times_s()
    assert np.allclose(np.diff(t), 0.022)


def test_no_pipelining_gives_transfer_gap(small_cfg):
    cam = open_camera(small_cfg.replace(queue_blocks=1), sim=True)
    cam.configure()
    cam.start()
    (_, a), (_, b) = cam.fetch_burst(), cam.fetch_burst()
    cam.stop()
    gap = (b.t0_ns - a.t0_ns) * 1e-9 - 40 * 0.022
    from c4ppg.feasibility import transfer_time_s
    assert gap == pytest.approx(transfer_time_s(40, 96, 96), rel=1e-6)
    cam.close()


def test_decode_matches_heliotis_examples():
    raw = np.array([0x8000 | (512 * 4), 0x8000 | (1023 * 4) | 3, 4 * 7], np.uint16)
    assert decode(raw).tolist() == [512, 1023, 7]


def test_real_camera_requires_cti(monkeypatch):
    monkeypatch.setenv("DIAPHUS_GENTL64_FILE", r"Z:\nope\diaphus.cti")
    from c4ppg.camera import find_cti
    with pytest.raises(FileNotFoundError):
        find_cti()


def test_config_defaults_valid():
    c = Config()
    assert c.validate() == []
    assert c.frame_period_s == pytest.approx(0.022)
    assert c.burst_duration_s == pytest.approx(2.2)
