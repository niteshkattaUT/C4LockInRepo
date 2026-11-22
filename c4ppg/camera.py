"""heliCam C4 wrapper on top of Harvesters (GenTL producer = Diaphus.cti).

Works identically against the real camera and ``sim.FakeHarvester``.
Configuration is only allowed while stopped (configuration mode); ``configure()``
asserts that and returns the read-back actual values.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config

DEFAULT_CTI = r"C:\Program Files\C4Utility\diaphus\win64-x64\diaphus.cti"


def find_cti() -> str:
    """DIAPHUS_GENTL64_FILE is set by the C4Utility installer; fall back to the default install path."""
    path = os.environ.get("DIAPHUS_GENTL64_FILE") or DEFAULT_CTI
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GenTL producer not found: {path} (install C4Utility or set DIAPHUS_GENTL64_FILE)")
    return path


def decode(data: np.ndarray) -> np.ndarray:
    """Raw sensor word -> 10-bit value centred near 512 (same as the Heliotis examples)."""
    return (data % 2 ** 15 // 4).astype(np.uint16)


@dataclass
class BurstMeta:
    index: int                      # burst counter since start()
    n_frames: int
    t_host: float                   # time.time() when the burst arrived
    t0_ns: Optional[int]            # ChunkTimestamp: FrameStart of the first frame (device clock)
    frame_id: Optional[int]         # ChunkFrameID of the first frame
    frame_period_s: float           # from the actual reference frequency and periods
    fetch_s: float                  # time spent blocked in fetch()
    frame_t_ns: Optional[list] = None   # per-frame timestamps if the transport exposes them
    flags: list = field(default_factory=list)

    def frame_times_s(self) -> np.ndarray:
        """Per-frame FrameStart times in seconds (device clock if available, else host clock)."""
        if self.frame_t_ns is not None:
            return np.asarray(self.frame_t_ns, float) * 1e-9
        t0 = self.t0_ns * 1e-9 if self.t0_ns is not None else self.t_host - self.n_frames * self.frame_period_s
        return t0 + np.arange(self.n_frames) * self.frame_period_s

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class C4Camera:
    """Thin wrapper around a Harvesters ImageAcquirer."""

    def __init__(self, cfg: Config, harvester=None, device_index: int = 0):
        self.cfg = cfg
        if harvester is None:
            from harvesters.core import Harvester
            harvester = Harvester()
            harvester.add_file(find_cti())
        self.h = harvester
        self.h.update()
        if not self.h.device_info_list:
            raise RuntimeError("No heliCam found - check the GigE link (192.168.2.x), firewall and power")
        self.device_info = self.h.device_info_list[device_index]
        self.ia = self.h.create(device_index)
        self.nm = self.ia.remote_device.node_map
        self.actual: dict = {}
        self.running = False
        self._burst_index = 0

    @property
    def is_sim(self) -> bool:
        return hasattr(self.ia, "sim")

    # --- configuration (only allowed while stopped) ----------------------------------------
    def configure(self) -> dict:
        problems = self.cfg.validate()
        if problems:
            raise ValueError("Invalid config: " + "; ".join(problems))
        if self.running:
            raise RuntimeError("configure() needs configuration mode - call stop() first")
        nm, c = self.nm, self.cfg

        nm.DeviceOperationMode.value = "LockInCam"
        nm.Scan3dExtractionMethod.value = "rawIQ"

        # freerunning: all start triggers off (as in c4IntensityPipelinedFreerunning.py)
        for trig in ("RecordingStart", "FrameStart", "AcquisitionStart"):
            nm.TriggerSelector.value = trig
            nm.TriggerMode.value = "Off"

        nm.LockInReferenceSourceType.value = c.reference_source
        if c.reference_source == "External":
            nm.LockInReferenceSourceSignal.value = c.reference_input
            nm.LockInReferenceFrequencyScaler.value = "Off"
            nm.LockInExpectedFrequencyDeviation.value = int(c.expected_freq_dev_pct)
        nm.LockInTargetReferenceFrequency.value = float(c.ref_freq_hz)
        nm.LockInTargetTimeConstantNPeriods.value = int(c.n_periods)
        nm.LockInTargetBlankDurationNPeriods.value = int(c.n_blank)
        nm.LockInSensitivity.value = float(c.sensitivity)
        nm.LockInCoupling.value = c.coupling
        nm.AcquisitionBurstFrameCount.value = int(c.burst_frames)
        nm.TransferSelector.value = "All"
        nm.TransferQueueMaxBlockCount.value = int(c.queue_blocks)

        # real-time outputs to the LED controller
        self.route(c.frame_sync_line, "LockInFrameSync")
        self.route(c.reference_line, "LockInReference")
        if c.recording_active_line:
            self.route(c.recording_active_line, "RecordingActive")

        # chunk data: timestamp + frame ID per frame
        nm.ChunkModeActive.value = True
        for chunk in ("Timestamp", "FrameID"):
            nm.ChunkSelector.value = chunk
            nm.ChunkEnable.value = True

        # the heliCam's own LED module (LightController0) is not used in the two-LED setup
        self.set_internal_ldm(False)

        self.actual = self.read_actuals()
        return self.actual

    def route(self, line: str, source: str):
        """Put an internal signal on a D3 line.  ftrDesc 6.7: LineSource needs LineMode = Output."""
        nm = self.nm
        nm.LineSelector.value = line
        try:
            if nm.LineMode.value != "Output":
                nm.LineMode.value = "Output"
        except Exception as e:  # read-only on fixed output lines such as RTIO2/RTIO3
            if nm.LineMode.value != "Output":
                raise RuntimeError(f"{line} cannot be an output: {e}") from e
        nm.LineSource.value = source

    def set_internal_ldm(self, on: bool, freq_hz: float = 0.0, amplitude_pct: float = 10.0,
                         offset_pct: float = 20.0, modulate: bool = True):
        """Internal LED module via the signal generator (beat test, build step 2)."""
        nm = self.nm
        try:
            nm.LightControllerSelector.value = "LightController0"
            if on:
                nm.SignalGeneratorOffset.value = float(offset_pct)
                nm.SignalGeneratorAmplitude.value = float(amplitude_pct)
                nm.SignalGeneratorModulationMode.value = "On" if modulate else "Off"
                nm.SignalGeneratorFrequency.value = float(freq_hz)
                nm.LightControllerSource.value = "SignalGenerator"
            else:
                nm.LightControllerSource.value = "Off"   # VERIFY: harmless without the LDM attached
        except Exception as e:
            if on:
                raise
            print(f"note: could not switch LightController0 off ({e})")

    def read_actuals(self) -> dict:
        nm = self.nm

        def get(name, default=None):
            try:
                return getattr(nm, name).value
            except Exception:
                return default

        temps = {}
        for loc in ("Mainboard", "Processor", "Optic"):
            try:
                nm.DeviceTemperatureSelector.value = loc
                temps[loc] = nm.DeviceTemperature.value
            except Exception:
                pass
        return dict(
            ref_freq_hz=get("LockInActualReferenceFrequency"),
            n_periods=get("LockInActualTimeConstantNPeriods"),
            n_blank=get("LockInActualBlankDurationNPeriods"),
            frame_rate=get("FrameRate"),
            sensitivity=get("LockInSensitivity"),
            coupling=get("LockInCoupling"),
            burst_frames=get("AcquisitionBurstFrameCount"),
            queue_blocks=get("TransferQueueMaxBlockCount"),
            reference_source=get("LockInReferenceSourceType"),
            firmware=get("DeviceFirmwareVersion"),
            model=get("DeviceModelName"),
            serial=get("DeviceSerialNumber"),
            sensor=get("SensorName"),
            height=get("Height"),
            width=get("Width"),
            temperatures_c=temps,
        )

    @property
    def frame_period_s(self) -> float:
        a = self.actual
        if a.get("ref_freq_hz") and a.get("n_periods") is not None:
            return (a["n_periods"] + a["n_blank"]) / a["ref_freq_hz"]
        return self.cfg.frame_period_s

    # --- acquisition -----------------------------------------------------------------------
    def start(self):
        if not self.actual:
            raise RuntimeError("configure() first")
        self.ia.start()
        self.running = True
        self._burst_index = 0

    def stop(self):
        if self.running:
            self.ia.stop()
        self.running = False

    def fetch_burst(self, timeout: Optional[float] = None):
        """Block until the next burst; return (iq, meta) with iq uint16 (2, N, H, W): [I, Q]."""
        n = int(self.actual["burst_frames"])
        h, w = int(self.actual["height"]), int(self.actual["width"])
        t_call = time.perf_counter()
        with self.ia.fetch(timeout=timeout or self.cfg.fetch_timeout_s) as buf:
            comps = buf.payload.components
            if len(comps) != 2 * n:
                raise RuntimeError(f"expected {2 * n} components (I and Q per frame), got {len(comps)}")
            # copy out of the GenTL buffer before it is re-queued
            iq = decode(np.stack([np.asarray(c.data).reshape(h, w) for c in comps])).reshape(2, n, h, w)
            meta = BurstMeta(
                index=self._burst_index, n_frames=n, t_host=time.time(),
                # VERIFY: with multi-part bursts Harvesters may expose ChunkTimestamp only for the
                # buffer (= first frame).  Per-frame times are then derived from frame_period_s.
                t0_ns=self._safe(lambda: int(self.nm.ChunkTimestamp.value)),
                frame_id=self._safe(lambda: int(self.nm.ChunkFrameID.value)),
                frame_period_s=self.frame_period_s,
                fetch_s=time.perf_counter() - t_call,
            )
        self._burst_index += 1
        return iq, meta

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    def close(self):
        try:
            self.stop()
        finally:
            try:
                self.ia.destroy()
            finally:
                self.h.reset()


def open_camera(cfg: Config, sim: bool = False) -> C4Camera:
    if sim:
        from .sim import FakeHarvester
        return C4Camera(cfg, FakeHarvester(cfg))
    return C4Camera(cfg)
