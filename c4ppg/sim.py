"""Simulated heliCam C4 behind the same Harvesters surface the real camera uses.

``FakeHarvester`` -> ``FakeImageAcquirer`` (``remote_device.node_map``, ``start``,
``stop``, ``fetch``, ``destroy``) so ``C4Camera`` runs unchanged with or without
hardware.  The node map only accepts the feature names and enum values listed in
ftrDesc v1.12.0, applies the documented ranges, refuses configuration writes
during acquisition and rounds targets to the 80 MHz clock, so a typo or an illegal
setting fails here before it fails on the real camera.

The scene is a fundus-like phantom: vignetted disc, an optic-disc reflex and a
vessel tree.  Each LED's lock-in amplitude follows

    A = gain * sensitivity * P_led * reflectance * (1 - depth * pulse(t))

with a cardiac pulse (systolic peak + dicrotic notch) and a weaker respiratory
drift, so the expected ratio-of-ratios in a region is depth_760 / depth_850.
Room light only raises the light-dependent offset and the noise, as in the real
lock-in.  A, B alternation comes from ``LedControllerModel`` run edge by edge.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

from .config import RAW_MID, SENSOR_SHAPE, Config
from .controller import LedControllerModel
from .feasibility import transfer_time_s

CLOCK_HZ = 80e6


class AccessException(RuntimeError):
    """Raised like genicam's AccessException when a feature is not writable now."""


class FakeTimeoutException(TimeoutError):
    pass


# ---------------------------------------------------------------------------------------
# Node map
# ---------------------------------------------------------------------------------------
TRIGGERS = ["AcquisitionStart", "FrameStart", "FrameBurstStart", "RecordingStart", "Reference"]
LINES = (["Line0", "Line1", "Line2", "Line3", "RTIO2", "RTIO3", "RTIO4"]
         + [f"DIO{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17)]
         + ["RTIO6", "RTIO7", "FI2", "FI3", "SwCtrlVdd5"])
OUTPUT_ONLY = {"RTIO2", "RTIO3", "Line2", "Line3"}
INPUT_ONLY = {"RTIO4", "RTIO6", "RTIO7", "FI2", "FI3", "Line0", "Line1"}
LINE_SOURCES = (["Off"] + [f"UserOutput{i}" for i in range(8)]
                + ["FrameTriggerWait", "FrameTriggerMissed", "FrameActive", "RecordingTriggerWait",
                   "RecordingTriggerMissed", "RecordingActive", "StreamTransferActive", "StageInMotion",
                   "LockInReference", "LockInFrameSync"])
CHUNKS = ["PartCount", "PartFixpointScaling", "PartFrameNo", "PartType", "ZTagCount", "ZTagValue",
          "Scan3dDistanceUnit", "Scan3dOutputMode", "Scan3dCoordinateScale", "Scan3dCoordinateOffset",
          "ProcessingString", "ConfigString", "Timestamp", "FrameID"]
LIGHT_SOURCES = (["Off"] + [f"UserOutput{i}" for i in range(8)]
                 + [f"DIO{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17)]
                 + ["SFIO5", "SignalGenerator"] + [f"FieldbusBool{i}" for i in range(8)])


class _Node:
    def __init__(self, nm: "FakeNodeMap", name: str):
        self._nm, self._name = nm, name

    @property
    def value(self):
        return self._nm._get(self._name)

    @value.setter
    def value(self, v):
        self._nm._set(self._name, v)

    def execute(self):
        self._nm._execute(self._name)

    def __repr__(self):
        return f"<Node {self._name}={self.value!r}>"


class FakeNodeMap:
    """Spec-checked subset of the gen4 feature set used by this project."""

    # name: (kind, spec, writable_in_acquisition)
    RW = {
        "DeviceOperationMode": ("enum", ["Topography", "Tomography", "LockInCam"], False),
        "Scan3dExtractionMethod": ("enum", ["Default", "rawIQ", "rawAmp", "rawExtSimpMax", "centerOfMass"], False),
        "TriggerSelector": ("enum", TRIGGERS, True),
        "TriggerMode": ("enum", ["Off", "On"], False),
        "TriggerSource": ("enum", ["Software", "FI2", "FI3", "RTIO6", "RTIO7"], False),
        "LockInReferenceSourceType": ("enum", ["Internal", "External"], False),
        "LockInReferenceSourceSignal": ("enum", ["Other", "FI2", "FI3", "RTIO6", "RTIO7"], False),
        "LockInReferenceFrequencyScaler": ("enum", ["Off", "MultiplyBy4", "MultiplyBy2", "DivideBy2", "DivideBy4",
                                                    "DivideBy8", "DivideBy16", "DivideBy32", "DivideBy64"], False),
        "LockInTargetReferenceFrequency": ("float", (306.0, 250000.0), False),
        "LockInExpectedFrequencyDeviation": ("int", (0, 100, 1), False),
        "LockInTargetTimeConstantNPeriods": ("int", (1, 100, 1), False),
        "LockInTargetBlankDurationNPeriods": ("int", (0, 100, 1), False),
        "LockInSensitivity": ("float", (0.0, 1.0), False),
        "LockInCoupling": ("enum", ["DC", "AC"], False),
        "AcquisitionBurstFrameCount": ("int", (4, 900, 2), False),
        "TransferSelector": ("enum", ["All"], True),
        "TransferQueueMaxBlockCount": ("int", (1, 127, 1), False),
        "LineSelector": ("enum", LINES, True),
        "LineMode": ("enum", ["Input", "Output"], True),
        "LineSource": ("enum", LINE_SOURCES, False),
        "LineInverter": ("bool", None, False),
        "ChunkModeActive": ("bool", None, False),
        "ChunkSelector": ("enum", CHUNKS, True),
        "ChunkEnable": ("bool", None, False),
        "ChunkPartSelector": ("int", (0, 10_000, 1), True),
        "LightControllerSelector": ("enum", ["LightController0", "LED_A", "LED_B", "LED_C"], True),
        "LightControllerSource": ("enum", LIGHT_SOURCES, False),
        "SignalGeneratorFrequency": ("float", (100.0, 50000.0), False),
        "SignalGeneratorAmplitude": ("float", (1.0, 100.0), False),
        "SignalGeneratorOffset": ("float", (1.0, 100.0), False),
        "SignalGeneratorModulationMode": ("enum", ["Off", "On", "Pulsed"], False),
        "DeviceTemperatureSelector": ("enum", ["Mainboard", "Processor", "Optic"], True),
        "ReverseX": ("bool", None, False),
        "ReverseY": ("bool", None, False),
    }
    SELECTED = {"TriggerMode": "TriggerSelector", "TriggerSource": "TriggerSelector",
                "LineMode": "LineSelector", "LineSource": "LineSelector", "LineInverter": "LineSelector",
                "ChunkEnable": "ChunkSelector", "LightControllerSource": "LightControllerSelector",
                "TransferQueueMaxBlockCount": "TransferSelector", "DeviceTemperature": "DeviceTemperatureSelector"}
    RO = ("LockInActualReferenceFrequency", "LockInActualTimeConstantNPeriods", "LockInActualBlankDurationNPeriods",
          "FrameRate", "DeviceFirmwareVersion", "DeviceModelName", "DeviceSerialNumber", "SensorName", "Height",
          "Width", "DeviceTemperature", "ChunkTimestamp", "ChunkFrameID", "ChunkPartCount", "ChunkPartFrameNo",
          "Timestamp", "PayloadSize", "LineStatus")

    def __init__(self, cam: "SimCamera"):
        object.__setattr__(self, "_cam", cam)
        object.__setattr__(self, "_vals", {
            "DeviceOperationMode": "Topography", "Scan3dExtractionMethod": "Default",
            "TriggerSelector": "FrameStart", "LockInReferenceSourceType": "Internal",
            "LockInReferenceSourceSignal": "FI2", "LockInReferenceFrequencyScaler": "Off",
            "LockInTargetReferenceFrequency": 10000.0, "LockInExpectedFrequencyDeviation": 5,
            "LockInTargetTimeConstantNPeriods": 10, "LockInTargetBlankDurationNPeriods": 0,
            "LockInSensitivity": 0.5, "LockInCoupling": "DC", "AcquisitionBurstFrameCount": 60,
            "TransferSelector": "All", "LineSelector": "RTIO2", "ChunkModeActive": False,
            "ChunkSelector": "Timestamp", "ChunkPartSelector": 0,
            "LightControllerSelector": "LightController0", "SignalGeneratorFrequency": 10000.0,
            "SignalGeneratorAmplitude": 10.0, "SignalGeneratorOffset": 20.0,
            "SignalGeneratorModulationMode": "On", "DeviceTemperatureSelector": "Mainboard",
            "ReverseX": False, "ReverseY": False,
        })
        object.__setattr__(self, "_sel", {
            "TriggerMode": {t: "Off" for t in TRIGGERS} | {"FrameStart": "On"},
            "TriggerSource": {t: "Software" for t in TRIGGERS},
            "LineMode": {ln: ("Output" if ln in OUTPUT_ONLY else "Input") for ln in LINES},
            "LineSource": {ln: "Off" for ln in LINES},
            "LineInverter": {ln: False for ln in LINES},
            "ChunkEnable": {c: False for c in CHUNKS},
            "LightControllerSource": {c: "Off" for c in ["LightController0", "LED_A", "LED_B", "LED_C"]},
            "TransferQueueMaxBlockCount": {"All": 1},
        })
        object.__setattr__(self, "_writes", [])

    # attribute access == node access, like harvesters' node_map
    def __getattr__(self, name):
        if name in self.RW or name in self.RO or name in ("TriggerSoftware",):
            return _Node(self, name)
        raise AttributeError(f"heliCam node map has no feature {name!r} (check ftrDesc)")

    def __setattr__(self, name, value):
        raise AttributeError("use node_map.<Feature>.value = ...")

    def __dir__(self):
        return list(self.RW) + list(self.RO)

    # ------------------------------------------------------------------------------------
    def _get(self, name):
        cam = self._cam
        if name in self.SELECTED:
            sel = self._vals[self.SELECTED[name]] if name != "DeviceTemperature" else None
            if name == "DeviceTemperature":
                return {"Mainboard": 41.5, "Processor": 55.0, "Optic": 35.2}[self._vals["DeviceTemperatureSelector"]]
            return self._sel[name][sel]
        if name in self._vals:
            return self._vals[name]
        a = cam.actual()
        ro = {
            "LockInActualReferenceFrequency": a["ref_freq_hz"],
            "LockInActualTimeConstantNPeriods": a["n_periods"],
            "LockInActualBlankDurationNPeriods": a["n_blank"],
            "FrameRate": a["frame_rate"],
            "DeviceFirmwareVersion": "1.12.0-sim", "DeviceModelName": "heliCam C4.2-S40U (simulated)",
            "DeviceSerialNumber": "SIM0001", "SensorName": "S40U-sim",
            "Height": cam.shape[0], "Width": cam.shape[1],
            "ChunkTimestamp": cam.chunk.get("Timestamp") if self._vals["ChunkModeActive"] else None,
            "ChunkFrameID": cam.chunk.get("FrameID") if self._vals["ChunkModeActive"] else None,
            "ChunkPartCount": cam.chunk.get("PartCount"),
            "ChunkPartFrameNo": self._vals["ChunkPartSelector"] % max(1, self._vals["AcquisitionBurstFrameCount"]),
            "Timestamp": cam.now_ns(), "PayloadSize": 2 * self._vals["AcquisitionBurstFrameCount"] * cam.shape[0]
            * cam.shape[1] * 2, "LineStatus": False,
        }
        if name in ("ChunkTimestamp", "ChunkFrameID") and ro[name] is None:
            raise AccessException(f"{name} not available (ChunkModeActive is False or no buffer fetched)")
        return ro[name]

    def _set(self, name, v):
        if name in self.RO:
            raise AccessException(f"{name} is read-only")
        kind, spec, in_acq = self.RW[name]
        if self._cam.acquiring and not in_acq:
            raise AccessException(f"{name} is not writable in acquisition mode (call stop() first)")
        if kind == "enum":
            if v not in spec:
                raise ValueError(f"{name}: {v!r} is not one of {spec}")
        elif kind == "int":
            lo, hi, inc = spec
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
                raise TypeError(f"{name} expects an int, got {v!r}")
            if not lo <= v <= hi or (v - (lo % inc)) % inc:
                raise ValueError(f"{name}: {v} outside {lo}..{hi} step {inc}")
            v = int(v)
        elif kind == "float":
            lo, hi = spec
            v = float(v)
            if not lo <= v <= hi:
                raise ValueError(f"{name}: {v} outside {lo}..{hi}")
        elif kind == "bool":
            v = bool(v)
        if name in self.SELECTED:
            sel = self._vals[self.SELECTED[name]]
            if name == "LineMode" and (sel in OUTPUT_ONLY and v != "Output" or sel in INPUT_ONLY and v != "Input"):
                raise AccessException(f"LineMode of {sel} is fixed")
            if name == "LineSource" and self._sel["LineMode"][sel] != "Output":
                raise AccessException(f"LineSource[{sel}] needs LineMode = Output first")
            self._sel[name][sel] = v
            self._writes.append((f"{name}[{sel}]", v))
        else:
            self._vals[name] = v
            self._writes.append((name, v))

    def _execute(self, name):
        if name == "TriggerSoftware":
            self._cam.software_trigger.set()
        else:
            raise AccessException(f"{name} is not a command")

    # helpers for tests / the sim
    def selected(self, name: str, selector_value: str):
        return self._sel[name][selector_value]


# ---------------------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------------------
class Scene:
    """Static maps + time courses of the fundus phantom."""

    def __init__(self, cfg: Config, shape):
        self.cfg = cfg
        H, W = shape
        rng = np.random.default_rng(cfg.sim_seed)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        cy, cx = H / 2, W / 2
        r = np.hypot((y - cy) / (H / 2), (x - cx) / (W / 2))
        disc = np.clip(1.15 - r ** 2, 0, None)                  # vignetted fundus
        reflex = 0.9 * np.exp(-(((y - 0.45 * H) / (0.06 * H)) ** 2 + ((x - 0.62 * W) / (0.06 * W)) ** 2))

        vessel = np.zeros((H, W), np.float32)
        for k, (a, b, c, wdt) in enumerate([(0.50, 0.0, 0.0, 0.035), (0.30, 0.25, 0.6, 0.025),
                                           (0.70, -0.3, -0.5, 0.025), (0.50, 0.6, 0.3, 0.02)]):
            # gently curved vessels running left to right
            yc = (a + b * ((x / W) - 0.5) + 0.1 * c * np.sin(3 * x / W)) * H
            vessel = np.maximum(vessel, np.exp(-((y - yc) / (wdt * H)) ** 2))
        self.vessel = vessel
        self.fpn_i = rng.normal(0, 6, (H, W)).astype(np.float32)
        self.fpn_q = rng.normal(0, 6, (H, W)).astype(np.float32)
        # reflectance per LED (760 more absorbed by blood than 850)
        self.refl = {"A": disc * (1 - 0.45 * vessel) + reflex, "B": disc * (1 - 0.30 * vessel) + 1.2 * reflex}
        self.led_power = {"A": 1.0, "B": 1.25}
        dv, dt = cfg.sim_ppg_depth["vessel"], cfg.sim_ppg_depth["tissue"]
        self.depth = {"A": dt[0] + (dv[0] - dt[0]) * vessel, "B": dt[1] + (dv[1] - dt[1]) * vessel}
        self.gain = 230.0 / 0.15          # counts per unit reflectance at sensitivity 0.15

    def pulse(self, t: np.ndarray) -> np.ndarray:
        """Cardiac volume waveform (0..~1): systolic peak + dicrotic wave, plus slow respiration."""
        f = self.cfg.sim_heart_rate_bpm / 60.0
        ph = (t * f) % 1.0
        sys_ = np.exp(-((ph - 0.15) / 0.07) ** 2)
        dic = 0.35 * np.exp(-((ph - 0.45) / 0.09) ** 2)
        resp = 0.15 * np.sin(2 * np.pi * 0.25 * t)
        return sys_ + dic + resp


# ---------------------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------------------
class _Component:
    def __init__(self, data):
        self.data = data
        self.height, self.width = data.shape


class _Payload:
    def __init__(self, comps):
        self.components = comps


class _Buffer:
    def __init__(self, comps):
        self.payload = _Payload(comps)


class SimCamera:
    """Physics and timing of the simulated heliCam (owned by the FakeImageAcquirer)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shape = tuple(cfg.sim_shape) if cfg.sim_shape else SENSOR_SHAPE
        self.nm = FakeNodeMap(self)
        self.controller = LedControllerModel(cfg.switch_after_periods)
        self.scene = Scene(cfg, self.shape)
        self.rng = np.random.default_rng(cfg.sim_seed + 1)
        self.acquiring = False
        self.software_trigger = threading.Event()
        self.chunk: dict = {}
        self._t_origin = time.perf_counter()
        self._frame_id = 0
        self._cache: dict = {}
        self._reset_clock()

    # --- clock -------------------------------------------------------------------------
    def _reset_clock(self):
        self._burst_k = 0
        self._prev_end = None
        self._fetch_times: list[float] = []
        self._start_wall = None

    def now_ns(self) -> int:
        return int((time.perf_counter() - self._t_origin) * 1e9)

    def actual(self) -> dict:
        v = self.nm._vals
        f = CLOCK_HZ / round(CLOCK_HZ / v["LockInTargetReferenceFrequency"])   # 80 MHz clock rounding
        n, b = v["LockInTargetTimeConstantNPeriods"], v["LockInTargetBlankDurationNPeriods"]
        return {"ref_freq_hz": f, "n_periods": n, "n_blank": b, "frame_rate": f / (n + b)}

    # --- acquisition -------------------------------------------------------------------
    def start(self):
        if self.nm._vals["DeviceOperationMode"] != "LockInCam":
            raise AccessException("DeviceOperationMode must be LockInCam")
        self.acquiring = True
        self._reset_clock()
        self._start_wall = time.perf_counter()

    def stop(self):
        self.acquiring = False

    def _scaled(self, seconds: float) -> float:
        return seconds * self.cfg.sim_time_scale

    def next_burst(self, timeout: float):
        if not self.acquiring:
            raise AccessException("fetch() before start()")
        scale = self.cfg.sim_time_scale
        t_call = (time.perf_counter() - self._start_wall) / scale if scale else 0.0   # camera time of the request
        nm = self.nm._vals
        a = self.actual()
        N = nm["AcquisitionBurstFrameCount"]
        T = 1.0 / a["frame_rate"]
        dur = N * T
        Q = self.nm._sel["TransferQueueMaxBlockCount"]["All"]
        k = self._burst_k
        # camera-time start of burst k: after the previous burst plus dead time, and only
        # once a transfer block is free (block k-Q has been fetched by the host).
        tt = transfer_time_s(N, self.shape[1], self.shape[0])    # Heliotis calculator link model
        if self._prev_end is None:
            start = 0.0
        else:
            start = self._prev_end + (self.cfg.sim_gap_s if Q > 1 else tt)
            if Q > 1 and k - Q >= 0:
                start = max(start, self._fetch_times[k - Q])
        end = start + dur
        # block the caller until the burst is complete (+ transfer) in scaled wall time
        ready = end + (tt if Q > 1 else 0.0)
        data = self._render(start, N, T, a)          # render first: costs host time, not camera time
        wait = self._scaled(ready) - (time.perf_counter() - self._start_wall)
        if wait > timeout:
            time.sleep(max(timeout, 0))
            raise FakeTimeoutException("fetch timed out (simulated)")
        if wait > 0:
            time.sleep(wait)
        # the host frees block k when it asks for it (or when it is ready, if it asked early);
        # the simulator's own rendering time is not host lateness
        self._fetch_times.append(max(t_call, ready))
        self._prev_end = end
        self._burst_k += 1
        self.chunk = {"Timestamp": int(start * 1e9), "FrameID": self._frame_id, "PartCount": 2 * N}
        self._frame_id += N
        return data

    # --- image formation ---------------------------------------------------------------
    def _render(self, t0: float, N: int, T: float, a: dict) -> list:
        nm, sc, cfg = self.nm._vals, self.scene, self.cfg
        H, W = self.shape
        sens = nm["LockInSensitivity"]
        ac = nm["LockInCoupling"] == "AC"
        internal_ldm = self.nm._sel["LightControllerSource"]["LightController0"] == "SignalGenerator"
        n_per, n_blank = a["n_periods"], a["n_blank"]
        t_mid = t0 + (np.arange(N) + 0.5 * n_per / (n_per + n_blank)) * T

        if not internal_ldm:     # RecordingActive rising edge at the burst start
            if cfg.sim_reset_miss_prob and self.rng.random() < cfg.sim_reset_miss_prob:
                self.controller.miss_next_reset = True
            self.controller.on_recording_active()
        IQ = np.empty((2, N, H, W), np.float32)
        pulse = sc.pulse(t_mid)
        for j in range(N):
            if internal_ldm:
                led, mod = "LDM", nm["SignalGeneratorModulationMode"] == "On"
            else:
                if cfg.sim_sync_miss_prob and self.rng.random() < cfg.sim_sync_miss_prob:
                    self.controller.miss_next_sync = True
                led, mod = self.controller.simulate_frame(n_per, n_blank)
            st = self._static(led, mod, sens, ac)
            if led == "LDM":
                phase = 0.3 + 2 * np.pi * (nm["SignalGeneratorFrequency"] - a["ref_freq_hz"]) * t_mid[j]
            elif led is None:
                phase = 0.0
            else:
                phase = cfg.sim_driver_phase_rad[0 if led == "A" else 1]
            # amp = base * (1 - depth * pulse) = base - base_depth * pulse
            for c, trig in ((0, np.cos(phase)), (1, np.sin(phase))):
                out = IQ[c, j]
                self.rng.standard_normal((H, W), dtype=np.float32, out=out)
                out *= st["noise"]
                out += st["off"][c]
                if st["base"] is not None:
                    out += st["base"] * np.float32(trig)
                    if st["base_depth"] is not None:
                        out -= st["base_depth"] * np.float32(trig * pulse[j])
                if st["flat"] is not None:   # DC saturation: flat output at mid scale
                    out[st["flat"]] = RAW_MID + 0.3 * self.rng.standard_normal(st["n_flat"], dtype=np.float32)
        IQ = IQ.reshape(2 * N, H, W)
        np.rint(IQ, out=IQ)
        np.clip(IQ, 0, 1023, out=IQ)
        # encode like the sensor stream: value * 4 with the top (flag) bit set; decode = %2**15 // 4
        enc = IQ.astype(np.uint16)
        enc <<= 2
        enc |= np.uint16(0x8000)
        return [_Component(enc[i]) for i in range(enc.shape[0])]

    def _static(self, led, mod: bool, sens: float, ac: bool) -> dict:
        """Per-LED maps that do not change from frame to frame (cached)."""
        nm, sc, cfg = self.nm._vals, self.scene, self.cfg
        key = (led, mod, sens, ac, nm["SignalGeneratorAmplitude"], nm["SignalGeneratorOffset"])
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        H, W = self.shape
        base = base_depth = None
        if led == "LDM":
            dc_light = np.full((H, W), nm["SignalGeneratorOffset"] / 20.0, np.float32)
            if mod:
                base = np.full((H, W), sc.gain * sens * nm["SignalGeneratorAmplitude"] / 10.0 * 0.8, np.float32)
        elif led is None:
            dc_light = np.zeros((H, W), np.float32)
        else:
            refl = sc.refl[led] * sc.led_power[led]
            dc_light = 0.5 * refl                                   # mean of a 0/full square wave
            if mod:
                base = (sc.gain * sens * refl).astype(np.float32)
                base_depth = (base * sc.depth[led]).astype(np.float32)
        light = dc_light + cfg.sim_room_light * sc.refl["A"] * 0.5   # room light: offset + noise only
        k = 0.0 if ac else sens / 0.15
        off = (RAW_MID + sc.fpn_i + 18.0 * k * light, RAW_MID + sc.fpn_q + 12.0 * k * light)  # light-dependent
        flat = (light * sens / 0.15 > 2.0) if not ac else None
        if flat is not None and not flat.any():
            flat = None
        st = dict(base=base, base_depth=base_depth, off=[o.astype(np.float32) for o in off],
                  noise=(cfg.sim_noise_counts * np.sqrt(1 + light)).astype(np.float32),
                  flat=flat, n_flat=int(flat.sum()) if flat is not None else 0)
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[key] = st
        return st


class _RemoteDevice:
    def __init__(self, nm):
        self.node_map = nm


class FakeImageAcquirer:
    def __init__(self, cfg: Config):
        self.sim = SimCamera(cfg)
        self.remote_device = _RemoteDevice(self.sim.nm)
        self.destroyed = False

    def start(self):
        self.sim.start()

    def stop(self):
        self.sim.stop()

    @contextmanager
    def fetch(self, timeout: float = 10.0):
        comps = self.sim.next_burst(timeout)
        yield _Buffer(comps)

    def destroy(self):
        self.sim.stop()
        self.destroyed = True


class _DevInfo:
    def __init__(self):
        self.id_ = "heliCamC4-sim"
        self.serial_number = "SIM0001"
        self.model = "heliCam C4.2-S40U (simulated)"


class FakeHarvester:
    """Drop-in for ``harvesters.core.Harvester`` with one simulated camera."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.files: list[str] = []
        self.device_info_list: list = []
        self.acquirer: Optional[FakeImageAcquirer] = None

    def add_file(self, path):
        self.files.append(path)

    def update(self):
        self.device_info_list = [_DevInfo()]

    def create(self, index=0):
        self.acquirer = FakeImageAcquirer(self.cfg)
        return self.acquirer

    def reset(self):
        pass
