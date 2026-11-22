"""Every tunable value in one place.

The dataclass defaults are the start values from the design doc.  Override them
without editing code with a YAML file (``--config my.yaml``) and/or
``--set key=value`` on the command line.  ``python -m c4ppg config`` prints the
full current configuration as YAML.
"""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# Sensor limits from the manual / ftrDesc v1.12.0
SENSOR_SHAPE = (542, 512)          # Height, Width (read-only on the C4.2-S40U, no ROI)
RAW_FULL_SCALE = 1023              # 10-bit after  raw % 2**15 // 4
RAW_MID = 512
F_LIM_HZ = 4960.0                  # frame-rate limit constant for the heliCam C4 (manual 3.5.2.7)


@dataclass
class Config:
    # ---- lock-in (camera) ------------------------------------------------------------
    ref_freq_hz: float = 1000.0          # carrier = LockInTargetReferenceFrequency (306 Hz .. 250 kHz)
    # Frame timing.  Leave n_periods / n_blank / switch_after_periods empty (None) and they follow the
    # carrier: integration_ms of demodulation and frame_ms per frame, within the camera's 1..100 /
    # 0..100 period limits.  Give them explicit values to pin them.
    integration_ms: float = 20.0         # target demodulation time per frame
    frame_ms: float = 22.0               # target frame duration (the rest is blank, min min_blank_periods)
    min_blank_periods: int = 2           # dead time for the LED switch
    n_periods: Optional[int] = None      # LockInTargetTimeConstantNPeriods (1..100); None = from integration_ms
    n_blank: Optional[int] = None        # LockInTargetBlankDurationNPeriods (0..100); None = from frame_ms
    carrier_presets_hz: tuple = (1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0)   # live-view selector
    allow_link_limited: bool = False     # carriers above ~21.6 kHz: frames shorter than the 9.25 ms link time,
                                         # so bursts come with gaps; allow it (with short bursts)
    max_gap_ms: float = 100.0            # largest acceptable gap between bursts in link-limited mode
    sensitivity: float = 0.15            # LockInSensitivity (manual suggests ~0.2 for S40U)
    coupling: str = "DC"                 # LockInCoupling: "DC" | "AC"
    burst_frames: int = 100              # AcquisitionBurstFrameCount (even, 4..900)
    queue_blocks: int = 2                # TransferQueueMaxBlockCount (1 = no pipelining; try up to 4)
    reference_source: str = "Internal"   # LockInReferenceSourceType ("External" = controller is master)
    reference_input: str = "FI2"         # LockInReferenceSourceSignal, only used if External
    expected_freq_dev_pct: int = 5       # LockInExpectedFrequencyDeviation, only used if External

    # ---- real-time outputs to the LED controller ---------------------------------------
    frame_sync_line: str = "RTIO2"       # LineSource = LockInFrameSync
    reference_line: str = "RTIO3"        # LineSource = LockInReference (gates both LEDs)
    recording_active_line: Optional[str] = None   # e.g. "DIO1" - TODO pick from the D3 data sheet

    # ---- LED / wavelength assignment -----------------------------------------------------
    led_a_nm: int = 760                  # LED enabled in even frames (controller resets to A)
    led_b_nm: int = 850                  # LED enabled in odd frames
    switch_after_periods: Optional[int] = None   # controller: LockInReference edges after FrameSync before
                                                 # switching; None = middle of the blank (11 at 1 kHz, 20+2)
    led_power_mw: Optional[tuple] = None # peak light power on the whole pixel array per LED (A, B) in mW, once
                                         # measured; None -> `plan` reports the allowed power range instead
    expected_acdc: float = 0.01          # expected PPG AC/DC (pulsation depth), for the SNR estimate in `plan`
    ppg_snr_target: float = 10.0         # wanted PPG SNR per frame for the ROI mean (first ROI in `rois`)

    # ---- processing ----------------------------------------------------------------------
    rois: dict = field(default_factory=lambda: {"vessel": [256, 276, 230, 250],
                                                "background": [150, 190, 330, 370]})  # name: [y0,y1,x0,x1]
    ppg_window_s: float = 10.0           # rolling window for AC/DC and R
    ppg_min_window_s: float = 4.0        # do not report R before this much data exists
    ppg_band_hz: tuple = (0.5, 5.0)
    ppg_filter_order: int = 2
    ac_method: str = "rms"               # "rms" (robust) | "p2p" (peak-to-peak)
    noise_correct: bool = True           # rms only: subtract the white-noise floor measured in noise_band_hz
    noise_band_hz: tuple = (6.0, 10.5)   # above the pulse band, below the per-wavelength Nyquist (11.4 Hz)
    history_s: float = 60.0              # how much trace history the live view keeps
    alternation_min_contrast: float = 3.0   # |A-B| frame contrast, in units of its own scatter
    alternation_min_match: float = 0.9      # fraction of frames that must follow the A/B pattern
    auto_fix_swap: bool = True           # if a whole burst starts on B, swap it back (and flag it)
    sat_low: int = 0                     # raw I/Q at or below -> clipped
    sat_high: int = 1023                 # raw I/Q at or above -> clipped
    dc_sat_tol: int = 2                  # |I-512| and |Q-512| below this in every frame -> "flat" pixel

    # ---- offset maps -----------------------------------------------------------------------
    offset_bursts: int = 3               # bursts averaged for I0/Q0 (LEDs at their modulated mean)

    # ---- acquisition / threads -----------------------------------------------------------
    fetch_timeout_s: float = 10.0
    max_consecutive_timeouts: int = 3
    raw_q_max: int = 8                   # bursts waiting for the Processor (Acquirer drops if full)
    view_q_max: int = 1                  # viewer only shows the newest result
    write_q_max: int = 16                # raw bursts waiting for disk.  NOTE: 1 burst = ~111 MB,
                                         # so 16 = ~1.8 GB RAM; the design's 64 would be ~7 GB.

    # ---- output -------------------------------------------------------------------------
    save_dir: str = "./data"
    save_raw: bool = True
    compression: Optional[str] = "lzf"   # h5py compression for raw bursts (None = fastest)

    # ---- LED controller link -------------------------------------------------------------
    controller: str = "manual"           # "manual" (prompts) | "serial:COM7" | "sim"
    controller_baud: int = 115200

    # ---- simulation (only used with --sim) --------------------------------------------------
    sim_time_scale: float = 1.0          # 1.0 = real time; 0 = as fast as possible
    sim_shape: Optional[tuple] = None    # override sensor shape for fast tests, e.g. (64, 64)
    sim_heart_rate_bpm: float = 72.0
    sim_ppg_depth: dict = field(default_factory=lambda: {"vessel": [0.012, 0.007],      # 760, 850
                                                         "tissue": [0.004, 0.003]})
    sim_noise_counts: float = 2.0
    sim_room_light: float = 0.3          # relative to LED mean; rejected by the lock-in
    sim_driver_phase_rad: tuple = (0.6, 0.75)   # LED A, LED B phase delay (amplitude is phase-proof)
    sim_reset_miss_prob: float = 0.0     # per burst: controller misses the RecordingActive reset
    sim_sync_miss_prob: float = 0.0      # per frame: controller misses a FrameSync edge -> A/B pattern breaks
    sim_gap_s: float = 0.002             # camera dead time between pipelined bursts
    sim_seed: int = 1

    # ------------------------------------------------------------------------------------
    def __post_init__(self):
        """Derive the period counts that were left empty from the carrier and the target times."""
        auto = set()
        fd = float(self.ref_freq_hz)
        if self.n_periods is None:
            self.n_periods = int(min(100, max(1, round(self.integration_ms * fd / 1000.0))))
            auto.add("n_periods")
        if self.n_blank is None:
            total = int(round(self.frame_ms * fd / 1000.0))
            self.n_blank = int(min(100, max(self.min_blank_periods, total - self.n_periods)))
            auto.add("n_blank")
        if self.switch_after_periods is None:
            # FrameSync rises at the frame centre; switch in the middle of the blank periods (VERIFY on a scope)
            total = self.n_periods + self.n_blank
            target = self.n_periods + self.n_blank // 2 if self.n_blank else total - 1
            self.switch_after_periods = max(1, target - total // 2 + 1)
            auto.add("switch_after_periods")
        self._auto = frozenset(auto)

    @property
    def timing_auto(self) -> frozenset:
        return getattr(self, "_auto", frozenset())

    def with_carrier(self, freq_hz: float) -> "Config":
        """Same config at another carrier, with the frame timing re-derived for it."""
        return dataclasses.replace(self, ref_freq_hz=float(freq_hz), n_periods=None, n_blank=None,
                                   switch_after_periods=None)

    @property
    def frame_period_s(self) -> float:
        return (self.n_periods + self.n_blank) / self.ref_freq_hz

    @property
    def burst_duration_s(self) -> float:
        return self.burst_frames * self.frame_period_s

    def validate(self) -> list[str]:
        """Return a list of problems (empty if fine).  The camera enforces the same limits."""
        p = []
        if not (4 <= self.burst_frames <= 900) or self.burst_frames % 2:
            p.append("burst_frames must be even and 4..900")
        if not 1 <= self.n_periods <= 100:
            p.append("n_periods must be 1..100")
        if not 0 <= self.n_blank <= 100:
            p.append("n_blank must be 0..100")
        if not 306 <= self.ref_freq_hz <= 250000:
            p.append("ref_freq_hz must be 306 Hz .. 250 kHz")
        if not 0.0 <= self.sensitivity <= 1.0:
            p.append("sensitivity must be 0..1")
        if self.coupling not in ("DC", "AC"):
            p.append("coupling must be DC or AC")
        if not 1 <= self.queue_blocks <= 127:
            p.append("queue_blocks must be 1..127")
        if 1.0 / self.frame_period_s > F_LIM_HZ:   # simplified form of the manual's 3.5.2.7 inequality
            p.append(f"frame rate {1 / self.frame_period_s:.0f} Hz above the camera limit {F_LIM_HZ:g} Hz")
        if self.switch_after_periods >= self.n_periods + self.n_blank:
            p.append("switch_after_periods must be shorter than one frame")
        lo, hi = self.ppg_band_hz
        nyq = 0.5 / (2 * self.frame_period_s)
        if not 0 < lo < hi < nyq:
            p.append(f"ppg_band_hz must satisfy 0 < lo < hi < {nyq:.2f} Hz (per-wavelength Nyquist)")
        nlo, nhi = self.noise_band_hz
        if self.noise_correct and not hi < nlo < nhi < nyq:
            p.append(f"noise_band_hz must lie between ppg_band_hz and {nyq:.2f} Hz")
        for name, r in self.rois.items():
            if len(r) != 4 or not (r[0] < r[1] and r[2] < r[3]):
                p.append(f"ROI {name!r} must be [y0, y1, x0, x1] with y0<y1, x0<x1")
        return p

    # ---- (de)serialisation ------------------------------------------------------------
    def to_dict(self, resolved: bool = False) -> dict:
        """Derived period counts are written as None (so they re-derive) unless ``resolved``."""
        d = dataclasses.asdict(self)
        if not resolved:
            for k in self.timing_auto:
                d[k] = None
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(d) - set(names)
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            default = names[k].default
            kw[k] = tuple(v) if isinstance(default, tuple) and isinstance(v, list) else v
        return cls(**kw)

    def replace(self, **kw) -> "Config":
        """Like dataclasses.replace, but derived period counts stay derived (re-computed)."""
        keep_auto = {k: None for k in self.timing_auto if k not in kw}
        return dataclasses.replace(self, **keep_auto, **kw)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def load_config(path: Optional[str] = None, overrides: Optional[list[str]] = None) -> Config:
    d: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            d.update(yaml.safe_load(fh) or {})
    names = {f.name for f in dataclasses.fields(Config)}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in names:
            raise KeyError(f"Unknown config key: {key}")
        d[key] = _parse_value(raw.strip())
    return Config.from_dict(d)


def parse_frequency(text) -> float:
    """'20k', '20 kHz', '1.5k', '20000' -> Hz."""
    s = str(text).strip().lower().replace("hz", "").replace(" ", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    return float(s) * mult


def _parse_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        low = raw.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        return raw
