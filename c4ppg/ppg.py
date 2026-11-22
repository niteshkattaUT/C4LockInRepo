"""Step 6: rolling PPG metrics from the ROI traces (cheap, runs on ROI means only).

Per ROI and wavelength, over the last ``ppg_window_s``:
    DC = window mean of A
    AC = RMS (x 2*sqrt(2), i.e. peak-to-peak of an equivalent sine) or peak-to-peak
         of the 0.5-5 Hz band-passed trace
    R  = (AC_760 / DC_760) / (AC_850 / DC_850)

With each LED square-modulated from 0 to full current the demodulated amplitude is
proportional to that LED's reflected light and room light is rejected, so no
separate DC readout is needed for the ratio of ratios.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal

from .config import Config


class Trace:
    """Growing (t, value) series with trimming to ``keep_s``."""

    def __init__(self, keep_s: float):
        self.keep_s = keep_s
        self.t = np.empty(0)
        self.v = np.empty(0)

    def extend(self, t, v):
        self.t = np.concatenate([self.t, np.asarray(t, float)])
        self.v = np.concatenate([self.v, np.asarray(v, float)])
        if self.t.size and self.t[-1] - self.t[0] > self.keep_s:
            k = np.searchsorted(self.t, self.t[-1] - self.keep_s)
            self.t, self.v = self.t[k:], self.v[k:]

    def last(self, seconds: float):
        if not self.t.size:
            return self.t, self.v
        k = np.searchsorted(self.t, self.t[-1] - seconds)
        return self.t[k:], self.v[k:]


@dataclass
class WaveMetrics:
    dc: float
    ac: float
    ac_dc: float
    t: np.ndarray                 # uniform time grid of the window
    filtered: np.ndarray          # band-passed trace / DC  (relative pulsation)


@dataclass
class RoiMetrics:
    name: str
    a: Optional[WaveMetrics]      # LED A (e.g. 760 nm)
    b: Optional[WaveMetrics]      # LED B (e.g. 850 nm)
    R: float = float("nan")
    hr_bpm: float = float("nan")


@dataclass
class PPGState:
    t_now: float
    window_s: float
    rois: dict = field(default_factory=dict)     # name -> RoiMetrics
    R_hist: dict = field(default_factory=dict)   # name -> (t[], R[])


def resample_uniform(t, v, fs):
    """Linear interpolation onto a uniform grid (bridges small gaps between bursts)."""
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    return tu, np.interp(tu, t, v)


def bandpass_sos(cfg: Config, fs: float):
    lo, hi = cfg.ppg_band_hz
    return signal.butter(cfg.ppg_filter_order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def _enbw(sos, fs: float) -> float:
    """Equivalent noise bandwidth (Hz) of a zero-phase sosfiltfilt pass (power gain |H|^4)."""
    f, h = signal.sosfreqz(sos, worN=8192, fs=fs)
    return float(np.trapezoid(np.abs(h) ** 4, f))


class NoiseFloor:
    """In-band white-noise power estimated from a band above the pulse.

    Sensor noise adds to the AC of both wavelengths and pulls R towards 1 in weakly
    pulsating regions; subtracting the expected in-band noise power removes that bias.
    """

    def __init__(self, cfg: Config, fs: float, sos_band):
        self.sos = signal.butter(4, list(cfg.noise_band_hz), btype="bandpass", fs=fs, output="sos")
        self.scale = _enbw(sos_band, fs) / _enbw(self.sos, fs)

    def in_band_var(self, x: np.ndarray) -> float:
        y = signal.sosfiltfilt(self.sos, x)
        return float(np.mean(y ** 2)) * self.scale


def wave_metrics(t, v, cfg: Config, fs: float, sos, noise: Optional["NoiseFloor"] = None) -> Optional[WaveMetrics]:
    if t.size < 8 or t[-1] - t[0] < cfg.ppg_min_window_s:
        return None
    tu, vu = resample_uniform(t, v, fs)
    padlen = 3 * (2 * sos.shape[0] + 1)
    if vu.size <= padlen:
        return None
    dc = float(np.mean(vu))
    if not np.isfinite(dc) or dc <= 0:
        return None
    x = signal.sosfiltfilt(sos, vu - dc)
    # ignore filter edge effects in the AC estimate
    edge = min(int(fs * 0.5), x.size // 6)
    core = x[edge:x.size - edge] if edge else x
    if cfg.ac_method == "p2p":
        ac = float(np.percentile(core, 99) - np.percentile(core, 1))
    else:
        var = float(np.mean(core ** 2))
        if noise is not None:
            nv = noise.in_band_var(vu - dc)
            var = max(var - nv, 0.0)
        ac = float(np.sqrt(var) * 2 * np.sqrt(2))
    return WaveMetrics(dc, ac, ac / dc, tu, x / dc)


def heart_rate(x: np.ndarray, fs: float, lo_hz: float = 0.7, hi_hz: float = 3.5) -> float:
    if x.size < int(4 * fs):
        return float("nan")
    f, p = signal.periodogram(x * np.hanning(x.size), fs=fs, nfft=max(4096, x.size))
    m = (f >= lo_hz) & (f <= hi_hz)
    return float(f[m][np.argmax(p[m])] * 60.0) if m.any() else float("nan")


class PPGTracker:
    """Accumulates BurstResults and computes metrics on demand."""

    def __init__(self, cfg: Config, frame_period_s: Optional[float] = None):
        self.cfg = cfg
        T = frame_period_s or cfg.frame_period_s
        self.fs = 1.0 / (2 * T)                      # per-wavelength sample rate
        self.sos = bandpass_sos(cfg, self.fs)
        self.noise = NoiseFloor(cfg, self.fs, self.sos) if (cfg.noise_correct and cfg.ac_method == "rms") else None
        keep = max(cfg.history_s, cfg.ppg_window_s) + 5
        self.keep = keep
        self.a: dict[str, Trace] = {}
        self.b: dict[str, Trace] = {}
        self.R_hist: dict[str, list] = {}
        self.t0: Optional[float] = None

    def add(self, res) -> None:
        if self.t0 is None:
            self.t0 = float(res.t_a[0])
        for name in res.roi_a:
            self.a.setdefault(name, Trace(self.keep)).extend(res.t_a - self.t0, res.roi_a[name])
            self.b.setdefault(name, Trace(self.keep)).extend(res.t_b - self.t0, res.roi_b[name])

    def reset_roi(self, name: str):
        self.a.pop(name, None)
        self.b.pop(name, None)
        self.R_hist.pop(name, None)

    def compute(self) -> PPGState:
        cfg = self.cfg
        t_now = max((tr.t[-1] for tr in self.a.values() if tr.t.size), default=0.0)
        st = PPGState(t_now, cfg.ppg_window_s)
        for name in self.a:
            ta, va = self.a[name].last(cfg.ppg_window_s)
            tb, vb = self.b[name].last(cfg.ppg_window_s)
            ma = wave_metrics(ta, va, cfg, self.fs, self.sos, self.noise)
            mb = wave_metrics(tb, vb, cfg, self.fs, self.sos, self.noise)
            rm = RoiMetrics(name, ma, mb)
            if ma and mb and mb.ac_dc > 0:
                rm.R = ma.ac_dc / mb.ac_dc
                n = min(ma.filtered.size, mb.filtered.size)
                rm.hr_bpm = heart_rate(ma.filtered[-n:] + mb.filtered[-n:], self.fs)
                h = self.R_hist.setdefault(name, [[], []])
                if not h[0] or t_now > h[0][-1]:
                    h[0].append(t_now)
                    h[1].append(rm.R)
            st.rois[name] = rm
        st.R_hist = {k: (np.array(v[0]), np.array(v[1])) for k, v in self.R_hist.items()}
        return st

    def traces(self) -> dict:
        """name -> (t_a, v_a, t_b, v_b) for plotting/export."""
        return {n: (self.a[n].t, self.a[n].v, self.b[n].t, self.b[n].v) for n in self.a}
