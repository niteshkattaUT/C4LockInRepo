"""Per-burst processing (steps 1-5 of the pipeline).  Pure functions, no threads.

1. decode            (done in camera.fetch_burst -> uint16 10-bit, (2, N, H, W))
2. offset correction  I - I0[parity], Q - Q0[parity]
3. de-interleave      even frames = LED A, odd = LED B; checked every burst
4. amplitude          A = sqrt(I^2 + Q^2)   (independent of the LED driver's phase delay)
5. ROI traces         mean of A over each ROI, one value per frame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import RAW_MID, Config


@dataclass
class OffsetMap:
    """I0/Q0 per frame parity: shape (2, H, W) each; [0] for even (LED A) frames, [1] for odd (LED B).

    The manual says the offset depends on the total light, and LED A and LED B frames
    see different light, so each parity gets its own map.  Recorded with the controller
    in OFFSET mode (same A/B alternation, LEDs at their modulated mean, not gated).
    """
    I0: np.ndarray
    Q0: np.ndarray
    n_frames: int = 0

    @classmethod
    def from_bursts(cls, bursts: list[np.ndarray]) -> "OffsetMap":
        iq = np.concatenate([b.astype(np.float32) for b in bursts], axis=1)   # (2, N_total, H, W)
        I0 = np.stack([iq[0, 0::2].mean(axis=0), iq[0, 1::2].mean(axis=0)])
        Q0 = np.stack([iq[1, 0::2].mean(axis=0), iq[1, 1::2].mean(axis=0)])
        return cls(I0, Q0, iq.shape[1])

    @classmethod
    def mid_scale(cls, shape) -> "OffsetMap":
        """Fallback when no offset was recorded: nominal 512 (leaves FPN in the data)."""
        z = np.full((2, *shape), float(RAW_MID), np.float32)
        return cls(z, z.copy(), 0)


@dataclass
class Alternation:
    state: str               # "ok" | "swapped" | "broken" | "indeterminate"
    contrast: float          # median |A - B| zig-zag in units of its scatter
    match: float             # fraction of frames following the expected pattern
    sign: int                # +1 if LED A frames are brighter in the check region, -1 otherwise


@dataclass
class BurstResult:
    index: int
    t_a: np.ndarray                      # (N/2,) seconds, LED A frame times
    t_b: np.ndarray                      # (N/2,) seconds, LED B frame times (one frame later)
    roi_a: dict                          # name -> (N/2,) mean amplitude, LED A
    roi_b: dict                          # name -> (N/2,) mean amplitude, LED B
    img_a: np.ndarray                    # mean amplitude image, LED A
    img_b: np.ndarray                    # mean amplitude image, LED B
    frame_means: np.ndarray              # (N,) mean amplitude per frame in the check region
    alternation: Alternation
    swapped: bool                        # True if the burst was swapped back (started on B)
    sat_clipped: int                     # pixels with raw I or Q at 0 / 1023 in any frame
    sat_flat: int                        # pixels flat near 512 in every frame (DC saturation sign)
    meta: object = None
    flags: list = field(default_factory=list)
    proc_s: float = 0.0


def amplitude(iq: np.ndarray, offset: OffsetMap, parity_shift: int = 0) -> np.ndarray:
    """(2, N, H, W) uint16 -> (N, H, W) float32 offset-corrected amplitude.

    ``parity_shift=1`` applies the LED-B offset to even frames (burst that started on B)."""
    N = iq.shape[1]
    par = (np.arange(N) + parity_shift) % 2
    I = iq[0].astype(np.float32)
    I -= offset.I0[par]
    Q = iq[1].astype(np.float32)
    Q -= offset.Q0[par]
    np.hypot(I, Q, out=I)
    return I


def roi_means(A: np.ndarray, rois: dict) -> dict:
    return {name: A[:, y0:y1, x0:x1].mean(axis=(1, 2)) for name, (y0, y1, x0, x1) in rois.items()}


def check_alternation(frame_means: np.ndarray, cfg: Config, expected_sign: Optional[int] = None) -> Alternation:
    """Does the per-frame amplitude alternate A, B, A, B ... ?

    z[k] = m[k] - (m[k-1] + m[k+1]) / 2 removes slow trends (pulse, drift); for a clean
    alternation z has sign +s on even frames and -s on odd ones.  ``expected_sign`` is
    the sign seen on the first good burst; a burst with the opposite sign started on
    LED B ("swapped", e.g. the controller missed the RecordingActive reset).
    """
    m = np.asarray(frame_means, float)
    if m.size < 4:
        return Alternation("indeterminate", 0.0, 0.0, 0)
    z = m[1:-1] - 0.5 * (m[:-2] + m[2:])
    alt = np.where(np.arange(1, m.size - 1) % 2 == 0, z, -z)      # +|d| on a good pattern with sign +1
    med = float(np.median(alt))
    scatter = float(np.median(np.abs(alt - med))) * 1.4826 + 1e-9
    contrast = abs(med) / scatter
    sign = 1 if med > 0 else -1
    match = float(np.mean(np.sign(alt) == sign))
    if contrast < cfg.alternation_min_contrast:
        return Alternation("indeterminate", contrast, match, sign)
    if match < cfg.alternation_min_match:
        return Alternation("broken", contrast, match, sign)
    if expected_sign is not None and sign != expected_sign:
        return Alternation("swapped", contrast, match, sign)
    return Alternation("ok", contrast, match, sign)


def saturation_counts(iq: np.ndarray, cfg: Config) -> tuple[int, int]:
    """(clipped, flat) pixel counts.

    clipped: raw I or Q at 0 or 1023 in any frame of the burst.
    flat: I and Q both within ``dc_sat_tol`` of 512 in every frame.  Under bright light
    this is the manual's sign of DC saturation; in dark regions it just means no signal.
    """
    lo, hi = cfg.sat_low, cfg.sat_high
    clipped = ((iq <= lo) | (iq >= hi)).any(axis=(0, 1))
    dev = np.abs(iq.astype(np.int16) - RAW_MID).max(axis=1)        # (2, H, W)
    flat = (dev[0] <= cfg.dc_sat_tol) & (dev[1] <= cfg.dc_sat_tol)
    return int(clipped.sum()), int(flat.sum())


class BurstProcessor:
    """Stateful wrapper: remembers the A/B sign of the first good burst."""

    def __init__(self, cfg: Config, offset: OffsetMap, rois: Optional[dict] = None):
        self.cfg = cfg
        self.offset = offset
        self.rois = dict(rois if rois is not None else cfg.rois)
        self.expected_sign: Optional[int] = None
        self.check_region = None          # (y0, y1, x0, x1) or None = whole frame

    def process(self, iq: np.ndarray, meta) -> BurstResult:
        import time
        t = time.perf_counter()
        flags = []
        A = amplitude(iq, self.offset)
        if self.check_region:
            y0, y1, x0, x1 = self.check_region
            fm = A[:, y0:y1, x0:x1].mean(axis=(1, 2))
        else:
            fm = A.mean(axis=(1, 2))
        alt = check_alternation(fm, self.cfg, self.expected_sign)
        if alt.state == "ok" and self.expected_sign is None:
            self.expected_sign = alt.sign
        swapped = False
        if alt.state == "swapped":
            flags.append("alternation swapped (burst started on LED B)")
            if self.cfg.auto_fix_swap:
                swapped = True
                A = amplitude(iq, self.offset, parity_shift=1)
        elif alt.state == "broken":
            flags.append(f"alternation broken (match {alt.match:.2f})")

        a_sl, b_sl = (slice(1, None, 2), slice(0, None, 2)) if swapped else (slice(0, None, 2), slice(1, None, 2))
        means = roi_means(A, self.rois)
        times = meta.frame_times_s() if meta is not None else np.arange(A.shape[0]) * self.cfg.frame_period_s
        n2 = A.shape[0] // 2
        clipped, flat = saturation_counts(iq, self.cfg)
        if clipped:
            flags.append(f"{clipped} clipped px")
        res = BurstResult(
            index=getattr(meta, "index", 0),
            t_a=times[a_sl][:n2], t_b=times[b_sl][:n2],
            roi_a={k: v[a_sl][:n2] for k, v in means.items()},
            roi_b={k: v[b_sl][:n2] for k, v in means.items()},
            img_a=A[a_sl].mean(axis=0), img_b=A[b_sl].mean(axis=0),
            frame_means=fm, alternation=alt, swapped=swapped,
            sat_clipped=clipped, sat_flat=flat, meta=meta, flags=flags,
        )
        res.proc_s = time.perf_counter() - t
        return res
