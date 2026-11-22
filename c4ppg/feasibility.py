"""Real-time feasibility: the Heliotis "C4 calculator" (C4_calculator_20240712_S4_S3.xlsx) in code,
plus the streaming rules that tie the camera settings to back-to-back burst acquisition.

Spreadsheet model (cell references in brackets):
    tqp = 1e6 / (4 fd)                                   quarter period [us]            [G5]
    tf  = (Np + bs) * 4 tqp                              frame duration [us]            [G6]
    tv  = tf * Nf / 1e6                                  volume (burst) duration [s]    [G7]
    tt_rawIQ = 64/3 * npx * npy * Nf / 8e8               raw I/Q transfer to PC [s]     [K6]
    tt_amp   = 0.075 + 2*16 * npx * npy * Nf / 8e8       amp/phase volume transfer [s]  [K5]  (+-25 %)
    photon budget -> photo-e per quarter period, full-well fill (<50 %), shot + read noise,
    minimum detectable AC/DC and SNR at an expected AC/DC.                             [C26..D56]

64/3 bits per pixel and frame is 10-bit I + 10-bit Q packed with overhead, at an effective
800 Mbit/s GigE rate: one full 542 x 512 frame takes 7.40 ms on the link (~135 fps max).

Our frames also contain blank periods (LED switching), which the spreadsheet does not
have: tf = (Np + n_blank + bs) / fd, with bs = 1 for AC coupling (background measured in
the first period of each frame, manual Fig. "AC coupling").

Streaming rules (pipelined freerunning bursts, TransferQueueMaxBlockCount >= 2):
  1. link:     transfer of one burst (+25 % margin) <= acquisition of the next burst
               -> the transfer queue never grows; independent of the burst length because both
               scale with Nf, so it is a rule on the frame duration: tf >= 1.25 * 7.40 ms.
  2. pipelining on (queue_blocks >= 2); with 1 the camera idles during every transfer.
  3. host:     processing one burst < acquiring one burst (measured, see `live` summary).
  4. PPG:      per-wavelength sample rate 1 / (2 tf) > 2 x upper PPG band edge.
  5. camera:   even Nf in 4..900, frame rate below f_lim.
  6. latency:  live-view delay ~ tv + tt (one burst + its transfer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .config import F_LIM_HZ, Config

H_PLANCK = 6.626e-34      # as in the spreadsheet
C_LIGHT = 3.0e8
LINK_BITS_PER_PX = 64 / 3
LINK_BPS = 8e8
LINK_MARGIN = 0.25        # the spreadsheet's +-25 %
AMP_OVERHEAD_S = 0.075


@dataclass(frozen=True)
class Sensor:
    name: str
    bsf: float        # background suppression factor
    etc: float        # electrons per ADC count
    rnc: float        # read-noise constant contribution
    rnm: float        # read-noise per integrated period
    npx: int
    npy: int
    ppx_um: float
    ppy_um: float
    area_mm2: float
    fill_pct: float   # optical fill factor
    fullwell_ke: float
    fd_max_hz: float


SENSORS = {
    "S40": Sensor("S40", 30, 460, 0.81, 0.32, 542, 512, 24, 24, 160, 21, 500, 250000),
    "S40U": Sensor("S40U", 30, 460, 0.81, 0.32, 542, 512, 24, 24, 160, 55, 500, 250000),
    "S41U": Sensor("S41U", 30, 115, 0.81, 0.8, 542, 512, 24, 24, 160, 55, 125, 250000),
    "S4M0": Sensor("S4M0", 5, 70, 1.9, 0.9, 1102, 1024, 12, 12, 160, 23, 70, 50000),
    "S41MU": Sensor("S41MU", 30, 140, 1.5, 0.6, 1102, 1024, 12, 12, 160, 55, 140, 50000),
    "S3-ML1": Sensor("S3-ML1", 30, 160, 1.0, 0.5, 292, 280, 40, 40, 40 * 40 * 300 * 300 / 1e6, 50, 175, 250000),
}

# QE table from the spreadsheet (S40U).  Only goes to 800 nm.
QE_TABLE = [(400, 61.7), (450, 73.1), (500, 78.7), (550, 78.6), (600, 74.9), (650, 72.9),
            (700, 69.4), (750, 58.4), (800, 45.5)]


def qe_at(wavelength_nm: float) -> tuple[float, bool]:
    """Linear interpolation in the QE table; returns (QE %, extrapolated?).

    Beyond 800 nm the last segment is extended linearly - a rough guess (850 nm -> ~32.6 %).
    Get the real S40U curve from Heliotis for 850 nm.
    """
    xs = [x for x, _ in QE_TABLE]
    ys = [y for _, y in QE_TABLE]
    if xs[0] <= wavelength_nm <= xs[-1]:
        for (x0, y0), (x1, y1) in zip(QE_TABLE, QE_TABLE[1:]):
            if x0 <= wavelength_nm <= x1:
                return y0 + (y1 - y0) * (wavelength_nm - x0) / (x1 - x0), False
    if wavelength_nm > xs[-1]:
        (x0, y0), (x1, y1) = QE_TABLE[-2], QE_TABLE[-1]
    else:
        (x0, y0), (x1, y1) = QE_TABLE[0], QE_TABLE[1]
    return max(0.0, y0 + (y1 - y0) * (wavelength_nm - x0) / (x1 - x0)), True


# ---------------------------------------------------------------------------------------
# Timing / link
# ---------------------------------------------------------------------------------------
def frame_duration_s(fd_hz: float, n_periods: int, n_blank: int = 0, ac_coupling: bool = False) -> float:
    return (n_periods + n_blank + (1 if ac_coupling else 0)) / fd_hz


def transfer_time_s(n_frames: int, npx: int = 542, npy: int = 512) -> float:
    """Raw I/Q burst transfer time to the PC (spreadsheet K6)."""
    return LINK_BITS_PER_PX * npx * npy * n_frames / LINK_BPS


def amp_phase_transfer_time_s(n_frames: int, npx: int = 542, npy: int = 512) -> float:
    """Amplitude/phase volume transfer time (spreadsheet K5)."""
    return AMP_OVERHEAD_S + 2 * 16 * npx * npy * n_frames / LINK_BPS


def max_streaming_fps(npx: int = 542, npy: int = 512, margin: float = LINK_MARGIN) -> float:
    return 1.0 / (transfer_time_s(1, npx, npy) * (1 + margin))


# ---------------------------------------------------------------------------------------
# Photon budget (spreadsheet rows 26-56)
# ---------------------------------------------------------------------------------------
@dataclass
class Budget:
    inputs: dict
    tqp_us: float
    tf_us: float
    tv_s: float
    tt_rawiq_s: float
    tt_amp_s: float
    qe_pct: float
    qe_extrapolated: bool
    Pd: float
    Pp_uW: float
    E_uJ: float
    Eph_uJ: float
    Nph: float
    Npe: float
    Nper: float
    fullwell_fill_pct: float
    fits: bool
    Npef: float
    Npef_after_bs: float
    psn_e: float
    psn_cnt: float
    rn_cnt: float
    rn_e: float
    noise_e: float
    noise_cnt: float
    racdc_frame: float
    racdc_volume: float
    racdc_10x10: float
    snr_frame: float
    snr_volume: float


def calculator(fd_hz: float = 10000.0, n_periods: int = 100, n_frames: int = 900, bs: int = 1,
               power_mw: float = 45.0, wavelength_nm: float = 700.0, sensor: str = "S40U",
               qe_pct: Optional[float] = None, snr_target: float = 2.0, expected_acdc: float = 0.02,
               n_blank: int = 0) -> Budget:
    """The spreadsheet, cell for cell.  Defaults = the values saved in the spreadsheet.

    ``n_blank`` (not in the spreadsheet) only lengthens tf/tv; it does not add signal.
    """
    s = SENSORS[sensor]
    ext = False
    if qe_pct is None:
        qe_pct, ext = qe_at(wavelength_nm)
    tqp = 1e6 * 0.25 / fd_hz
    tf = (n_periods + bs + n_blank) * tqp * 4
    tv = tf * n_frames / 1e6
    Pd = power_mw / s.area_mm2
    Pp = (Pd * 1000 * s.area_mm2 * s.fill_pct / 100) / (s.npx * s.npy)
    E = Pp * tqp / 1e6
    Eph = 1e6 * H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)
    Nph = E / Eph
    Npe = Nph * qe_pct / 100
    Nper = Npe if bs == 0 else Npe / s.bsf
    fill = 100 * Nper / (s.fullwell_ke * 1000)
    Npef = Npe * n_periods * 2
    Npef_bs = Nper * n_periods * 2
    psn_e = math.sqrt((1 + bs) * Npef)
    psn_c = psn_e / s.etc
    rn_c = math.sqrt(s.rnc + n_periods * s.rnm)
    rn_e = rn_c * s.etc
    noise_e = math.sqrt(psn_e ** 2 + rn_e ** 2)
    noise_c = math.sqrt(psn_c ** 2 + rn_c ** 2)
    racdc = snr_target * noise_e / Npef
    snr = Npef / noise_e * expected_acdc
    return Budget(
        inputs=dict(fd_hz=fd_hz, n_periods=n_periods, n_frames=n_frames, bs=bs, power_mw=power_mw,
                    wavelength_nm=wavelength_nm, sensor=sensor, snr_target=snr_target,
                    expected_acdc=expected_acdc, n_blank=n_blank),
        tqp_us=tqp, tf_us=tf, tv_s=tv,
        tt_rawiq_s=transfer_time_s(n_frames, s.npx, s.npy), tt_amp_s=amp_phase_transfer_time_s(n_frames, s.npx, s.npy),
        qe_pct=qe_pct, qe_extrapolated=ext, Pd=Pd, Pp_uW=Pp, E_uJ=E, Eph_uJ=Eph, Nph=Nph, Npe=Npe, Nper=Nper,
        fullwell_fill_pct=fill, fits=fill < 50, Npef=Npef, Npef_after_bs=Npef_bs,
        psn_e=psn_e, psn_cnt=psn_c, rn_cnt=rn_c, rn_e=rn_e, noise_e=noise_e, noise_cnt=noise_c,
        racdc_frame=racdc, racdc_volume=racdc / math.sqrt(n_frames), racdc_10x10=racdc / 10.0,
        snr_frame=snr, snr_volume=snr * math.sqrt(n_frames),
    )


# ---------------------------------------------------------------------------------------
# Streaming plan for a Config
# ---------------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hard: bool = True      # False = advisory


@dataclass
class Plan:
    tf_s: float
    tv_s: float
    tt_s: float
    link_util: float                 # transfer time / acquisition time (nominal)
    link_util_worst: float           # with +25 %
    fps: float
    fs_per_wavelength: float
    latency_s: float
    host_mb_s: float                 # unpacked 16-bit I/Q into RAM / disk
    wire_mbit_s: float
    checks: list = field(default_factory=list)
    budgets: dict = field(default_factory=dict)   # wavelength -> (Budget, limits, power)
    mode: str = "continuous"         # "continuous" | "link-limited"
    gap_s: float = 0.0               # camera idle time per burst (link-limited, worst case)
    avg_fps: float = 0.0             # average delivered frame rate
    max_burst_link_limited: int = 0  # largest burst whose gap stays <= max_gap_ms

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.hard)


def plan(cfg: Config, shape=(542, 512), proc_s_per_burst: Optional[float] = None,
         sensor: str = "S40U") -> Plan:
    npy, npx = shape
    ac = cfg.coupling == "AC"
    tf = frame_duration_s(cfg.ref_freq_hz, cfg.n_periods, cfg.n_blank, ac)
    N = cfg.burst_frames
    tv = N * tf
    tt = transfer_time_s(N, npx, npy)
    util = tt / tv
    worst = util * (1 + LINK_MARGIN)
    fs = 1.0 / (2 * tf)
    p = Plan(tf_s=tf, tv_s=tv, tt_s=tt, link_util=util, link_util_worst=worst, fps=1 / tf, fs_per_wavelength=fs,
             latency_s=tv + tt, host_mb_s=2 * npx * npy * 2 / tf / 1e6,
             wire_mbit_s=LINK_BITS_PER_PX * npx * npy / tf / 1e6)
    c = p.checks
    tf_min = transfer_time_s(1, npx, npy) * (1 + LINK_MARGIN)
    p.avg_fps = p.fps
    if worst <= 1.0:
        c.append(Check("link keeps up (transfer +25 % <= acquisition)", True,
                       f"burst transfer {tt * 1e3:.0f} ms vs acquisition {tv * 1e3:.0f} ms: link {util * 100:.0f} % "
                       f"busy ({worst * 100:.0f} % worst case); needs frame >= {tf_min * 1e3:.2f} ms, "
                       f"have {tf * 1e3:.2f} ms"))
    else:
        # frames shorter than their link time: the camera records a burst, then waits for the link
        p.mode = "link-limited"
        p.gap_s = N * tf_min - tv
        p.avg_fps = 1.0 / tf_min
        per_frame_gap = tf_min - tf
        nmax = int(cfg.max_gap_ms / 1e3 / per_frame_gap) // 2 * 2
        p.max_burst_link_limited = nmax if nmax >= 4 else 0
        f_cont = 200.0 / tf_min          # 100 + 100 periods at the longest possible frame
        detail = (f"frame {tf * 1e3:.2f} ms < link time {tf_min * 1e3:.2f} ms: the camera records {N} frames in "
                  f"{tv * 1e3:.0f} ms, then waits ~{p.gap_s * 1e3:.0f} ms for the link; average {p.avg_fps:.0f} fps")
        if cfg.allow_link_limited:
            c.append(Check(f"link-limited bursts: gap <= {cfg.max_gap_ms:g} ms", p.gap_s * 1e3 <= cfg.max_gap_ms,
                           detail + (f"; use burst_frames <= {p.max_burst_link_limited}" if p.max_burst_link_limited
                                     else "; no burst size gets the gap that short")))
        else:
            c.append(Check("link keeps up (transfer +25 % <= acquisition)", False,
                           detail + f". Continuous streaming needs a carrier <= {f_cont / 1e3:.1f} kHz "
                           f"(max 100 + 100 periods per frame), or set allow_link_limited: true"
                           + (f" with burst_frames <= {p.max_burst_link_limited}" if p.max_burst_link_limited else "")))
    c.append(Check("pipelining on (TransferQueueMaxBlockCount >= 2)", cfg.queue_blocks >= 2,
                   f"queue_blocks = {cfg.queue_blocks}"
                   + ("" if cfg.queue_blocks >= 2 else
                      f"; camera idles {tt * 1e3:.0f} ms per burst -> duty cycle {tv / (tv + tt) * 100:.0f} %")))
    c.append(Check("burst size legal (even, 4..900)", 4 <= N <= 900 and N % 2 == 0, f"{N} frames"))
    c.append(Check("frame rate below f_lim", p.fps <= F_LIM_HZ, f"{p.fps:.1f} fps (f_lim {F_LIM_HZ:g})"))
    hi = cfg.ppg_band_hz[1]
    fs_eff = p.avg_fps / 2
    c.append(Check("PPG band below per-wavelength Nyquist", hi < fs_eff / 2,
                   f"{fs_eff:.2f} Hz per wavelength -> Nyquist {fs_eff / 2:.2f} Hz, band to {hi:g} Hz"))
    c.append(Check("PPG window holds several bursts", cfg.ppg_window_s >= 2 * tv,
                   f"window {cfg.ppg_window_s:g} s, burst {tv:.2f} s", hard=False))
    c.append(Check("live latency <= 5 s", p.latency_s <= 5.0,
                   f"~{p.latency_s:.2f} s (one burst + transfer); smaller bursts = lower latency", hard=False))
    ram = N * 2 * npx * npy * 2 / 1e6
    c.append(Check("queued bursts fit in RAM (<= 4 GB)", ram * (cfg.raw_q_max + cfg.write_q_max) <= 4000,
                   f"{ram:.0f} MB per burst x {cfg.raw_q_max + cfg.write_q_max} queued = "
                   f"{ram * (cfg.raw_q_max + cfg.write_q_max) / 1000:.1f} GB", hard=False))
    if proc_s_per_burst is not None:
        c.append(Check("host processing < burst duration", proc_s_per_burst < tv,
                       f"{proc_s_per_burst * 1e3:.0f} ms per burst vs {tv * 1e3:.0f} ms"))
    if cfg.coupling == "AC" and cfg.n_blank == 0:
        c.append(Check("AC coupling period", True, "AC coupling adds 1 period per frame (counted above)", hard=False))

    # photon budget per LED: frames are lit by one LED; square wave 0..full = full power in the on quarters
    bs = 1 if ac else 0
    roi_px = _first_roi_px(cfg)
    powers = cfg.led_power_mw or (None, None)
    for led, nm, pw in (("A", cfg.led_a_nm, powers[0]), ("B", cfg.led_b_nm, powers[1])):
        lim = power_limits(cfg, nm, roi_px, sensor)
        if pw is None:
            b = calculator(cfg.ref_freq_hz, cfg.n_periods, N, bs, lim["p_max_mw"], nm, sensor,
                           expected_acdc=cfg.expected_acdc, n_blank=cfg.n_blank)
            p.budgets[nm] = (b, lim, None)
            c.append(Check(f"LED {led} {nm} nm power window", lim["p_min_mw"] <= lim["p_max_mw"],
                           f"keep {lim['p_min_mw']:.3g} .. {lim['p_max_mw']:.3g} mW on the sensor "
                           f"(min for PPG SNR {cfg.ppg_snr_target:g} in a {roi_px}-px ROI, max for 50 % full well)"
                           + (" - QE extrapolated" if b.qe_extrapolated else ""), hard=False))
        else:
            b = calculator(cfg.ref_freq_hz, cfg.n_periods, N, bs, pw, nm, sensor,
                           expected_acdc=cfg.expected_acdc, n_blank=cfg.n_blank)
            p.budgets[nm] = (b, lim, pw)
            c.append(Check(f"LED {led} {nm} nm full well < 50 % per quarter period", b.fits,
                           f"{b.fullwell_fill_pct:.1f} % at {pw:g} mW (max {lim['p_max_mw']:.3g} mW)"
                           + (" - QE extrapolated" if b.qe_extrapolated else "")))
            snr_roi = b.snr_frame * math.sqrt(roi_px)
            c.append(Check(f"LED {led} {nm} nm PPG SNR", snr_roi >= cfg.ppg_snr_target,
                           f"{snr_roi:.0f} per frame in a {roi_px}-px ROI at AC/DC {cfg.expected_acdc:g} "
                           f"(target {cfg.ppg_snr_target:g}; min power {lim['p_min_mw']:.3g} mW)", hard=False))
    return p


def _first_roi_px(cfg: Config) -> int:
    if not cfg.rois:
        return 1
    y0, y1, x0, x1 = next(iter(cfg.rois.values()))
    return max(1, (y1 - y0) * (x1 - x0))


def power_limits(cfg: Config, wavelength_nm: float, roi_px: int, sensor: str = "S40U") -> dict:
    """Light power on the pixel array (mW) between 'enough for the PPG' and '50 % full well'.

    p_max: full-well fill is linear in power -> 50 % / fill(1 mW).
    p_min: ROI-mean SNR per frame = snr_frame(P) * sqrt(roi_px) reaches ppg_snr_target (bisection;
           shot noise grows with sqrt(P), read noise is constant).
    """
    bs = 1 if cfg.coupling == "AC" else 0
    args = (cfg.ref_freq_hz, cfg.n_periods, cfg.burst_frames, bs)

    def bud(pw):
        return calculator(*args, pw, wavelength_nm, sensor, expected_acdc=cfg.expected_acdc, n_blank=cfg.n_blank)

    p_max = 50.0 / bud(1.0).fullwell_fill_pct
    target = cfg.ppg_snr_target / math.sqrt(roi_px)
    lo, hi = 1e-12, max(p_max, 1e-9) * 1e3
    if bud(hi).snr_frame < target:
        p_min = float("inf")
    else:
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if bud(mid).snr_frame < target else (lo, mid)
            if hi / lo < 1.0001:
                break
        p_min = hi
    return {"p_max_mw": p_max, "p_min_mw": p_min}


def burst_options(cfg: Config, sizes=(10, 20, 50, 100, 200, 450, 900), shape=(542, 512)) -> list[dict]:
    """Latency / transfer table for candidate burst sizes at the current frame settings."""
    out = []
    for n in sizes:
        p = plan(cfg.replace(burst_frames=n), shape)
        out.append(dict(frames=n, burst_s=p.tv_s, transfer_s=p.tt_s, latency_s=p.latency_s,
                        link_pct=p.link_util * 100, ram_mb=n * 2 * shape[0] * shape[1] * 2 / 1e6,
                        ok=p.ok))
    return out


def carrier_table(cfg: Config, freqs=None, shape=(542, 512)) -> list[dict]:
    """What each carrier means with the frame timing derived from it (integration_ms / frame_ms)."""
    rows = []
    for f in (freqs or cfg.carrier_presets_hz):
        c = cfg.with_carrier(f)
        p = plan(c, shape)
        lims = [p.budgets[nm][1]["p_max_mw"] for nm in (c.led_a_nm, c.led_b_nm)]
        if p.mode == "continuous":
            verdict = "streams" if p.ok else "FAILS"
        else:
            verdict = (f"bursts <= {p.max_burst_link_limited} frames" if p.max_burst_link_limited
                       else "too fast") + ("" if c.allow_link_limited else " (allow_link_limited)")
        rows.append(dict(freq_hz=f, n_periods=c.n_periods, n_blank=c.n_blank, switch_after=c.switch_after_periods,
                         integration_ms=c.n_periods / f * 1e3, frame_ms=p.tf_s * 1e3, avg_fps=p.avg_fps,
                         link_pct=min(p.link_util, 1 / (1 + LINK_MARGIN)) * 100, mode=p.mode, verdict=verdict,
                         pmax_a_mw=lims[0], pmax_b_mw=lims[1], ok=p.ok))
    return rows


def format_carrier_table(cfg: Config, rows: list[dict]) -> str:
    L = [f"Carrier choices (integration {cfg.integration_ms:g} ms / frame {cfg.frame_ms:g} ms targets, "
         f"camera max 100 + 100 periods):",
         "   carrier  periods+blank  switch  integr_ms  frame_ms  avg_fps  max mW A/B (50% FW)   verdict"]
    for r in rows:
        L.append(f"  {r['freq_hz'] / 1e3:6.1f}k  {r['n_periods']:>6} + {r['n_blank']:<4}  {r['switch_after']:>6}  "
                 f"{r['integration_ms']:9.2f}  {r['frame_ms']:8.2f}  {r['avg_fps']:7.1f}  "
                 f"{r['pmax_a_mw']:8.3g} / {r['pmax_b_mw']:<8.3g}  {r['verdict']}")
    return "\n".join(L)


def format_plan(cfg: Config, p: Plan, shape=(542, 512)) -> str:
    L = [f"Carrier {cfg.ref_freq_hz / 1e3:g} kHz: {cfg.n_periods} + {cfg.n_blank} periods, LED switch after "
         f"{cfg.switch_after_periods} periods ({'derived' if cfg.timing_auto else 'pinned'})",
         f"Frame {p.tf_s * 1e3:.2f} ms ({p.fps:.2f} fps, {p.fs_per_wavelength:.2f} Hz per wavelength), "
         f"burst {cfg.burst_frames} frames = {p.tv_s:.3f} s, transfer {p.tt_s:.3f} s, mode {p.mode}",
         f"Link {p.wire_mbit_s:.0f} Mbit/s of 800 ({p.link_util * 100:.0f} %, {p.link_util_worst * 100:.0f} % worst case), "
         f"host {p.host_mb_s:.0f} MB/s unpacked; max streaming {max_streaming_fps():.0f} fps",
         ""]
    for ch in p.checks:
        tag = ("PASS" if ch.ok else "FAIL") if ch.hard else ("ok" if ch.ok else "WARN")
        L.append(f"  [{tag:>4}] {ch.name}: {ch.detail}")
    L.append("")
    L.append(f"Photon budget per LED (S40U, {cfg.ref_freq_hz:g} Hz, {cfg.n_periods} periods, {cfg.coupling} coupling, "
             f"expected AC/DC {cfg.expected_acdc:g}):")
    for nm, (b, lim, pw) in p.budgets.items():
        at = f"at {pw:g} mW" if pw is not None else f"at the full-well limit {lim['p_max_mw']:.3g} mW"
        L.append(f"  {nm} nm: QE {b.qe_pct:.1f} %{'*' if b.qe_extrapolated else ''}; allowed power "
                 f"{lim['p_min_mw']:.3g} .. {lim['p_max_mw']:.3g} mW on the sensor")
        L.append(f"          {at}: {b.Npe:.3g} e-/quarter period, full well {b.fullwell_fill_pct:.1f} %, "
                 f"noise {b.noise_cnt:.1f} counts/frame/px, min AC/DC {b.racdc_frame:.2e} (1 px, 1 frame, SNR 2), "
                 f"SNR {b.snr_frame:.0f} per px-frame")
    if any(b.qe_extrapolated for b, _, _ in p.budgets.values()):
        L.append("  * QE table ends at 800 nm; value extrapolated - ask Heliotis for the S40U curve.")
    L.append("  Note: the spreadsheet has no LockInSensitivity term; the full-well check assumes the "
             "sensor model at its defaults.")
    L.append("")
    L.append(f"Burst size options at these frame settings:")
    L.append("   frames  burst_s  transfer_s  latency_s  link_%  RAM_MB  ok")
    for o in burst_options(cfg, shape=shape):
        L.append(f"   {o['frames']:>6}  {o['burst_s']:7.3f}  {o['transfer_s']:10.3f}  {o['latency_s']:9.3f}  "
                 f"{o['link_pct']:6.0f}  {o['ram_mb']:6.0f}  {'yes' if o['ok'] else 'no'}")
    return "\n".join(L)
