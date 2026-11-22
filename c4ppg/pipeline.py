"""Threads: Acquirer -> raw_q -> Processor -> (view_q -> viewer on the main thread, write_q -> Writer).

* raw_q   (cfg.raw_q_max):  if it fills, the Acquirer drops and counts the burst rather
                            than blocking the camera (a blocked fetch loop = missed bursts).
* view_q  (cfg.view_q_max): only the newest result is kept; older ones are discarded.
* write_q (cfg.write_q_max): raw data goes to disk untouched, before any processing.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .camera import BurstMeta, C4Camera
from .config import Config
from .ppg import PPGTracker
from .processing import BurstProcessor, OffsetMap

log = logging.getLogger("c4ppg")
_STOP = object()


@dataclass
class Stats:
    """Thread-safe counters shared by all workers."""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    bursts: int = 0
    dropped_raw: int = 0
    dropped_write: int = 0
    timeouts: int = 0
    written: int = 0
    write_bytes: int = 0
    gaps_s: list = field(default_factory=list)        # FrameStart gap between consecutive bursts
    frame_id_jumps: int = 0
    alternation: dict = field(default_factory=dict)   # state -> count
    swapped: int = 0
    last_clipped: int = 0
    last_flat: int = 0
    proc_s: list = field(default_factory=list)
    fetch_s: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    t_start: float = field(default_factory=time.time)

    def inc(self, name: str, n: int = 1):
        with self.lock:
            setattr(self, name, getattr(self, name) + n)

    def summary(self) -> dict:
        with self.lock:
            g = np.array(self.gaps_s) if self.gaps_s else np.array([np.nan])
            return dict(
                bursts=self.bursts, dropped_raw=self.dropped_raw, dropped_write=self.dropped_write,
                timeouts=self.timeouts, written=self.written, write_mb=round(self.write_bytes / 1e6, 1),
                gap_ms_mean=float(np.mean(g) * 1e3) if self.gaps_s else float("nan"),
                gap_ms_max=float(np.max(g) * 1e3) if self.gaps_s else float("nan"),
                gaps_n=len(self.gaps_s), frame_id_jumps=self.frame_id_jumps,
                alternation=dict(self.alternation), swapped=self.swapped,
                last_clipped=self.last_clipped, last_flat=self.last_flat,
                proc_ms_mean=float(np.mean(self.proc_s) * 1e3) if self.proc_s else float("nan"),
                errors=list(self.errors), elapsed_s=round(time.time() - self.t_start, 1),
            )


class GapTracker:
    """Detect gaps and lost frames between bursts from ChunkTimestamp / ChunkFrameID."""

    def __init__(self):
        self.prev: Optional[BurstMeta] = None

    def update(self, meta: BurstMeta) -> tuple[Optional[float], bool]:
        gap, jump = None, False
        p = self.prev
        if p is not None:
            if p.t0_ns is not None and meta.t0_ns is not None:
                gap = (meta.t0_ns - p.t0_ns) * 1e-9 - p.n_frames * p.frame_period_s
            else:
                gap = (meta.t_host - p.t_host) - meta.n_frames * meta.frame_period_s
            if p.frame_id is not None and meta.frame_id is not None:
                jump = meta.frame_id != p.frame_id + p.n_frames   # VERIFY wrap-around value
        self.prev = meta
        return gap, jump


class Acquirer(threading.Thread):
    """camera.start() once, then fetch bursts back to back."""

    def __init__(self, cam: C4Camera, out_q: queue.Queue, stop_evt: threading.Event, stats: Stats,
                 max_bursts: Optional[int] = None, write_q: Optional[queue.Queue] = None):
        super().__init__(name="Acquirer", daemon=True)
        self.cam, self.out_q, self.stop_evt, self.stats = cam, out_q, stop_evt, stats
        self.max_bursts = max_bursts
        self.gaps = GapTracker()
        self.error: Optional[BaseException] = None

    def run(self):
        consecutive = 0
        try:
            self.cam.start()
            while not self.stop_evt.is_set():
                if self.max_bursts is not None and self.stats.bursts >= self.max_bursts:
                    break
                try:
                    iq, meta = self.cam.fetch_burst()
                except Exception as e:
                    if "timeout" in type(e).__name__.lower() or isinstance(e, TimeoutError):
                        self.stats.inc("timeouts")
                        consecutive += 1
                        log.warning("fetch timeout (%d in a row)", consecutive)
                        if consecutive >= self.cam.cfg.max_consecutive_timeouts:
                            raise RuntimeError(f"{consecutive} fetch timeouts in a row") from e
                        continue
                    raise
                consecutive = 0
                gap, jump = self.gaps.update(meta)
                with self.stats.lock:
                    self.stats.bursts += 1
                    self.stats.fetch_s.append(meta.fetch_s)
                    if gap is not None:
                        self.stats.gaps_s.append(gap)
                    if jump:
                        self.stats.frame_id_jumps += 1
                if jump:
                    meta.flags.append("frame ID jump (frames lost)")
                try:
                    self.out_q.put_nowait((iq, meta))
                except queue.Full:
                    self.stats.inc("dropped_raw")
                    log.warning("raw_q full - dropped burst %d", meta.index)
        except BaseException as e:   # noqa: BLE001 - reported to the main thread
            self.error = e
            with self.stats.lock:
                self.stats.errors.append(f"Acquirer: {e!r}")
            log.exception("Acquirer stopped")
        finally:
            try:
                self.cam.stop()
            finally:
                try:
                    self.out_q.put(_STOP, timeout=10)
                except queue.Full:
                    log.error("could not signal the Processor to stop (raw_q full)")


class Processor(threading.Thread):
    """Offset, A/B split, amplitude, ROI traces, saturation, PPG metrics."""

    def __init__(self, cfg: Config, in_q: queue.Queue, view_q: Optional[queue.Queue],
                 write_q: Optional[queue.Queue], offset: OffsetMap, stop_evt: threading.Event, stats: Stats,
                 frame_period_s: Optional[float] = None):
        super().__init__(name="Processor", daemon=True)
        self.cfg, self.in_q, self.view_q, self.write_q = cfg, in_q, view_q, write_q
        self.stop_evt, self.stats = stop_evt, stats
        self.proc = BurstProcessor(cfg, offset)
        self.ppg = PPGTracker(cfg, frame_period_s)
        self.roi_lock = threading.Lock()
        self._pending_rois: Optional[dict] = None
        self.results: list = []          # light per-burst summaries (for export)
        self.error: Optional[BaseException] = None

    def set_roi(self, name: str, box):
        """Called from the viewer (main thread) when the user draws an ROI."""
        with self.roi_lock:
            rois = dict(self._pending_rois or self.proc.rois)
            rois[name] = [int(v) for v in box]
            self._pending_rois = rois

    def run(self):
        try:
            while True:
                try:
                    item = self.in_q.get(timeout=0.5)
                except queue.Empty:
                    continue          # the Acquirer always ends the stream with _STOP
                if item is _STOP:
                    break
                iq, meta = item
                if self.write_q is not None:
                    self._to_writer(iq, meta)
                with self.roi_lock:
                    if self._pending_rois is not None:
                        for n, box in self._pending_rois.items():
                            if self.proc.rois.get(n) != box:
                                self.ppg.reset_roi(n)
                        self.proc.rois = self._pending_rois
                        self._pending_rois = None
                res = self.proc.process(iq, meta)
                self.ppg.add(res)
                state = self.ppg.compute()
                with self.stats.lock:
                    st = self.stats
                    st.alternation[res.alternation.state] = st.alternation.get(res.alternation.state, 0) + 1
                    st.swapped += int(res.swapped)
                    st.last_clipped, st.last_flat = res.sat_clipped, res.sat_flat
                    st.proc_s.append(res.proc_s)
                for fl in res.flags + meta.flags:
                    log.info("burst %d: %s", res.index, fl)
                self.results.append(self._summary(res, state))
                if self.view_q is not None:
                    self._latest(self.view_q, (res, state))
        except BaseException as e:   # noqa: BLE001
            self.error = e
            with self.stats.lock:
                self.stats.errors.append(f"Processor: {e!r}")
            log.exception("Processor stopped")
            self.stop_evt.set()
        finally:
            if self.write_q is not None:
                self.write_q.put(_STOP)

    def _to_writer(self, iq, meta):
        # never block forever: if the disk cannot keep up the burst is counted as not written
        while True:
            try:
                self.write_q.put((iq, meta), timeout=0.5)
                return
            except queue.Full:
                if self.stop_evt.is_set():
                    self.stats.inc("dropped_write")
                    return

    @staticmethod
    def _latest(q: queue.Queue, item):
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass

    @staticmethod
    def _summary(res, state) -> dict:
        d = dict(index=res.index, alternation=res.alternation.state, swapped=res.swapped,
                 clipped=res.sat_clipped, flat=res.sat_flat, flags=list(res.flags))
        for n, m in state.rois.items():
            d[f"R_{n}"] = m.R
            d[f"HR_{n}"] = m.hr_bpm
        return d


class Writer(threading.Thread):
    """Raw bursts + settings to HDF5."""

    def __init__(self, writer, in_q: queue.Queue, stats: Stats):
        super().__init__(name="Writer", daemon=True)
        self.writer, self.in_q, self.stats = writer, in_q, stats
        self.error: Optional[BaseException] = None

    def run(self):
        try:
            while True:
                item = self.in_q.get()
                if item is _STOP:
                    break
                iq, meta = item
                self.writer.write_burst(iq, meta)
                with self.stats.lock:
                    self.stats.written += 1
                    self.stats.write_bytes += iq.nbytes
        except BaseException as e:   # noqa: BLE001
            self.error = e
            with self.stats.lock:
                self.stats.errors.append(f"Writer: {e!r}")
            log.exception("Writer stopped")


# ---------------------------------------------------------------------------------------
def record_offset_map(cam: C4Camera, controller, n_bursts: Optional[int] = None) -> OffsetMap:
    """I0/Q0 with the LEDs at constant current equal to their modulated mean (controller OFFSET mode).

    Redo after any change of sensitivity, coupling or temperature.
    """
    n_bursts = n_bursts or cam.cfg.offset_bursts
    controller.set_mode("offset")
    stacks = []
    cam.start()
    try:
        cam.fetch_burst()                 # discard the first burst after the mode change
        for _ in range(n_bursts):
            stacks.append(cam.fetch_burst()[0])
    finally:
        cam.stop()
    return OffsetMap.from_bursts(stacks)


@dataclass
class Session:
    """Wires camera, controller and threads; the caller runs the viewer (or waits) on the main thread."""
    cfg: Config
    cam: C4Camera
    controller: object
    offset: OffsetMap
    save_path: Optional[str] = None
    max_bursts: Optional[int] = None
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        c = self.cfg
        self.stop_evt = threading.Event()
        self.raw_q: queue.Queue = queue.Queue(maxsize=c.raw_q_max)
        self.view_q: queue.Queue = queue.Queue(maxsize=c.view_q_max)
        self.write_q: Optional[queue.Queue] = queue.Queue(maxsize=c.write_q_max) if self.save_path else None
        self.h5 = None
        if self.save_path:
            from .storage import SessionWriter
            self.h5 = SessionWriter(self.save_path, c, self.cam.actual, self.offset)
        self.acquirer = Acquirer(self.cam, self.raw_q, self.stop_evt, self.stats, self.max_bursts)
        self.processor = Processor(c, self.raw_q, self.view_q, self.write_q, self.offset, self.stop_evt,
                                   self.stats, self.cam.frame_period_s)
        self.writer = Writer(self.h5, self.write_q, self.stats) if self.h5 else None
        self.threads = [t for t in (self.acquirer, self.processor, self.writer) if t]

    def start(self):
        self.controller.set_mode("run")
        for t in self.threads:
            t.start()

    @property
    def alive(self) -> bool:
        return self.acquirer.is_alive() or self.processor.is_alive()

    def stop(self, timeout: float = 30.0) -> dict:
        """Stop acquisition, drain the queues, close the file.  Safe to call twice."""
        self.stop_evt.set()
        for t in self.threads:
            t.join(timeout=timeout)
        try:
            self.controller.set_mode("off")
        except Exception as e:     # noqa: BLE001
            log.warning("could not switch the LED controller off: %s", e)
        summary = self.stats.summary()
        if self.h5 is not None:
            self.h5.close(summary)
            self.h5 = None
        return summary

    def wait(self, duration_s: Optional[float] = None, poll_s: float = 0.2):
        """Headless: block until max_bursts/duration is reached or a worker fails."""
        t_end = None if duration_s is None else time.time() + duration_s
        while self.alive and not self.stop_evt.is_set():
            if t_end is not None and time.time() >= t_end:
                break
            time.sleep(poll_s)
