"""HDF5 session files: raw bursts exactly as fetched (decoded 10-bit I/Q), before any processing.

Layout of ``c4ppg_YYYYmmdd_HHMMSS.h5``::

    attrs: config_json, actual_json, created, software
    /offset/I0, /offset/Q0          (2, H, W) float32, [0] = even (LED A) frames, [1] = odd (LED B)
    /bursts/burst_00000              (2, N, H, W) uint16  I/Q,  attrs: meta_json
    /bursts/burst_00001 ...
    attrs at close: n_bursts, dropped_bursts, stats_json

Reprocess offline with ``python -m c4ppg reprocess FILE.h5``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional

import numpy as np

from . import __version__
from .camera import BurstMeta
from .config import Config
from .processing import OffsetMap


def _json(o) -> str:
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    return json.dumps(o, default=default)


def session_path(save_dir: str, prefix: str = "c4ppg") -> str:
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, time.strftime(f"{prefix}_%Y%m%d_%H%M%S.h5"))


class SessionWriter:
    def __init__(self, path: str, cfg: Config, actual: dict, offset: Optional[OffsetMap] = None):
        import h5py
        self.path = path
        self.cfg = cfg
        self.f = h5py.File(path, "w")
        self.f.attrs["config_json"] = _json(cfg.to_dict())
        self.f.attrs["actual_json"] = _json(actual)
        self.f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.f.attrs["software"] = f"c4ppg {__version__}"
        for k, v in actual.items():           # also flat, so HDFView shows them directly
            if isinstance(v, (int, float, str)):
                self.f.attrs[f"actual_{k}"] = v
        self.g = self.f.create_group("bursts")
        self.n = 0
        self.bytes = 0
        if offset is not None:
            self.write_offset(offset)

    def write_offset(self, offset: OffsetMap):
        g = self.f.require_group("offset")
        for name, arr in (("I0", offset.I0), ("Q0", offset.Q0)):
            if name in g:
                del g[name]
            g.create_dataset(name, data=arr.astype(np.float32))
        g.attrs["n_frames"] = offset.n_frames

    def write_burst(self, iq: np.ndarray, meta: BurstMeta):
        chunks = (2, 1, *iq.shape[2:])
        d = self.g.create_dataset(f"burst_{self.n:05d}", data=iq, chunks=chunks, compression=self.cfg.compression)
        d.attrs["meta_json"] = _json(meta.to_dict())
        self.n += 1
        self.bytes += iq.nbytes

    def close(self, stats: Optional[dict] = None):
        if self.f:
            self.f.attrs["n_bursts"] = self.n
            if stats is not None:
                self.f.attrs["stats_json"] = _json(stats)
            self.f.close()
            self.f = None


class SessionReader:
    def __init__(self, path: str):
        import h5py
        self.path = path
        self.f = h5py.File(path, "r")
        self.cfg = Config.from_dict(json.loads(self.f.attrs["config_json"]))
        self.actual = json.loads(self.f.attrs["actual_json"])
        self.names = sorted(self.f["bursts"].keys()) if "bursts" in self.f else []

    def offset(self) -> Optional[OffsetMap]:
        if "offset" not in self.f:
            return None
        g = self.f["offset"]
        return OffsetMap(g["I0"][()], g["Q0"][()], int(g.attrs.get("n_frames", 0)))

    def __len__(self):
        return len(self.names)

    def bursts(self) -> Iterator[tuple[np.ndarray, BurstMeta]]:
        for n in self.names:
            d = self.f["bursts"][n]
            m = json.loads(d.attrs["meta_json"])
            yield d[()], BurstMeta(**m)

    def stats(self) -> dict:
        s = self.f.attrs.get("stats_json")
        return json.loads(s) if s else {}

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
