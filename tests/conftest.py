import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4ppg.config import Config  # noqa: E402


@pytest.fixture
def small_cfg(tmp_path):
    """Fast simulated setup: 96 x 96 px, no real-time waits."""
    return Config(
        sim_shape=(96, 96), sim_time_scale=0.0, burst_frames=40,
        rois={"vessel": [44, 52, 30, 60], "background": [6, 30, 56, 88]},
        save_dir=str(tmp_path), controller="sim", ppg_window_s=8.0,
    )


@pytest.fixture
def sim_cam(small_cfg):
    from c4ppg.camera import open_camera
    cam = open_camera(small_cfg, sim=True)
    cam.configure()
    yield cam
    cam.close()
