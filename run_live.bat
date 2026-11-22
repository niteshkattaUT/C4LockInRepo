@echo off
rem Live two-LED PPG. Add --sim to run without the camera.
"%~dp0.venv\Scripts\python.exe" -m c4ppg live %*
