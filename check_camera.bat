@echo off
rem Build step 1: configure the camera and check the actual values. Add --sim to try it.
"%~dp0.venv\Scripts\python.exe" -m c4ppg check %*
