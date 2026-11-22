# heliCam C4 two-LED real-time acquisition: code design

The script streams lock-in frames from the heliCam C4.2-S40U and alternates two LEDs
(~760 nm and ~850 nm) frame by frame. Its live outputs are two PPG traces, one per
wavelength, and their ratio. Everything below follows from the limits in the user
manual ([heliCamC4_UserManual-2025-08-28.pdf](heliCamC4_UserManual-2025-08-28.pdf)) and the
feature description (ftrDesc v1.12.0, installed at `C:\Program Files\C4Utility\doc\ftrDesc.pdf`).

## 1. Constraints

| Constraint | Value | What it means for the design |
|---|---|---|
| Acquisition unit | Bursts of 4 to 900 frames, even counts only (`AcquisitionBurstFrameCount`, Inc 2); no single-frame mode | "Real time" means bursts back to back, fetched in a loop |
| Settings | Can only be changed in configuration mode, not during acquisition | The camera cannot switch LEDs per frame. An external controller must do it. |
| Pipelining | `TransferQueueMaxBlockCount` > 1 lets the camera record the next burst while the last one transfers | Use the freerunning pattern from `c4IntensityPipelinedFreerunning.py` |
| Time constant | 1 to 100 periods (`LockInTargetTimeConstantNPeriods`) plus 0 to 100 blank periods (`LockInTargetBlankDurationNPeriods`) | Blank periods give the LEDs a dead time to switch between frames |
| Frame rate limit | f_lim = 4960 Hz in the manual's formula | Not a concern at a 1 kHz carrier |
| Image size | 542 x 512 px, fixed (Height/Width read-only, no ROI); Mono16 | 1.1 MB per frame (I and Q) in host RAM. On the wire the Heliotis calculator uses 64/3 bits per pixel at 800 Mbit/s: 7.40 ms per frame, ~108 fps maximum with its +-25 % margin (section 3a). |
| Data scaling | Raw value = `img.data % 2**15 // 4`, 10-bit, centred near 512 | Same conversion as the Heliotis examples |
| Timing metadata | `ChunkTimestamp` (ns, at FrameStart) and `ChunkFrameID` via chunk mode | Use these to detect gaps between bursts |
| Real-time outputs | `LockInReference`, `LockInFrameSync` (rising edge at the centre of each frame), `RecordingActive`, `FrameActive` on RTIO2/RTIO3; ~1 µs latency | The LED controller can follow the camera's own clock |
| Odd harmonics | Pulse-wave demodulation also responds to 3f, 5f, ... | Drive both LEDs with the same waveform so harmonics scale the same for both |

A key simplification: with each LED square-modulated from 0 to full current, the
demodulated amplitude A is proportional to that LED's reflected light. Room light is
rejected. The cardiac ripple is therefore the slow change in A, and AC/DC per
wavelength is ripple(A) / mean(A). No separate DC readout is needed for the
ratio of ratios.

## 2. Hardware timing and wiring

The camera runs its internal 1 kHz reference and is the only clock. Its
`LockInReference` output gates both LEDs, so they are phase-locked to the
demodulation by construction. A small microcontroller only decides which LED is
enabled in each frame ([firmware/led_controller/led_controller.ino](../firmware/led_controller/led_controller.ino)).

```
 heliDriver D3                         LED controller (MCU)              LED drivers
 RTIO3  LockInReference  1 kHz ──┬───► REF  (count edges)
                                 ├──────────────────────────► AND ──► LED A (760 nm)
                                 └──────────────────────────► AND ──► LED B (850 nm)
 RTIO2  LockInFrameSync  ───────────► SYNC (arm, count 11)     ▲          ▲
 DIO?   RecordingActive  ───────────► REC  (reset to A)       EN_A       EN_B
                                       USB serial ◄──► PC (R / O / A / B / X)
```

Per frame: 20 periods with one LED square-modulated, then 2 blank periods in which
the controller switches to the other LED. The controller sees one `LockInFrameSync`
rising edge per frame at the frame centre. It switches about half a frame
(11 periods, counted on `LockInReference`) after that edge, landing in the blank
periods. **Confirm where the blanks fall on a scope first.**

This replaces the 180-degree A/B scheme in the earlier notes. That scheme only
measures A minus B, so the two wavelengths cannot be separated. It also avoids the
per-pixel crosstalk calibration that quadrature multiplexing would need.

**Alternative (controller as master):** the controller makes the 1 kHz reference into
FI2 (`LockInReferenceSourceType = External`, config `reference_source: External`) and
drives the LEDs itself. Use this only if the camera's `LockInReference` turns out to
be unusable. The camera averages the external reference before each recording, and
`LockInExpectedFrequencyDeviation` costs some signal.

## 3. Camera configuration

Start with a 1 kHz internal reference and 20 demodulated periods plus 2 blank periods
per frame. That gives 22 ms frames (about 45 fps, about 23 fps per wavelength), 269 Mbit/s
on the link (34 % busy, 42 % worst case) and 50 MB/s unpacked into host RAM. All values sit in one
`Config` dataclass ([c4ppg/config.py](../c4ppg/config.py)) so they can be changed without
editing code (`--config file.yaml`, `--set key=value`).

| Feature | Start value | Config key | Why |
|---|---|---|---|
| `DeviceOperationMode` | LockInCam | - | Required for the C4 |
| `Scan3dExtractionMethod` | rawIQ | - | Raw I/Q per frame |
| `LockInReferenceSourceType` | Internal | `reference_source` | Camera is the master clock |
| `LockInTargetReferenceFrequency` | 1000 Hz | `ref_freq_hz` | `LockInActualReferenceFrequency` is logged after setting it |
| `LockInTargetTimeConstantNPeriods` | 20 | `n_periods` | 20 ms integration per frame |
| `LockInTargetBlankDurationNPeriods` | 2 | `n_blank` | 2 ms dead time for the LED switch |
| `LockInSensitivity` | 0.15 | `sensitivity` | Manual suggests ~0.2 for S40U; start lower against retinal DC |
| `LockInCoupling` | DC | `coupling` | Switch to AC only if frames saturate |
| `AcquisitionBurstFrameCount` | 100 | `burst_frames` | 2.2 s per burst, an even count of A/B pairs |
| `TriggerSelector` = FrameStart, RecordingStart, AcquisitionStart → `TriggerMode` | Off | - | Freerunning, as in the pipelined example |
| `TransferQueueMaxBlockCount` | 2 (try up to 4) | `queue_blocks` | Record the next burst during transfer |
| `LineSelector` = RTIO2 → `LineMode` Output, `LineSource` | LockInFrameSync | `frame_sync_line` | Frame clock to the LED controller |
| `LineSelector` = RTIO3 → `LineMode` Output, `LineSource` | LockInReference | `reference_line` | 1 kHz square that gates the LEDs |
| `LineSource` on a spare DIO output | RecordingActive | `recording_active_line` | Lets the controller reset to LED A at each burst start; check the D3 data sheet for the pin |
| `ChunkModeActive`; `ChunkSelector` = Timestamp, FrameID → `ChunkEnable` | True | - | Per-frame time and ID for gap detection |
| `LightControllerSelector` = LightController0 → `LightControllerSource` | Off | - | Internal LED module not used (only for the beat test) |

ftrDesc 6.7: `LineSource` requires `LineMode = Output`. `C4Camera.route()` sets it
(RTIO2/RTIO3 are fixed outputs; DIO lines default to Input).

`configure()` reads back and logs every actual value (`LockInActual*`, `FrameRate`,
`DeviceFirmwareVersion`, `SensorName`, temperatures) and stores it in each saved file.
The camera rounds targets to what its 80 MHz clock allows.

## 3a. Real-time feasibility (Heliotis C4 calculator)

`c4ppg/feasibility.py` is the Heliotis spreadsheet `C4_calculator_20240712_S4_S3.xlsx` in code (tests
reproduce every number in it), extended with the blank periods. `python -m c4ppg plan` runs it on the
current config without a camera; `check` and `live` run it too, and `live` refuses settings that cannot
stream (override with `--force`).

* Frame: `tf = (Np + n_blank + bs) / fd`, with bs = 1 for AC coupling.
* Raw I/Q transfer: `tt = 64/3 * 542 * 512 * Nf / 8e8` s = 7.40 ms per frame.
* **Streaming rule:** a pipelined burst must transfer (+25 % margin) faster than the next one is acquired.
  Both scale with Nf, so the rule is on the frame, not the burst: `tf >= 9.25 ms` (<= 108 fps).
  The burst length then only sets latency (one burst + its transfer) and RAM per burst.
* PPG: per-wavelength rate `1/(2 tf)` must be above twice the upper band edge (tf <= 50 ms for 5 Hz).
* Photon budget per LED: full well per quarter period must stay below 50 %. At 1 kHz the quarter
  period is 250 us, so only ~0.24 mW (760 nm) / ~0.36 mW (850 nm, QE extrapolated) may reach the whole
  pixel array in DC coupling, ~30x more with AC coupling. A 400-px ROI reaches PPG SNR 10 per frame
  at AC/DC 1 % with ~1.5 uW. Set `led_power_mw` once measured. The spreadsheet has no
  `LockInSensitivity` term.

| Setting | Frame | Link busy | Verdict |
|---|---|---|---|
| 1 kHz, 20 + 2 periods (default) | 22 ms | 34 % (42 % worst) | streams |
| 1 kHz, 8 + 2 periods | 10 ms | 74 % (93 % worst) | streams, no headroom |
| 10 kHz, 10 + 0 (beat test) | 1 ms | 740 % | single bursts only |

## 3b. Selectable carrier

The carrier (`ref_freq_hz`, default 1 kHz) can be changed with `--carrier 20k` on any command, in
the config, or with the **carrier selector in the live window** (presets `carrier_presets_hz`). The
frame timing follows the carrier: `n_periods` / `n_blank` / `switch_after_periods` are derived from
`integration_ms` (20) and `frame_ms` (22) within the camera's 1..100 / 0..100 period limits, unless
pinned in the config. The controller gets the new switch delay (`S<n>`).

Picking another carrier in the window ends the current run (its own HDF5 file, since every camera
setting changes), reconfigures the camera, re-records the offset map and continues. A carrier that
cannot stream is refused and the run stays at the old one.

| Carrier | Periods + blank | Integration | Frame | Streams? | Max light on sensor, 760 / 850 nm (DC coupling) |
|---|---|---|---|---|---|
| 1 kHz | 20 + 2 | 20 ms | 22 ms | yes | 0.24 / 0.36 mW |
| 2 kHz | 40 + 4 | 20 ms | 22 ms | yes | 0.47 / 0.72 mW |
| 5 kHz | 100 + 10 | 20 ms | 22 ms | yes | 1.2 / 1.8 mW |
| 10 kHz | 100 + 100 | 10 ms | 20 ms | yes | 2.4 / 3.6 mW |
| 20 kHz | 100 + 100 | 5 ms | 10 ms | yes (74 % link) | 4.7 / 7.2 mW |
| 50 kHz | 100 + 100 | 2 ms | 4 ms | link-limited: bursts <= 18 frames | 12 / 18 mW |

The limit: a frame holds at most 100 + 100 periods, and continuous streaming needs frames >= 9.25 ms,
so **continuous streaming works up to ~21.6 kHz**. Above that the camera records a burst faster than
the link can send it and then waits (`allow_link_limited: true`, `burst_frames` small enough that the
gap stays below `max_gap_ms`, e.g. 18 frames at 50 kHz: 72 ms on, ~95 ms gap, 108 fps average).
Higher carriers integrate fewer periods but allow proportionally more light (shorter quarter periods),
and move further from 1/f noise and mains flicker harmonics. The LED drivers and AND gates must
switch cleanly at the carrier.

## 4. Software architecture

Three worker threads plus the main thread, joined by queues, so a slow plot or disk
write never makes `camera.fetch()` miss a burst ([c4ppg/pipeline.py](../c4ppg/pipeline.py)).

```
             raw_q (8)                      view_q (1, newest only)
 Acquirer ─────────────► Processor ───────────────────────────► viewer (main thread)
 fetch loop  drop+count    │ offset, A/B, amplitude,
 if full                   │ ROI traces, saturation, PPG
                           └─ write_q (16) ─► Writer ─► HDF5 (raw, before processing)
```

* `raw_q` (`raw_q_max` = 8 bursts): if it fills, the Acquirer drops and counts the burst rather than blocking the camera.
* `view_q` (`view_q_max` = 1): the viewer only ever shows the newest result; older ones are skipped.
* `write_q` (`write_q_max` = 16): raw data goes to disk untouched, before any processing, so you can reprocess offline.
  *Deviation from the first draft (64):* one 100-frame burst is ~111 MB, so 64 queued bursts would need ~7 GB of RAM; 16 is ~1.8 GB.
  If the disk cannot keep up, the Processor waits for the Writer, raw_q fills and the Acquirer drops (counted), so the camera is never blocked.

Matplotlib stays on the main thread (it is not thread-safe). If the live view becomes
the bottleneck, swap [c4ppg/viewer.py](../c4ppg/viewer.py) for pyqtgraph without touching the
other threads. At about 50 MB/s raw, a session of 10 minutes is about 30 GB before
compression. Write to a local SSD (`save_dir`).

## 5. Processing pipeline

Each burst goes through six steps. Only steps 1 to 4 run on full images; the PPG
steps run on ROI means, so they stay cheap ([c4ppg/processing.py](../c4ppg/processing.py), [c4ppg/ppg.py](../c4ppg/ppg.py)).

1. **Decode.** `raw = img.data % 2**15 // 4`, reshaped to (2, N, 542, 512) as I and Q. Cast to float32.
2. **Offset correction.** Subtract the per-pixel offset maps I0, Q0. Record them at session start with the LEDs at constant current equal to their modulated mean (the manual says the offset depends on total light). Redo after any change to sensitivity or temperature.
   *Refinement:* LED A frames and LED B frames see different light, so there is one I0/Q0 pair per frame parity, recorded with the controller in OFFSET mode (same A/B alternation, constant mean current).
3. **De-interleave.** The controller resets to LED A at each burst start, so even frames are A and odd frames are B. Checked every burst: the per-frame mean amplitude must zig-zag. States: `ok`, `swapped` (whole burst started on B; swapped back and flagged if `auto_fix_swap`), `broken` (pattern changes inside the burst, e.g. a missed FrameSync edge), `indeterminate` (LEDs too similar to tell).
4. **Amplitude.** `A = sqrt(I**2 + Q**2)` per pixel and frame. Amplitude, not I alone, so the driver's phase delay does not matter.
5. **ROI traces.** Mean of A over each ROI (vessel, background fundus), one value per frame per wavelength. Timestamps come from `ChunkTimestamp`; the B frame sits 22 ms after its A frame.
6. **PPG metrics.** Over a rolling 8 to 10 s window (`ppg_window_s`): band-pass the traces 0.5 to 5 Hz, then R = (AC/DC at 760 nm) / (AC/DC at 850 nm), where DC = window mean of A and AC = RMS (default, reported as 2√2·RMS) or peak-to-peak of the filtered trace. Heart rate comes from the spectral peak.

**Saturation check per burst:** pixels whose raw I or Q is at 0 or 1023, and pixels
where both sit near 512 in every frame (the manual's sign of DC saturation under
bright light). Both counts are shown in the live view.

## 6. Code map

| Piece | File | Responsibility |
|---|---|---|
| `Config` | config.py | Every tunable value; `frame_period_s` derived from periods + blanks; `validate()` |
| `C4Camera.configure()` | camera.py | Sets the features above, routes RTIO2/RTIO3 (+ DIO), enables chunks, returns the actual values |
| `C4Camera.fetch_burst()` | camera.py | One `camera.fetch()` → (2, N, 542, 512) uint16 + `BurstMeta` |
| `record_offset_map()` | pipeline.py | Averages a few bursts with the controller in OFFSET mode → I0, Q0 per parity |
| `Acquirer` thread | pipeline.py | `camera.start()` once, then fetches bursts back to back; gap and frame-ID checks |
| `Processor` thread | pipeline.py | Offset, A/B split, amplitude, ROI traces, saturation, PPG |
| `Writer` thread | pipeline.py, storage.py | Raw bursts + settings to HDF5 (h5py, lzf) |
| `run_viewer()` | viewer.py | Matplotlib live view on the main thread, ROI drawing |
| LED controller | controller.py, firmware/ | State machine (Python model + Arduino/Teensy sketch), serial link |
| Simulator | sim.py | Spec-checked fake Harvesters node map + fundus phantom with known R |
| Offline | offline.py | Reprocess an HDF5 session; CSV + PNG exports |

The core loop follows the Heliotis pipelined example:

```python
cam.start()                      # once
while running:
    iq, meta = cam.fetch_burst()  # blocks until the next burst arrives
    raw_q.put((iq, meta))
cam.stop()
```

## 7. Build order

Each step has a pass test before you move on. Steps marked (sim) already pass in the simulator.

- [ ] **Connect and configure** - `python -m c4ppg check`. Pass: reference about 1000 Hz, 20 + 2 periods, FrameRate about 45. (sim)
- [ ] **Single burst, internal LED module** - `python -m c4ppg beat --plot` (reproduces `c4DemodSimple.py`). Pass: the 25 Hz beat appears in I/Q. (sim)
- [ ] **Scope the outputs.** Probe RTIO2 and RTIO3 during `python -m c4ppg live --no-view --no-save --no-offset`. Pass: 1 kHz on RTIO3, one FrameSync edge every 22 ms on RTIO2. Note where the blank periods fall relative to the edge; adjust `switch_after_periods` (and the firmware `S<n>`).
- [ ] **One LED, no alternation.** Controller command `A`. Pass: steady amplitude, no saturation at sensitivity 0.15.
- [ ] **Alternation.** Controller `R`. Pass: ROI amplitude alternates high/low on a target lit by LED A only (live view shows `alternation ok`).
- [ ] **Pipelined streaming** - `python -m c4ppg live --no-view --no-save --duration 300`. Pass: no timeouts; the gap between bursts from ChunkTimestamp is measured and logged (summary `gap_ms_mean/max`). (sim)
- [ ] **Phantom PPG.** A pulsatile flow phantom or a finger. Pass: about 1 Hz ripple in both traces, R stable over a minute. (sim: R recovered within 12 % of the phantom's known value)

## 8. Open questions (ideally for Heliotis)

* How does Harvesters expose `ChunkTimestamp` per frame inside a multi-part burst, or only per burst? (Code assumes per burst = first frame, and derives the others from the frame period; `BurstMeta.frame_t_ns` is ready for per-frame values. `ChunkPartSelector`/`ChunkPartFrameNo` exist in ftrDesc and may help.)
* Where do blank periods sit within a frame relative to the `LockInFrameSync` edge?
* Is `LockInReference` output only while recording? If so, the LEDs are dark in any gap between bursts, which also helps eye exposure.
* With `TransferQueueMaxBlockCount` 2 to 4, how long is the gap between bursts at about 50 MB/s?
* Which D3 DIO output pin is best for `RecordingActive`?
* `ChunkFrameID` wrap-around value (used to detect lost frames).

## 9. Safety

Before any human retinal use, check the combined LED irradiance against the ocular
exposure limits (ANSI Z136 / ISO 15004-2) for both wavelengths.
