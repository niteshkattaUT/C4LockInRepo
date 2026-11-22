"""LED controller: the small microcontroller that decides which LED is enabled per frame.

The camera is the only clock.  Its LockInReference output (RTIO3) gates both LEDs
through an AND gate, so both are phase-locked to the demodulation by construction.
The controller only drives the two enable lines EN_A / EN_B:

* RecordingActive rising edge  -> select LED A (so every burst starts on A)
* LockInFrameSync rising edge  -> arm; count ``switch_after`` LockInReference edges,
                                  then toggle A <-> B.  This lands in the blank periods.

``LedControllerModel`` is the same state machine as ``firmware/led_controller/led_controller.ino``.
The simulator runs it edge by edge, so the Python pipeline is tested against the
exact switching rule the firmware uses.

The PC talks to the controller with one-character commands (see the firmware):
``R`` run (A/B alternation), ``O`` offset mode, ``A``/``B`` one LED only, ``X`` all off.
"""
from __future__ import annotations

import time
from typing import Optional

MODES = ("off", "run", "offset", "A", "B")
_CMD = {"run": "R", "offset": "O", "A": "A", "B": "B", "off": "X"}


class LedControllerModel:
    """Edge-driven model of the firmware state machine."""

    def __init__(self, switch_after: int = 11):
        self.switch_after = switch_after
        self.mode = "off"
        self.led = "A"
        self.armed = False
        self.count = 0
        self.miss_next_reset = False   # fault injection: ignore the next RecordingActive edge
        self.miss_next_sync = False    # fault injection: ignore the next FrameSync edge

    # --- inputs --------------------------------------------------------------------
    def on_recording_active(self):
        if self.miss_next_reset:
            self.miss_next_reset = False
            return
        self.led, self.armed, self.count = "A", False, 0

    def on_frame_sync(self):
        if self.miss_next_sync:
            self.miss_next_sync = False
            return
        self.armed, self.count = True, 0

    def on_reference_edge(self):
        if not self.armed:
            return
        self.count += 1
        if self.count >= self.switch_after:
            self.armed = False
            if self.mode in ("run", "offset"):
                self.led = "B" if self.led == "A" else "A"

    # --- outputs ---------------------------------------------------------------------
    def enabled(self) -> Optional[str]:
        """Which LED is lit right now (None = dark)."""
        if self.mode in ("run", "offset"):
            return self.led
        if self.mode in ("A", "B"):
            return self.mode
        return None

    @property
    def modulated(self) -> bool:
        """True if the lit LED is gated by LockInReference (False in offset mode: constant mean current)."""
        return self.mode in ("run", "A", "B")

    def simulate_frame(self, n_periods: int, n_blank: int) -> tuple[Optional[str], bool]:
        """Advance one camera frame; return (LED lit during the demodulation window, modulated).

        Timeline of one frame (VERIFY on the scope where the blanks really sit):
        periods 0..n_periods-1 demodulate, FrameSync rises at the centre of the frame,
        then the blank periods follow.
        """
        lit, mod = self.enabled(), self.modulated
        total = n_periods + n_blank
        centre = total // 2
        for p in range(total):
            if p == centre:
                self.on_frame_sync()
            self.on_reference_edge()
        return lit, mod


# ---------------------------------------------------------------------------------------
# Links from the PC to the controller
# ---------------------------------------------------------------------------------------
class ManualController:
    """No serial link: ask the operator to set the controller mode by hand."""

    def __init__(self, prompt=input):
        self.prompt = prompt
        self.mode = None

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != self.mode:
            self.prompt(f"Set the LED controller to {mode.upper()} mode, then press Enter ")
            self.mode = mode

    def close(self):
        if self.mode not in (None, "off"):
            print("Reminder: switch the LED controller OFF.")


class SerialController:
    """Talks to the firmware over USB serial (one-character commands, echo 'ok <mode>')."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)          # most Arduino-type boards reset when the port opens
        self.ser.reset_input_buffer()
        self.mode = None

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.ser.write((_CMD[mode] + "\n").encode())
        reply = self.ser.readline().decode(errors="replace").strip()
        if not reply.startswith("ok"):
            raise RuntimeError(f"LED controller did not acknowledge {mode!r}: {reply!r}")
        self.mode = mode

    def close(self):
        try:
            self.set_mode("off")
        finally:
            self.ser.close()


class SimController:
    """Forwards the mode to the simulated camera's controller model."""

    def __init__(self, model: LedControllerModel):
        self.model = model

    @property
    def mode(self):
        return self.model.mode

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.model.mode = mode

    def close(self):
        self.model.mode = "off"


def make_controller(spec: str, baud: int = 115200, sim_model: Optional[LedControllerModel] = None):
    if spec == "sim":
        if sim_model is None:
            raise ValueError("controller 'sim' needs the simulated camera")
        return SimController(sim_model)
    if spec.startswith("serial:"):
        return SerialController(spec.split(":", 1)[1], baud)
    if spec == "manual":
        return ManualController()
    raise ValueError(f"Unknown controller {spec!r} (manual | serial:COMx | sim)")
