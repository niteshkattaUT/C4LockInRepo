// led_controller.ino - two-LED frame-by-frame switching for the heliCam C4 lock-in PPG.
//
// DRAFT, NOT YET TESTED ON HARDWARE.  Same state machine as c4ppg/controller.py
// (LedControllerModel), which the simulator and the tests run edge by edge.
//
// The camera is the only clock.  Its LockInReference (RTIO3, 1 kHz square) gates both
// LEDs through an external AND gate, so both are phase-locked to the demodulation:
//
//     LED_A_drive = RTIO3 AND EN_A          LED_B_drive = RTIO3 AND EN_B
//
// This sketch only drives EN_A / EN_B:
//   * RecordingActive rising edge  -> LED A  (every burst starts on A, even frames = A)
//   * LockInFrameSync rising edge  -> arm; after SWITCH_AFTER LockInReference rising edges
//                                     toggle A <-> B (lands in the 2 blank periods)
//
// Offset mode (for the I0/Q0 maps): same A/B alternation, but the enabled LED is driven
// at its modulated mean instead of being gated.  With a 0..full square wave the mean is
// 50 %, so OFFSET_PIN_x outputs a 50 % PWM far from the reference and its odd harmonics
// (an EVEN multiple of the 1 kHz reference, e.g. 62.5 kHz on a Teensy) and the gate is
// bypassed with the BYPASS line (wire: LED_x_drive = (RTIO3 AND EN_x AND !BYPASS) OR
// (PWM AND EN_x AND BYPASS)).  VERIFY that the demodulated amplitude is ~0 in this mode.
//
// Serial protocol (115200 baud, one command per line, reply "ok <mode>"):
//   R  run (A/B alternation, gated)       O  offset mode (A/B alternation, constant mean)
//   A  LED A only, gated (build step 4)   B  LED B only, gated
//   X  all LEDs off (default at power-up) S<n> set SWITCH_AFTER   ?  status
//
// Carrier: the reference interrupt runs at the carrier frequency (1 kHz default, up to tens of kHz).
// A Teensy 4.x handles 50 kHz interrupts easily; the AND gates and LED drivers must switch cleanly at
// the carrier (check rise/fall times on the scope - slow edges cost lock-in signal).
//
// Pin numbers are for a Teensy 4.x (5 V tolerant inputs need level shifting from the
// D3's 24 V / 5 V outputs - check the heliDriver D3 data sheet for the RTIO/DIO levels).

const uint8_t PIN_FRAMESYNC = 2;     // RTIO2 = LockInFrameSync
const uint8_t PIN_REFERENCE = 3;     // RTIO3 = LockInReference (also goes to the AND gates)
const uint8_t PIN_RECACTIVE = 4;     // DIO?  = RecordingActive (TODO: choose the D3 pin)
const uint8_t PIN_EN_A = 5;
const uint8_t PIN_EN_B = 6;
const uint8_t PIN_BYPASS = 7;        // HIGH = offset mode (PWM instead of the reference gate)
const uint8_t PIN_PWM = 8;           // 50 % PWM for offset mode

volatile uint16_t switchAfter = 11;  // middle of the blank after the FrameSync edge; the PC sends S<n>
                                     // when the carrier changes (11 at 1 kHz 20+2, 51 at 5..50 kHz)
volatile bool armed = false;
volatile uint16_t count = 0;
volatile char led = 'A';
volatile char mode = 'X';            // X off, R run, O offset, A, B
volatile uint32_t frames = 0, bursts = 0;

static inline void applyOutputs() {
  bool alt = (mode == 'R' || mode == 'O');
  char lit = alt ? led : mode;
  digitalWriteFast(PIN_EN_A, lit == 'A' && mode != 'X');
  digitalWriteFast(PIN_EN_B, lit == 'B' && mode != 'X');
  digitalWriteFast(PIN_BYPASS, mode == 'O');
}

void onRecordingActive() {            // burst start
  led = 'A';
  armed = false;
  count = 0;
  bursts++;
  applyOutputs();
}

void onFrameSync() {                  // centre of each frame
  armed = true;
  count = 0;
  frames++;
}

void onReference() {                  // every reference period
  if (!armed) return;
  if (++count >= switchAfter) {
    armed = false;
    if (mode == 'R' || mode == 'O') {
      led = (led == 'A') ? 'B' : 'A';
      applyOutputs();
    }
  }
}

void setup() {
  pinMode(PIN_FRAMESYNC, INPUT);
  pinMode(PIN_REFERENCE, INPUT);
  pinMode(PIN_RECACTIVE, INPUT);
  pinMode(PIN_EN_A, OUTPUT);
  pinMode(PIN_EN_B, OUTPUT);
  pinMode(PIN_BYPASS, OUTPUT);
  analogWriteFrequency(PIN_PWM, 62500);   // even multiple of 1 kHz: rejected by the lock-in
  analogWrite(PIN_PWM, 128);
  applyOutputs();
  attachInterrupt(digitalPinToInterrupt(PIN_FRAMESYNC), onFrameSync, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_REFERENCE), onReference, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RECACTIVE), onRecordingActive, RISING);
  Serial.begin(115200);
}

void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;
  char c = cmd.charAt(0);
  noInterrupts();
  switch (c) {
    case 'R': case 'O': case 'A': case 'B': case 'X':
      mode = c; led = 'A'; armed = false; applyOutputs(); break;
    case 'S': switchAfter = (uint16_t)cmd.substring(1).toInt(); break;
    default: break;
  }
  interrupts();
  if (c == '?') {
    Serial.printf("ok %c led=%c switch_after=%u frames=%lu bursts=%lu\n", mode, led, switchAfter, frames, bursts);
  } else if (c == 'S') {
    Serial.printf("ok switch_after=%u\n", switchAfter);
  } else {
    const char *names[] = {"run", "offset", "A", "B", "off"};
    const char *n = c == 'R' ? names[0] : c == 'O' ? names[1] : c == 'A' ? names[2] : c == 'B' ? names[3]
                  : c == 'X' ? names[4] : "?";
    Serial.printf("ok %s\n", n);
  }
}
