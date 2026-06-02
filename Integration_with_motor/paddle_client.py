"""
paddle_client.py

Thin Python client for the Arduino paddle_server. Wraps a pyserial
connection and exposes simple methods for commanding the paddle in
either motor steps or physical meters.

Dependencies:
    pip install pyserial
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial


# ============================================================================
# PaddleState: snapshot of the motor's reported status
# ============================================================================
@dataclass
class PaddleState:
    position_steps: int   # current mechanical position, steps
    target_steps: int     # where the Arduino is trying to drive to
    busy: bool            # True if the motor is still slewing


# ============================================================================
# PaddleClient: low-level wrapper over the serial protocol
# ============================================================================
class PaddleClient:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        """
        Args:
            port:    serial device. Linux: "/dev/ttyACM0" or "/dev/ttyUSB0".
                     macOS: "/dev/tty.usbmodem*". Windows: "COM4" etc.
                     Run `python -m serial.tools.list_ports` to list.
            baud:    must match Serial.begin() in the Arduino sketch.
            timeout: read timeout in seconds; only matters for query().
        """
        self.ser = serial.Serial(port, baud, timeout=timeout)

        # The Arduino's bootloader takes ~1.5s after USB enumerates.
        # Commands sent during that window are eaten. Always wait.
        time.sleep(2.0)

        # Drain any stale output ("READY" from the Arduino's setup()).
        self.ser.reset_input_buffer()

        # Dedup cache: last goto target. We skip resends of identical
        # targets during high-rate control to keep the serial line from
        # backing up.
        self._last_target: Optional[int] = None

        # Serial writes from multiple threads need to be atomic; we use
        # a lock to protect the transport.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Private: push one line out the serial port
    # ------------------------------------------------------------------
    def _send(self, line: str) -> None:
        # pyserial writes bytes, so we encode the ASCII string. The "\n"
        # is what the Arduino's line parser triggers on.
        with self._lock:
            self.ser.write((line + "\n").encode("ascii"))

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------
    def goto_steps(self, steps: int, dedupe: bool = True) -> None:
        """Command the paddle to absolute step position `steps`.

        If dedupe=True (the default), identical consecutive targets are
        suppressed. This is almost always what you want during closed-loop
        control where the predictor may output the same value many times
        in a row.
        """
        steps = int(steps)
        if dedupe and steps == self._last_target:
            return
        self._last_target = steps
        self._send(f"G {steps}")

    def set_max_speed(self, sps: int) -> None:
        """Max speed in steps/s. Arduino clamps to its own safe range."""
        self._send(f"V {int(sps)}")

    def set_acceleration(self, sps2: int) -> None:
        """Acceleration in steps/s^2."""
        self._send(f"A {int(sps2)}")

    def stop(self) -> None:
        """Decelerate to rest via the ramp (not an e-stop)."""
        self._send("S")

    def zero_here(self) -> None:
        """Treat the current mechanical position as step 0.
        Clears the dedup cache because the position space has shifted."""
        self._send("Z")
        self._last_target = None

    def query(self, timeout: float = 0.1) -> Optional[PaddleState]:
        """Ask the Arduino for its current state. Returns None on timeout.

        Protocol reply is: "P <pos> T <tgt> B <busy>\\n"
        """
        with self._lock:
            # Drain any pending bytes first so we don't accidentally parse
            # a stale reply from a prior query.
            self.ser.reset_input_buffer()
            self.ser.write(b"?\n")

            deadline = time.time() + timeout
            line = b""
            while time.time() < deadline:
                chunk = self.ser.readline()
                if chunk:
                    line = chunk.strip()
                    break

        if not line:
            return None
        try:
            parts = line.decode("ascii").split()
            # parts should look like: ["P", "123", "T", "456", "B", "1"]
            return PaddleState(
                position_steps=int(parts[1]),
                target_steps=int(parts[3]),
                busy=(parts[5] == "1"),
            )
        except (IndexError, ValueError):
            return None

    def close(self) -> None:
        """Stop the motor and close the serial port."""
        try:
            self.stop()
        finally:
            self.ser.close()

    # Enable "with PaddleClient(...) as pc:" pattern for safety.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# RailCalibration: mapping between meters (physics world) and steps (motor)
# ============================================================================
@dataclass
class RailCalibration:
    steps_per_meter: float
    x_min_meters: float   # left soft limit, meters from mechanical zero
    x_max_meters: float   # right soft limit, meters from mechanical zero

    def meters_to_steps(self, x_m: float) -> int:
        # Clamp to soft limits so we never command out of usable range.
        x_m = max(self.x_min_meters, min(self.x_max_meters, x_m))
        return int(round(x_m * self.steps_per_meter))

    def steps_to_meters(self, steps: int) -> float:
        return steps / self.steps_per_meter


# ============================================================================
# CalibratedPaddle: convenience wrapper that accepts commands in meters
# ============================================================================
class CalibratedPaddle:
    def __init__(self, client: PaddleClient, calib: RailCalibration):
        self.client = client
        self.calib = calib

    def goto_meters(self, x_m: float) -> None:
        self.client.goto_steps(self.calib.meters_to_steps(x_m))

    def query_meters(self) -> Optional[float]:
        st = self.client.query()
        if st is None:
            return None
        return self.calib.steps_to_meters(st.position_steps)
