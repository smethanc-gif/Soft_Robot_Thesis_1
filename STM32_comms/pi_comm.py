"""
pi_comm.py — Raspberry Pi side of the Pi <-> STM32 UART link.

Protocol (both directions):
  [0xAA] [CMD] [LEN] [payload] [XOR checksum] [0x55]

Commands Pi sends:
  0x01  SET_TARGETS  — 6 pos floats + 6 vel floats (48 bytes)
  0x02  ENABLE       — no payload
  0x03  DISABLE      — no payload
  0x04  GET_STATE    — no payload; STM32 replies with CMD_STATE (0x10)

STM32 response:
  0x10  STATE        — 6 pos floats (24 bytes)

Hardware connection:
  Pi TX  (GPIO 14, pin 8)  → STM32 RX
  Pi RX  (GPIO 15, pin 10) → STM32 TX
  Common GND
  (3.3 V logic on both sides — no level shifter needed for H7 Nucleo)

Install dependency:  pip install pyserial
"""

import serial
import struct
import time

START  = 0xAA
END    = 0x55

CMD_SET_TARGETS = 0x01
CMD_ENABLE      = 0x02
CMD_DISABLE     = 0x03
CMD_GET_STATE   = 0x04
CMD_STATE       = 0x10


def _checksum(cmd: int, payload: bytes) -> int: # Calcuated on both ends, the result is compared to confirm there was no corruption
    cks = cmd ^ len(payload)
    for b in payload:
        cks ^= b
    return cks & 0xFF


def _build_packet(cmd: int, payload: bytes = b"") -> bytes: # Ech packet is: Start, Command, Length of payload, payload, the XOR check value, End
    cks = _checksum(cmd, payload)
    return bytes([START, cmd, len(payload)]) + payload + bytes([cks, END]) # 


class STM32Comm:
    def __init__(self, port: str = "/dev/ttyS0", baud: int = 115200, timeout: float = 0.1):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.1)  # let UART settle
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    # ------------------------------------------------------------------ #
    #  Commands                                                            #
    # ------------------------------------------------------------------ #

    def set_targets(self, positions: list[float], velocities: list[float]):
        """Send 6 position targets and 6 velocity targets (radians, rad/s)."""
        assert len(positions) == 6 and len(velocities) == 6
        payload = struct.pack("<6f6f", *positions, *velocities)
        self.ser.write(_build_packet(CMD_SET_TARGETS, payload))
 
    def enable(self):
        self.ser.write(_build_packet(CMD_ENABLE))

    def disable(self):
        self.ser.write(_build_packet(CMD_DISABLE))

    def get_state(self) -> list[float] | None:
        """Request current motor positions. Returns list of 6 floats or None on timeout."""
        self.ser.write(_build_packet(CMD_GET_STATE))
        return self._read_state_response()

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _read_state_response(self) -> list[float] | None:
        """
        Read one CMD_STATE packet from the STM32.
        Returns 6 floats or None on parse error / timeout.
        """
        # State machine: WAIT_START → WAIT_CMD → WAIT_LEN → PAYLOAD → WAIT_CKS → WAIT_END
        state = "WAIT_START"
        cmd = 0
        expected = 0
        buf = bytearray()
        cks_accum = 0

        deadline = time.monotonic() + self.ser.timeout
        while time.monotonic() < deadline:
            raw = self.ser.read(1)
            if not raw:
                continue
            b = raw[0]

            if state == "WAIT_START":
                if b == START:
                    state = "WAIT_CMD"

            elif state == "WAIT_CMD":
                cmd = b
                cks_accum = b
                state = "WAIT_LEN"

            elif state == "WAIT_LEN":
                expected = b
                cks_accum ^= b
                buf = bytearray()
                state = "PAYLOAD" if expected > 0 else "WAIT_CKS"

            elif state == "PAYLOAD":
                buf.append(b)
                cks_accum ^= b
                if len(buf) >= expected:
                    state = "WAIT_CKS"

            elif state == "WAIT_CKS":
                if b == (cks_accum & 0xFF):
                    state = "WAIT_END"
                else:
                    state = "WAIT_START"  # bad checksum

            elif state == "WAIT_END":
                if b == END and cmd == CMD_STATE and len(buf) == 24:
                    return list(struct.unpack("<6f", bytes(buf)))
                state = "WAIT_START"

        return None


# ------------------------------------------------------------------ #
#  Example usage                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    stm = STM32Comm(port="/dev/ttyS0", baud=115200)

    print("Enabling motors...")
    stm.enable()
    time.sleep(0.5)

    print("Sending sinusoidal targets for 5 seconds...")
    t_start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t_start
            if t > 5.0:
                break

            angle = 3.14159 * 0.5 * (1 - __import__("math").cos(2 * 3.14159 * 0.2 * t))
            positions  = [angle] * 6
            velocities = [0.0]  * 6

            stm.set_targets(positions, velocities)

            state = stm.get_state()
            if state:
                print(f"t={t:.2f}s  positions={[f'{p:.3f}' for p in state]}")

            time.sleep(0.05)  # 20 Hz

    finally:
        print("Disabling motors.")
        stm.disable()
        stm.close()
