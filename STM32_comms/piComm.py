'''Raspberry Pi side of the Pi <-> STM32 UART link.

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
'''

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


def _calcChecksum(_cmd: int, _payload: bytes) -> int:
    '''XOR checksum of a packet, calculated on both ends and compared to confirm there was no corruption.'''
    cks = _cmd ^ len(_payload)
    for b in _payload:
        cks ^= b
    return cks & 0xFF


def _buildPacket(_cmd: int, _payload: bytes = b"") -> bytes:
    '''One packet: start, command, payload length, payload, XOR checksum, end.'''
    cks = _calcChecksum(_cmd, _payload)
    return bytes([START, _cmd, len(_payload)]) + _payload + bytes([cks, END])


class STM32Comm:
    def __init__(self, _port: str = "/dev/ttyS0", _baud: int = 115200, _timeout: float = 0.1):
        '''Opens the UART link to the STM32 (_timeout = read timeout in s).'''
        self.ser = serial.Serial(_port, _baud, timeout=_timeout)
        time.sleep(0.1)  # let UART settle
        self.ser.reset_input_buffer()

    def close(self):
        '''Closes the UART link.'''
        self.ser.close()

    # ------------------------------------------------------------------ #
    #  Commands                                                            #
    # ------------------------------------------------------------------ #

    def setTargets(self, _positions: list[float], _velocities: list[float]):
        '''Send 6 position targets and 6 velocity targets (radians, rad/s).'''
        assert len(_positions) == 6 and len(_velocities) == 6
        payload = struct.pack("<6f6f", *_positions, *_velocities)
        self.ser.write(_buildPacket(CMD_SET_TARGETS, payload))
 
    def enable(self):
        '''Enables the motors.'''
        self.ser.write(_buildPacket(CMD_ENABLE))

    def disable(self):
        '''Disables the motors.'''
        self.ser.write(_buildPacket(CMD_DISABLE))

    def getState(self) -> list[float] | None:
        '''Request current motor positions. Returns list of 6 floats or None on timeout.'''
        self.ser.write(_buildPacket(CMD_GET_STATE))
        return self._readStateResponse()

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _readStateResponse(self) -> list[float] | None:
        '''
        Read one CMD_STATE packet from the STM32.
        Returns 6 floats or None on parse error / timeout.
        '''
        # State machine: WAIT_START → WAIT_CMD → WAIT_LEN → PAYLOAD → WAIT_CKS → WAIT_END
        state = "WAIT_START"
        cmd = 0
        expected = 0
        buf = bytearray()
        cksAccum = 0

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
                cksAccum = b
                state = "WAIT_LEN"

            elif state == "WAIT_LEN":
                expected = b
                cksAccum ^= b
                buf = bytearray()
                state = "PAYLOAD" if expected > 0 else "WAIT_CKS"

            elif state == "PAYLOAD":
                buf.append(b)
                cksAccum ^= b
                if len(buf) >= expected:
                    state = "WAIT_CKS"

            elif state == "WAIT_CKS":
                if b == (cksAccum & 0xFF):
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
    stm = STM32Comm(_port="/dev/ttyS0", _baud=115200)

    print("Enabling motors...")
    stm.enable()
    time.sleep(0.5)

    print("Sending sinusoidal targets for 5 seconds...")
    tStart = time.monotonic()
    try:
        while True:
            t = time.monotonic() - tStart
            if t > 5.0:
                break

            angle = 3.14159 * 0.5 * (1 - __import__("math").cos(2 * 3.14159 * 0.2 * t))
            positions  = [angle] * 6
            velocities = [0.0]  * 6

            stm.setTargets(positions, velocities)

            state = stm.getState()
            if state:
                print(f"t={t:.2f}s  positions={[f'{p:.3f}' for p in state]}")

            time.sleep(0.05)  # 20 Hz

    finally:
        print("Disabling motors.")
        stm.disable()
        stm.close()
