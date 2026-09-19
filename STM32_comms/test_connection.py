"""
test_connection.py — Connection test script for Pi <-> STM32 UART link.

What this does:
  1. Sends a series of fixed position targets so you can confirm receipt
     on the STM32 side via live expressions in the debugger.
  2. After each command, requests state back and writes it to CurrentPosition.txt.
  3. Every sent target is written to DesiredPosition.txt.
  4. Press 'e' on the keyboard to toggle motor enable/disable.

Install dependencies:
  pip install pyserial
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Continuum Robot Control root

import time
import math
import sys
import tty
import termios
import select
from pi_comm import STM32Comm
from PCC_control.CTRL import ContinuumRobot as CR

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

PORT          = "/dev/ttyS0"
BAUD          = 115200
DESIRED_FILE  = "DesiredPosition.txt"
CURRENT_FILE  = "CurrentPosition.txt"

# Test sequence: list of (position_rad, hold_time_seconds)
# All 6 motors get the same angle for simplicity.
TEST_SEQUENCE = [
    (0.0,              2.0),   # hold at 0 deg
    (math.pi / 2,      2.0),   # hold at 90 deg
    (math.pi,          2.0),   # hold at 180 deg
    (math.pi / 2,      2.0),   # back to 90 deg
    (0.0,              2.0),   # back to 0 deg
]


# ------------------------------------------------------------------ #
#  Non-blocking keyboard input                                         #
# ------------------------------------------------------------------ #

class RawKeyboard:
    """
    Puts the terminal in raw mode so individual key presses are readable
    without needing to press Enter. Restores normal mode on exit.

    Usage:
        with RawKeyboard() as kb:
            key = kb.read()   # returns a character or None if no key pressed
    """
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read(self) -> str | None:
        """Return the pressed key character, or None if no key is waiting."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ------------------------------------------------------------------ #
#  File helpers                                                        #
# ------------------------------------------------------------------ #

def writeDesired(positions: list[float], timestamp: float):
    with open(DESIRED_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")


def writeCurrent(positions: list[float], timestamp: float):
    with open(CURRENT_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")


def clear_files():
    open(DESIRED_FILE, "w").close()
    open(CURRENT_FILE, "w").close()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    clear_files()

    stm = STM32Comm(port=PORT, baud=BAUD, timeout=1)
    t_start = time.monotonic()
    motors_enabled = False
    t_end = t_start + 20
    const_1 = math.pi*2*2/20
    robot = CR()
    print("Starting connection test sequence...\n")

    try:

        while time.monotonic() <= t_end: # 6 Seconds time limit        
            t_now = time.monotonic() - t_start
            
            #positions  = [math.pi *2* (1 + math.sin(math.pi*t_now))/3] *6 #[5,0,3,0,5,0]
            #positions = [t_now]*6
            x   , y   , z   = 100* math.sin(const_1*t_now) *math.sin(const_1*t_now/8), 100*math.cos(const_1*t_now)*math.sin(const_1*t_now/8), 290
            Rx  , Ry  , Rz  = 0, 0, 0
            targGlobPos = [x, y, z, Rx, Ry, Rz]

            targGlobVel = [0]  * 6
            positions, velocities  = robot.FullIK(targGlobPos, targGlobVel)
            if not motors_enabled:
                stm.enable()
                            
                    # --- Send target --- #
            stm.set_targets(positions, velocities)
            writeDesired(positions, t_now)

                    # --- Request and log state --- #
            state = stm.get_state()
            if state:
                writeCurrent(state, t_now)
                print(f"  t={t_now:.2f}s  desired={positions[1]:.3f}  "
                              f"actual=[{', '.join(f'{p:.3f}' for p in state)}]")
            else:
                print(f"  t={t_now:.2f}s  no response from STM32")

            time.sleep(1/30)  # 30 Hz polling        

        print("\nTest sequence complete.")
        print(f"Desired positions logged to: {DESIRED_FILE}")
        print(f"Current positions logged to: {CURRENT_FILE}")    

    finally:
        print("Disabling motors and cleaning up.")
        endPos = positions
        for i in range(1000):
            stm.set_targets([x *(1 - i/1000) for x in endPos], [0]*6)
            time.sleep(0.001)
        stm.set_targets([0.0]*6, [0.0]*6)
        time.sleep(1)
        stm.disable()
        stm.close()


if __name__ == "__main__":
    main()
