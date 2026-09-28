'''Connection test for the Pi <-> STM32 link: toggles the motors on and off every second, forever
(watch the STM32 side in the debugger). Ctrl+C to stop.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, so this file can be run directly
import time
from STM32_comms.piComm import STM32Comm

PORT          = "/dev/ttyS0"
BAUD          = 115200
DESIRED_FILE  = "DesiredPosition.txt"
CURRENT_FILE  = "CurrentPosition.txt"



def writeDesired(_positions: list[float], _timestamp: float):
    '''Appends one row of desired motor positions to DESIRED_FILE.'''
    with open(DESIRED_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in _positions)
        f.write(f"t={_timestamp:.3f}s  [{vals}]\n")


def writeCurrent(_positions: list[float], _timestamp: float):
    '''Appends one row of measured motor positions to CURRENT_FILE.'''
    with open(CURRENT_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in _positions)
        f.write(f"t={_timestamp:.3f}s  [{vals}]\n")


def clearFiles():
    '''Empties both log files.'''
    open(DESIRED_FILE, "w").close()
    open(CURRENT_FILE, "w").close()


def main():
    '''Enables and disables the motors once a second, forever.'''
    
    stm = STM32Comm(_port = PORT, _baud = BAUD, _timeout = 1)


    print("Starting connection sequence...\n")
    while True:
        stm.enable()
        time.sleep(1)
        stm.disable()
        time.sleep(1)
        

if __name__ == "__main__":
    main()
