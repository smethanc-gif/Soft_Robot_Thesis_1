
import time
from pi_comm import STM32Comm

PORT          = "/dev/ttyS0"
BAUD          = 115200
DESIRED_FILE  = "DesiredPosition.txt"
CURRENT_FILE  = "CurrentPosition.txt"



def writeDesired(positions: list[float], timestamp: float):
    with open(DESIRED_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")


def writeCurrent(positions: list[float], timestamp: float):
    with open(CURRENT_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")


def clearFiles():
    open(DESIRED_FILE, "w").close()
    open(CURRENT_FILE, "w").close()


def main():
    
    stm = STM32Comm(port = PORT, baud = BAUD, timeout = 1)


    print("Starting connection sequence...\n")
    while True:
        stm.enable()
        time.sleep(1)
        stm.disable()
        time.sleep(1)
        

if __name__ == "__main__":
    main()
