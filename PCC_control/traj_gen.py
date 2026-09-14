
import math
from math import sin as S
from math import cos as C
from math import pi as PI
import time
import traj_graph
from PCC_control import ContinuumRobot as control1

DESIRED_FILE = "TargetPosition.txt"
MIDPOINT_FILE = "MidPointPosition.txt"
CALCULATED_FILE = "CalculatedPosition.txt"


def write_desired(positions: list[float], timestamp: float):
    with open(DESIRED_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")


def write_current(positions: list[float], timestamp: float):
    with open(CALCULATED_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def write_mid(positions: list[float], timestamp: float):
    with open(MIDPOINT_FILE, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def clear_files():
    open(DESIRED_FILE, "w").close()
    open(CALCULATED_FILE, "w").close()
    open(MIDPOINT_FILE, "w").close()


def main():
    clear_files()
    t_start = time.monotonic()
    robot = control1()
    acc = 200
    while True:
        t_now = time.monotonic() - t_start
        const_1 = PI*0.2 * acc
        x   , y   , z   = 200, 35*S(const_1*t_now), 180 + 25*C(const_1*t_now) 
        Rx  , Ry  , Rz  = 0,  PI/2, 0
        target = [x,y,z,Rx,Ry,Rz] 
        thing = robot.toolTransform(target)
        robot.setTarget(thing, [0]*6)
        write_desired(target, t_now*acc)

        mid = robot.midpointCalc()
        write_mid(mid, t_now*acc)
        robot.PCC_Control()
        calced = robot.forwardKinematics()
        write_current(calced,t_now*acc)
        if t_now >= 0.2:
            return

        
if __name__ == "__main__":
    main()
    print("Complete")
    traj_graph.main()
