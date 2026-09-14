"""
cable_motor_traj.py - Graphs cable lengths and motor positions over the
Cartesian tool trajectory, mirroring traj_gen.py.

Steps the ContinuumRobot through the same sinusoidal tooltip trajectory as
traj_gen.py via FullIK, and plots the resulting cable lengths (mm) and
motor positions (shaft radians) against time, one 6-panel figure each -
channels A-F, matching the la/lb/lc/ld/le/lf order PCC_Control.py uses
internally (motor i drives cable i).

Samples are taken at fixed, evenly-spaced synthetic time steps rather than
timed off the wall clock like traj_gen.py's logging loop, so the plot is
deterministic and independent of how fast this machine runs.

Install dependency:
  pip install matplotlib
"""
import math
from math import sin as S
from math import cos as C
from math import pi as PI

import matplotlib.pyplot as plt
from PCC_control import ContinuumRobot

N_SAMPLES = 400     # points sampled across the trajectory
DURATION = 0.2      # s, matches traj_gen.py's t_now >= 0.2 cutoff
ACC = 200           # matches traj_gen.py's acc

CHANNEL_NAMES = ["A", "B", "C", "D", "E", "F"]


def trajectoryTarget(t_now):
    """Same tooltip trajectory as traj_gen.py, at wall-clock time t_now (s)."""
    const_1 = PI * 0.2 * ACC
    x   , y   , z   = 100* S(const_1*t_now) *S(const_1*t_now/8), 100*C(const_1*t_now)*S(const_1*t_now/8), 290
    Rx  , Ry  , Rz  = 0, 0, 0
    return [x, y, z, Rx, Ry, Rz]


def runTrajectory():
    robot = ContinuumRobot()
    times = []
    cables = [[] for _ in range(6)]
    motors = [[] for _ in range(6)]
    limitHits = 0
    for i in range(N_SAMPLES):
        t_now = DURATION * i / (N_SAMPLES - 1)
        target = trajectoryTarget(t_now)
        motorPos, _ = robot.FullIK(target, [0.0] * 6)
        cablePos = robot.getCablePos()
        if robot.limitsHit:
            limitHits += 1
        times.append(t_now * ACC)
        for j in range(6):
            cables[j].append(cablePos[j])
            motors[j].append(motorPos[j])
    if limitHits:
        print(f"Note: {limitHits}/{N_SAMPLES} samples were clamped by "
              f"_applyLimits (target outside sMin/sMax or thetaMax).")
    return times, cables, motors


def plotChannels(times, values, namePrefix, title, ylabel):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        ax.plot(times, values[i], color="tab:blue")
        ax.set_title(f"{namePrefix} {CHANNEL_NAMES[i]}")
        ax.set_xlabel("Time (t_now * acc)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    fig.suptitle(title)
    fig.tight_layout()


def main():
    times, cables, motors = runTrajectory()
    plotChannels(times, cables, "Cable", "Cable Lengths over Trajectory", "Length (mm)")
    plotChannels(times, motors, "Motor", "Motor Positions over Trajectory", "Position (rad)")
    plt.show()


if __name__ == "__main__":
    main()
