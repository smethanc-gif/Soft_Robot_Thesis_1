"""
3D visualization of a Piecewise Constant Curvature (PCC) continuum robot backbone.

Mirrors the robot parameters in ContinuumRobot.py:
    nodes         = 10    # points sampled per section
    sections      = 2
    robotDiameter = 20    # mm
    sectL         = 143   # mm

Each section is parameterized by:
    theta  -> total bend angle over the section (rad), 0 = straight
    phi    -> bending-plane direction about the local z-axis (rad)
    length -> arc length of the section (mm)

Sections are chained by carrying the tip pose (position + orientation) of
each section forward as the base frame for the next one, so the whole
backbone stays continuous.

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import time
import math
from math import sin as S
from math import cos as C
from math import pi as PI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
from Core.continuumRobot import ContinuumRobot as control1
from Core.settings import loadRobotGeometry
from main import MOVEMENT_FUNCTIONS, setupTime, taskTime, packupTime, trajectory
from controller.ctrlInput import Controller

# ---- Robot parameters (from ContinuumRobot.py) ----
NODES = 10
SECTIONS = 2
ROBOT_DIAMETER = 20   # mm
SECT_LENGTH = loadRobotGeometry()["sectL"]   # mm
CHECK = False
Robot = control1()
t_start = time.monotonic()

def pcc_section_curve(theta, phi, length, n_points, T0):
    """
    Backbone points for one constant-curvature section.

    theta    : total bend angle over the section (rad)
    phi      : bending-plane direction about local z-axis (rad)
    length   : arc length of the section (mm)
    n_points : number of samples along the section
    T0       : 4x4 homogeneous transform (world <- section base frame)

    Returns (world_pts, T_end) where world_pts is (n_points, 3) and T_end
    is the 4x4 transform of the section tip (world <- tip frame).
    """
    s = np.linspace(0, length, n_points)

    if abs(theta) < 1e-9:
        local_pts = np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
        T_tip = np.eye(4)
        T_tip[2, 3] = length
    else:
        kappa = theta / length
        x = (np.cos(phi) * (1 - np.cos(kappa * s))) / kappa
        y = (np.sin(phi) * (1 - np.cos(kappa * s))) / kappa
        z = np.sin(kappa * s) / kappa
        local_pts = np.stack([x, y, z], axis=1)

        # Bend is a rotation by theta about the axis perpendicular to phi in
        # the XY plane: R = Rz(phi) * Ry(theta) * Rz(-phi)
        ct, st = np.cos(theta), np.sin(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)
        Rz_phi = np.array([[cphi, -sphi, 0], [sphi, cphi, 0], [0, 0, 1]])
        Ry_theta = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R_tip = Rz_phi @ Ry_theta @ Rz_phi.T

        T_tip = np.eye(4)
        T_tip[:3, :3] = R_tip
        T_tip[:3, 3] = local_pts[-1]  # tip position = last sampled point

    homog_pts = np.hstack([local_pts, np.ones((n_points, 1))])
    world_pts = (T0 @ homog_pts.T).T[:, :3]
    T_end = T0 @ T_tip
    #print(T_end, end = "\n")

    return world_pts, T_end


def robot_backbone(section_params, n_points=NODES):
    """
    section_params: list of (theta, phi, length) tuples, one per section.
    Returns the concatenated (N, 3) backbone in world coordinates.
    """
    T = np.eye(4)
    all_points = []
    for theta, phi, length in section_params:
        pts, T = pcc_section_curve(theta, phi, length, n_points, T)
        all_points.append(pts)
    return np.vstack(all_points)


def plot_static():
    section_params = [
        (np.radians(40), np.radians(30), SECT_LENGTH),  # section 1: bent
        (np.radians(0), np.radians(0), SECT_LENGTH),     # section 2: straight
    ]
    backbone = robot_backbone(section_params)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(backbone[:, 0], backbone[:, 1], backbone[:, 2],
             color="tab:blue", linewidth=3, marker="o", markersize=3)
    ax.scatter(0, 0, 0, color="black", s=40, label="base")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("PCC Continuum Robot Backbone (2 sections)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate(use_controller: bool = False):
    """Sweep section-1 bending sinusoidally, similar to MOVEMENT_FUNCTIONS().

        use_controller=False (default): replays main.py's MOVEMENT_FUNCTIONS
            trajectory, cycling init -> task -> packup -> repeat.
        use_controller=True: drives the sim live from a real Xbox controller
            via controller.ctrlInput.Controller instead - needs a controller
            actually plugged in (constructing Controller() opens joystick 0),
            so this path isn't exercised by anything that runs without one."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], color="tab:blue", linewidth=3, marker="o", markersize=3)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 2 * SECT_LENGTH)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("PCC Backbone Animation" + (" (controller input)" if use_controller else ""))

    controller = Controller() if use_controller else None

    # Same phase shape as main.py's CTRL_INIT -> task loop -> finally-block
    # packup, but driven in Cartesian space (rather than ego/motor space,
    # like main.py does) since this simulator only ever talks to Robot via
    # Cartesian targets -> FullIK. "Rest" is the fully-straight pose, the
    # Cartesian analogue of main.py's ego=0.
    restPose = np.array([0.0, 0.0, 2*Robot.sectL, 0.0, 0.0, 0.0])
    taskStartPose = np.array(MOVEMENT_FUNCTIONS(0.0, trajectory)[0])
    taskEndPose = np.array(MOVEMENT_FUNCTIONS(taskTime, trajectory)[0])
    cycleLength = setupTime + taskTime + packupTime

    def targetForCycleTime(t_cycle: float) -> list[float]:
        if t_cycle < setupTime:                                  # CTRL_INIT-equivalent ramp-in
            frac = t_cycle / setupTime
            return (restPose + frac*(taskStartPose - restPose)).tolist()
        elif t_cycle < setupTime + taskTime:                      # main trajectory
            pos, _ = MOVEMENT_FUNCTIONS(t_cycle - setupTime, trajectory)
            return pos
        else:                                                     # packup-equivalent ramp-down
            frac = (t_cycle - setupTime - taskTime) / packupTime
            return (taskEndPose + frac*(restPose - taskEndPose)).tolist()

    def update(frame):
        if controller is not None:
            controller.update()
            target = controller.process().tolist()
        else:
            t_now = time.monotonic() - t_start
            t_cycle = t_now % cycleLength                         # loop forever until the window closes
            target = targetForCycleTime(t_cycle)

        Robot.FullIK(target, [0] * 6)

        k1, r1, s1, k2, r2, s2 = Robot.targCylPos

        theta1 = s1*k1     # Amount of Bend
        theta2 = s2*k2
        sectionParams = [
            (theta1, r1, s1),
            (theta2, r2, s2),
        ]

        backbone = robot_backbone(sectionParams)
        line.set_data(backbone[:, 0], backbone[:, 1])
        line.set_3d_properties(backbone[:, 2])
        return line,

    ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    # plot_static()
    print("Reading input...\n")
    animate(use_controller=True)  # flip to True to drive the sim live from an Xbox controller instead
