'''3D visualisation of the continuum robot's PCC backbone.

Each section is parameterised by:
    theta  -> total bend angle over the section (rad), 0 = straight
    phi    -> bending-plane direction about the local z-axis (rad)
    length -> arc length of the section (mm)

Sections are chained by carrying the tip pose (position + orientation) of
each section forward as the base frame for the next one, so the whole
backbone stays continuous. animateRobot drives it through the PCC IK, either
along the trajectory in Core/controlSettings.json or live from the gamepad.

Run:  python Core/testing/robotSim.py
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from Core.continuumRobot import ContinuumRobot
from Core.settings import loadRobotGeometry, loadControlSettings
from Core.trajectories import calcTargGlob, taskTime, trajectory
from controller.ctrlInput import Controller

POINTS_PER_SECTION = 10                         # backbone points drawn per section
SECT_LENGTH = loadRobotGeometry()["sectL"]      # mm
setupTime = loadControlSettings()["setupTime"]
packupTime = loadControlSettings()["packupTime"]
robot = ContinuumRobot()
tStart = time.monotonic()


def calcSectionCurve(_theta, _phi, _length, _nPoints, _T0):
    '''Backbone points for one constant-curvature section.

        _theta   : total bend angle over the section (rad)
        _phi     : bending-plane direction about local z-axis (rad)
        _length  : arc length of the section (mm)
        _nPoints : number of samples along the section
        _T0      : 4x4 homogeneous transform (world <- section base frame)

        Returns (worldPts, TEnd) where worldPts is (_nPoints, 3) and TEnd
        is the 4x4 transform of the section tip (world <- tip frame).'''
    s = np.linspace(0, _length, _nPoints)

    if abs(_theta) < 1e-9:
        localPts = np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
        TTip = np.eye(4)
        TTip[2, 3] = _length
    else:
        kappa = _theta / _length
        x = (np.cos(_phi) * (1 - np.cos(kappa * s))) / kappa
        y = (np.sin(_phi) * (1 - np.cos(kappa * s))) / kappa
        z = np.sin(kappa * s) / kappa
        localPts = np.stack([x, y, z], axis=1)

        # Bend is a rotation by theta about the axis perpendicular to phi in
        # the XY plane: R = Rz(phi) * Ry(theta) * Rz(-phi)
        ct, st = np.cos(_theta), np.sin(_theta)
        cphi, sphi = np.cos(_phi), np.sin(_phi)
        RzPhi = np.array([[cphi, -sphi, 0], [sphi, cphi, 0], [0, 0, 1]])
        RyTheta = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        RTip = RzPhi @ RyTheta @ RzPhi.T

        TTip = np.eye(4)
        TTip[:3, :3] = RTip
        TTip[:3, 3] = localPts[-1]  # tip position = last sampled point

    homogPts = np.hstack([localPts, np.ones((_nPoints, 1))])
    worldPts = (_T0 @ homogPts.T).T[:, :3]
    TEnd = _T0 @ TTip
    return worldPts, TEnd


def calcBackbone(_sectionParams, _nPoints=POINTS_PER_SECTION):
    '''Concatenated (N, 3) backbone in world coordinates from a list of (theta, phi, length) per section.'''
    T = np.eye(4)
    allPoints = []
    for theta, phi, length in _sectionParams:
        pts, T = calcSectionCurve(theta, phi, length, _nPoints, T)
        allPoints.append(pts)
    return np.vstack(allPoints)


def plotStatic():
    '''Plots one example backbone: section 1 bent 40 deg, section 2 straight.'''
    sectionParams = [
        (np.radians(40), np.radians(30), SECT_LENGTH),  # section 1: bent
        (np.radians(0), np.radians(0), SECT_LENGTH),     # section 2: straight
    ]
    backbone = calcBackbone(sectionParams)

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


def animateRobot(_useController: bool = False):
    '''Animates the backbone through the PCC IK.

        _useController=False (default): replays the trajectory from Core/trajectories.py,
            cycling setup -> task -> pack-up -> repeat.
        _useController=True: drives the sim live from a real Xbox controller
            via controller.ctrlInput.Controller instead - needs a controller
            actually plugged in (constructing Controller() opens joystick 0),
            so this path isn't exercised by anything that runs without one.'''
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], color="tab:blue", linewidth=3, marker="o", markersize=3)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 2 * SECT_LENGTH)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("PCC Backbone Animation" + (" (controller input)" if _useController else ""))

    controller = Controller() if _useController else None

    # Same phase shape as main.py's moveToStartPos -> runControlLoop -> packUp,
    # but driven in Cartesian space (rather than ego/motor space, like main.py
    # does) since this simulator only ever talks to the robot via Cartesian
    # targets -> solveFullIK. "Rest" is the fully-straight pose, the Cartesian
    # analogue of main.py's ego = 0.
    restPose = np.array([0.0, 0.0, 2*robot.sectL, 0.0, 0.0, 0.0])
    taskStartPose = np.array(calcTargGlob(0.0, trajectory)[0])
    taskEndPose = np.array(calcTargGlob(taskTime, trajectory)[0])
    cycleLength = setupTime + taskTime + packupTime

    def calcTargetForCycleTime(_tCycle: float) -> list[float]:
        '''Target tool pose at time _tCycle into the setup -> task -> pack-up cycle.'''
        if _tCycle < setupTime:                                   # moveToStartPos-equivalent ramp-in
            frac = _tCycle / setupTime
            return (restPose + frac*(taskStartPose - restPose)).tolist()
        elif _tCycle < setupTime + taskTime:                      # main trajectory
            pos, _ = calcTargGlob(_tCycle - setupTime, trajectory)
            return pos
        else:                                                     # pack-up-equivalent ramp-down
            frac = (_tCycle - setupTime - taskTime) / packupTime
            return (taskEndPose + frac*(restPose - taskEndPose)).tolist()

    def updateFrame(_frame):
        '''Solves the IK for the current target and redraws the backbone (FuncAnimation callback).'''
        if controller is not None:
            controller.update()
            target = controller.calcTargGlob().tolist()
        else:
            tNow = time.monotonic() - tStart
            tCycle = tNow % cycleLength                         # loop forever until the window closes
            target = calcTargetForCycleTime(tCycle)

        robot.solveFullIK(target, [0] * 6)

        k1, r1, s1, k2, r2, s2 = robot.targConfigPos

        theta1 = s1*k1     # amount of bend
        theta2 = s2*k2
        sectionParams = [
            (theta1, r1, s1),
            (theta2, r2, s2),
        ]

        backbone = calcBackbone(sectionParams)
        line.set_data(backbone[:, 0], backbone[:, 1])
        line.set_3d_properties(backbone[:, 2])
        return line,

    ani = FuncAnimation(fig, updateFrame, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    # plotStatic()
    print("Reading input...\n")
    animateRobot(_useController=True)  # False replays the trajectory instead of reading the Xbox controller
