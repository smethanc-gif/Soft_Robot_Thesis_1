'''Isometric 3D animations of one test for the thesis presentation: a simplified robot (two
constant-curvature sections plus the straight tool) moving through the desired trajectory, the
actual (camera) trajectory, both side by side, or both overlaid.

The desired shape comes from the logged configuration (CylinderParams.txt) when present, otherwise
from the PCC IK on the desired poses. The camera only gives the tip position, so the actual shape is
the PCC IK solved for the camera tip (home-referenced), with the tool tilt taken from the desired pose.

Run from anywhere:  python Results/analysis/thesisPrezo.py   (animates the test in main())
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Core.continuumRobot import ContinuumRobot
from Core.PCC.constantCurvature import ConstantCurvature
from Core.settings import loadRobotGeometry
from Results.analysis.analyse import _prepareTest

GEOMETRY = loadRobotGeometry()
TOOL_L = GEOMETRY["toolLength"]
BASE_R = 3*GEOMETRY["robotRadius"]          # mm, radius of the base plate drawn under the robot
MODES = ["desired", "actual", "sideBySide", "overlay"]
COLOURS = {"desired": "tab:blue", "actual": "tab:orange"}
ISO_ELEV = np.degrees(np.arctan(1/np.sqrt(2)))   # 35.26 deg, true isometric


# ---- Robot shape ----

def calcConfigFromTips(_tipPoses: np.ndarray) -> np.ndarray:
    '''PCC configuration [k1,r1,s1,k2,r2,s2] for each tool-tip pose [x,y,z,Rx,Ry,Rz], solved in
        order and warm-started like the live control loop.'''
    robot = ContinuumRobot("PCC")
    config = []
    for pose in _tipPoses:
        robot.solveFullIK(list(pose), [0]*6)
        config.append(robot.targConfigPos)
    return np.array(config)


def calcRobotShape(_config: np.ndarray, _nPts: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Points [pt][xyz] along section 1, section 2 and the tool for one configuration [k1,r1,s1,k2,r2,s2].'''
    k1, r1, s1, k2, r2, s2 = _config
    sect1 = np.array([ConstantCurvature.calcArcFK(k1, r1, s)[0] for s in np.linspace(0, s1, _nPts)])
    p1, R1 = ConstantCurvature.calcArcFK(k1, r1, s1)
    sect2 = np.array([p1 + R1 @ ConstantCurvature.calcArcFK(k2, r2, s)[0] for s in np.linspace(0, s2, _nPts)])
    _, R2 = ConstantCurvature.calcArcFK(k2, r2, s2)
    axis = (R1 @ R2)[:, 2]
    tool = np.array([sect2[-1], sect2[-1] + TOOL_L*axis])
    return sect1, sect2, tool


# ---- Data ----

def loadAnimationData(_test: int, _ctrl: str = "PCC", _frameStep: int = 1, _homeRef: bool = True) -> dict:
    '''Task (t >= 0) data of one test, every _frameStep-th sample: times, desired and actual tip paths
        and the configuration of each frame for both. Camera samples rejected by getCameraGateMask
        hold the last good reading.'''
    d, run, ok, dt, curh, home = _prepareTest(_test, _ctrl)
    des = d["des"][run]
    if _homeRef and np.isnan(home).any():  # older tests logged no straight pre-start pose
        print(f"Test {_test} has no pre-start samples to home-reference, using the raw camera path.")
        _homeRef = False
    cur = (curh if _homeRef else d["cur"][run, :3]).copy()
    for i in range(1, len(cur)):            # hold through rejected camera jumps
        if not ok[i]:
            cur[i] = cur[i - 1]
    idx = np.arange(0, len(des), _frameStep)
    desConfig = d["config"][run][idx] if "config" in d else calcConfigFromTips(des[idx])
    curPose = np.c_[cur[idx], des[idx, 3:]]          # camera has no orientation: borrow the desired tilt
    return {"t": d["t"][run][idx], "dt": dt*_frameStep,
            "desired": {"path": des[idx, :3], "config": desConfig},
            "actual": {"path": cur[idx], "config": calcConfigFromTips(curPose)}}


# ---- Drawing ----

def _setupWorkspaceAxes(_ax, _paths: list[np.ndarray], _title: str, _elev: float, _azim: float):
    '''Orthographic isometric axes with equal scaling, sized to hold the robot and every path, plus
        the base plate.'''
    allPts = np.vstack(_paths + [np.zeros((1, 3))])
    lo, hi = allPts.min(axis=0) - 20, allPts.max(axis=0) + 20
    lo[2] = 0
    _ax.set_proj_type("ortho")
    _ax.view_init(elev=_elev, azim=_azim)
    _ax.set_xlim(lo[0], hi[0])
    _ax.set_ylim(lo[1], hi[1])
    _ax.set_zlim(lo[2], hi[2])
    _ax.set_box_aspect(hi - lo)
    theta = np.linspace(0, 2*np.pi, 60)
    _ax.plot(BASE_R*np.cos(theta), BASE_R*np.sin(theta), np.zeros_like(theta), color="black", linewidth=1)
    _ax.plot_trisurf(np.r_[0, BASE_R*np.cos(theta)], np.r_[0, BASE_R*np.sin(theta)], np.zeros(61), color="lightgrey", alpha=0.5)
    _ax.set_xlabel("X (mm)")
    _ax.set_ylabel("Y (mm)")
    _ax.set_zlabel("Z (mm)")
    _ax.set_title(_title)


def _createRobotArtists(_ax, _path: np.ndarray, _colour: str, _label: str, _faded: bool = False) -> dict:
    '''Line artists for one robot on _ax: faint full path, growing trail, both sections, tool and tip.'''
    alpha = 0.45 if _faded else 1.0
    style = "--" if _faded else "-"
    _ax.plot(*_path.T, color=_colour, alpha=0.15, linewidth=1)
    return {"trail": _ax.plot([], [], [], color=_colour, alpha=0.8*alpha, linewidth=1.5, linestyle=style, label=f"{_label} path")[0],
            "sect1": _ax.plot([], [], [], color=_colour, alpha=alpha, linewidth=7, solid_capstyle="round")[0],
            "sect2": _ax.plot([], [], [], color=_colour, alpha=alpha, linewidth=5, solid_capstyle="round")[0],
            "tool": _ax.plot([], [], [], color="dimgrey", alpha=alpha, linewidth=2.5)[0],
            "tip": _ax.plot([], [], [], marker="o", color="black", markerfacecolor=_colour, markersize=8, alpha=alpha, linestyle="", label=f"{_label} tip")[0]}


def _updateRobotArtists(_artists: dict, _path: np.ndarray, _config: np.ndarray, _frame: int):
    '''Moves one robot's artists to frame _frame. The tip marker sits on the logged path point.'''
    sect1, sect2, tool = calcRobotShape(_config[_frame])
    _artists["trail"].set_data_3d(*_path[:_frame + 1].T)
    _artists["sect1"].set_data_3d(*sect1.T)
    _artists["sect2"].set_data_3d(*sect2.T)
    _artists["tool"].set_data_3d(*tool.T)
    _artists["tip"].set_data_3d(*_path[_frame:_frame + 1].T)


def animateWorkspace(_test: int, _ctrl: str = "PCC", _mode: str = "overlay", _frameStep: int = 2, _speed: float = 1.0, _homeRef: bool = True, _elev: float = ISO_ELEV, _azim: float = 45.0, _savePath: str = None) -> FuncAnimation:
    '''Isometric 3D animation of one test's robot moving through its trajectory.
        _mode       - "desired", "actual", "sideBySide" or "overlay"
        _frameStep  - animate every _frameStep-th logged sample
        _speed      - playback speed (1 = real time)
        _homeRef    - home-reference the camera path (see analyse.calcHomeRef)
        _savePath   - also write the animation to this file (.gif via Pillow, .mp4 needs ffmpeg)'''
    if _mode not in MODES:
        raise ValueError(f"Unknown mode: {_mode!r} (choose from {MODES})")
    print(f"Animating Workspace... \nTest {_test} for {_ctrl} control, {_mode} view.\n")
    data = loadAnimationData(_test, _ctrl, _frameStep, _homeRef)
    paths = [data["desired"]["path"], data["actual"]["path"]]
    shown = {"desired": ["desired"], "actual": ["actual"], "sideBySide": ["desired", "actual"], "overlay": ["desired", "actual"]}[_mode]

    if _mode == "sideBySide":
        fig = plt.figure(figsize=(13, 7))
        panels = [(fig.add_subplot(1, 2, i + 1, projection="3d"), [key]) for i, key in enumerate(shown)]
    else:
        fig = plt.figure(figsize=(8, 8))
        panels = [(fig.add_subplot(111, projection="3d"), shown)]

    robots = []
    for ax, keys in panels:
        title = " vs ".join(k.capitalize() for k in keys)
        _setupWorkspaceAxes(ax, paths, title, _elev, _azim)
        for key in keys:
            faded = _mode == "overlay" and key == "desired"
            robots.append((key, _createRobotArtists(ax, data[key]["path"], COLOURS[key], key.capitalize(), faded)))
        ax.legend(loc="upper left")
    timeText = fig.suptitle("")

    def drawFrame(_frame):
        '''Moves every robot to frame _frame and updates the clock.'''
        for key, artists in robots:
            _updateRobotArtists(artists, data[key]["path"], data[key]["config"], _frame)
        timeText.set_text(f"{_ctrl} Test {_test}    t = {data['t'][_frame]:5.2f} s")

    interval = 1000*data["dt"]/_speed
    anim = FuncAnimation(fig, drawFrame, frames=len(data["t"]), interval=interval, blit=False)
    fig.tight_layout()
    if _savePath is not None:
        writer = PillowWriter(fps=1000/interval) if _savePath.endswith(".gif") else None
        anim.save(_savePath, writer=writer)
        print(f"Saved animation to {_savePath}")
    plt.show()
    return anim


def main():
    '''Animates one example test - uncomment what's needed.'''
    exampleTest = 299
    # animateWorkspace(256, _ctrl="PCC", _mode="overlay",_savePath="S_ShapedCirc_PCC.gif")
    animateWorkspace(228, _ctrl="PCC", _mode="overlay", _savePath="fig8_PCC.gif")
    #animateWorkspace(229, _ctrl="PCC", _mode="overlay")
    #animateWorkspace(230, _ctrl="PCC", _mode="overlay")
    # animateWorkspace(293, _ctrl="PCC", _mode="overlay",_savePath="helix_PCC.gif")
    # animateWorkspace(259, _ctrl="PCC", _mode="overlay",_savePath="C_ShapedCirc_PCC.gif")
    # animateWorkspace(239, _ctrl="PCC", _mode="overlay", _savePath="C_StepReturn.gif")
    #animateWorkspace(exampleTest, _ctrl="PCC", _mode="sideBySide")
    #animateWorkspace(exampleTest, _ctrl="PCC", _mode="desired")
    #animateWorkspace(exampleTest, _ctrl="PCC", _mode="actual")
    #animateWorkspace(91, _ctrl="PCC_Cable", _mode="overlay", _savePath="fig8_PCC_Cable.gif")
    #animateWorkspace(42, _ctrl="Poly", _mode="sideBySide", _speed=2.0)


if __name__ == "__main__":
    main()
