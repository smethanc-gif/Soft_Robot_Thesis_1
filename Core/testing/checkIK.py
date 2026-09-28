'''Offline check of the three control methods before running them on the robot.

Replays the desired tip poses of logged tests, in order, through each method's
IK exactly as main.py calls it (ContinuumRobot.solveFullIK), then reports:
  - predicted tip error: where the fitted Poly model says each method's
    motor commands would put the tip, vs the target (Poly is judged by its own
    model here, so it should be ~0 - the point is to see how far PCC and
    PCC_Cable are from it, and that the Poly solve converges)
  - motor range and speed, against the 15 rad clamp / 20 rad STM32 trip
  - motor commands outside the range seen in the training data (the fitted
    model is only validated inside it)
  - IK time per control step (the loop has 40 ms in total), and for Poly the
    time of its own solve on top of PCC's shape solve
  - Poly solver iterations and fallbacks

Needs only numpy + the logs, so it can be run on the Pi to check timing there:
    python Core/testing/checkIK.py
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import time
import numpy as np
from Core.continuumRobot import ContinuumRobot
from Core.Poly import polyModel
from Core.settings import loadControlSettings
from Results.logReader import parseLog, getLogPath

TESTS = {196: "circle z250 mf2", 199: "circle z250 mf6", 206: "figure 8 z260 mf3", 209: "figure 8 z260 mf6",
         216: "helix 300->220 mf3", 214: "helix 300->230 mf6", 203: "step-return C", 204: "step-return S"}
# Settings that keep all three methods inside the motor range of the training data
# (generated like trajectories 4, 7 and 8 in Core/trajectories.py, 30 s at movementFreq 4)
COMPARISON_SET = {"circle r100 z280": ("circle", 100, 280, 280), "figure 8 a100 z300": ("figure 8", 100, 300, 300),
                  "helix r90 300->280": ("helix", 90, 300, 280)}
METHODS = ("PCC", "PCC_Cable", "Poly")
STEP_TIME = loadControlSettings()["timeStep"]


def loadDesiredPoses(_test: int, _ctrl: str = "PCC") -> tuple[np.ndarray, np.ndarray]:
    '''Times and desired tool poses [sample][x, y, z, Rx, Ry, Rz] of a logged test, task only (t >= 0).'''
    t, vals = parseLog(getLogPath(_test, _ctrl, "desiredGlob"))
    t, poses = np.array(t), np.array(vals).T
    return t[t >= 0], poses[t >= 0]


def calcGeneratedPoses(_kind: str, _size: float, _z0: float, _z1: float, _mf: int = 4, _taskTime: float = 30.0) -> np.ndarray:
    '''Desired tool poses of a "circle", "figure 8" or "helix" (z from _z0 to _z1) of size _size mm,
        _mf repetitions in _taskTime s, sampled every STEP_TIME.'''
    t = np.arange(0, _taskTime, STEP_TIME)
    w = 2*np.pi*_mf/_taskTime*t
    if _kind == "circle":
        x, y, z = _size*np.cos(w), _size*np.sin(w), np.full_like(t, _z0)
    elif _kind == "figure 8":
        x = _size*np.cos(w)/(1 + np.sin(w)**2)
        y = _size*np.sin(w)*np.cos(w)/(1 + np.sin(w)**2)
        z = np.full_like(t, _z0)
    else:
        x, y, z = _size*np.cos(w), _size*np.sin(w), _z0 + (_z1 - _z0)*t/_taskTime
    return np.c_[x, y, z, np.zeros((len(t), 3))]


def calcTrainingRange():
    '''Largest |measured motor angle| per motor over the training tests.'''
    saved = json.load(open(polyModel.CONSTANTS_FILE))
    top = np.zeros(6)
    for tests in saved["train_tests"].values():
        for t in tests:
            tt, vals = parseLog(getLogPath(t, "PCC", "currentEgo"))
            q = np.array(vals).T[np.array(tt) >= 0]
            top = np.maximum(top, np.abs(q).max(axis=0))
    return top


def replayMethod(_method: str, _poses: np.ndarray):
    '''Runs the poses in order through one method's IK. Returns (motor commands, IK time per step in ms,
        the robot, Poly solver iterations, Poly solve time in ms).'''
    robot = ContinuumRobot(_method)
    q, times, iters, solve = [], [], [], []
    for pose in _poses:
        t0 = time.perf_counter()
        cmd, _ = robot.solveFullIK(list(pose), [0]*6)
        times.append(time.perf_counter() - t0)
        q.append(np.array(cmd, dtype=float))
        if _method == "Poly":
            iters.append(robot.lastIterations)
            solve.append(robot.lastSolveMs)
    return np.array(q), np.array(times)*1000, robot, np.array(iters), np.array(solve)


def main():
    '''Replays the logged tests and the comparison set through all three methods and prints the table.'''
    constants = polyModel.loadConstants()
    top = calcTrainingRange()
    print("Training data motor range (max |angle|, rad) A..F: " + " ".join(f"{m} {v:.1f}" for m, v in zip("ABCDEF", top)))
    print("Predicted tip error = where the fitted Poly model puts the tip for each method's commands (mm).")
    print(" test | trajectory          | method    | predicted RMSE / max | max |motor| | beyond training | max speed  | IK time mean / max | notes")
    runs = [(f"{t} | {n:19s}", loadDesiredPoses(t)[1]) for t, n in TESTS.items()]
    runs += [(f"gen  | {n:19s}", calcGeneratedPoses(*spec)) for n, spec in COMPARISON_SET.items()]
    for label, poses in runs:
        for method in METHODS:
            q, ms, robot, iters, solve = replayMethod(method, poses)
            tip = polyModel.calcPolyTip(constants["poly"], q, constants["gains"])
            err = np.linalg.norm(tip - poses[:, :3], axis=1)
            speed = np.abs(np.diff(q, axis=0)).max()/STEP_TIME
            notes = ""
            if method == "Poly":
                notes = (f"own solve {solve.mean():.1f} / {solve.max():.1f} ms, iterations mean {iters.mean():.1f} max {iters.max()},"
                         f" limited {robot.limitCount}, failed {robot.fallbackCount}, correction vs PCC_Cable max {np.abs(q - replayMethod('PCC_Cable', poses)[0]).max():.2f} rad")
            if method != "PCC" and robot.clampCount:
                notes += f" clamped {robot.clampCount}x"
            print(f" {label} | {method:9s} | {np.sqrt(np.mean(err**2)):6.1f} / {err.max():5.1f}"
                  f"       | {np.abs(q).max():5.2f} rad  | {100*np.mean((np.abs(q) > top).any(axis=1)):5.1f}% of steps  |"
                  f" {speed:5.1f} rad/s | {ms.mean():5.1f} / {ms.max():5.1f} ms | {notes}")


if __name__ == "__main__":
    main()
