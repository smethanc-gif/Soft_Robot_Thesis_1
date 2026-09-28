'''Offline check of the three control methods before running them on the robot.

Replays the desired tip poses of logged tests, in order, through each method's
IK exactly as main.py calls it (ContinuumRobot.FullIK), then reports:
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
    python results/checkIK.py
'''
import os
import sys
import time
import numpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from PCC_control.CTRL import ContinuumRobot
from PCC_control import polyModel
from results.Graphing import parse_log

TESTS = {196: "circle z250 mf2", 199: "circle z250 mf6", 206: "figure 8 z260 mf3", 209: "figure 8 z260 mf6",
         216: "helix 300->220 mf3", 214: "helix 300->230 mf6", 203: "step-return C", 204: "step-return S"}
# Settings that keep all three methods inside the motor range of the training data
# (generated like main.py's trajectories 4, 7 and 8, 30 s at movementfreq 4)
COMPARISON_SET = {"circle r100 z280": ("circle", 100, 280, 280), "figure 8 a100 z300": ("figure 8", 100, 300, 300),
                  "helix r90 300->280": ("helix", 90, 300, 280)}
METHODS = ("PCC", "PCC_Cable", "Poly")
STEP_TIME = 0.04


def desiredPoses(test: int, ctrl: str = "PCC"):
    t, vals = parse_log(os.path.join(REPO, ctrl, f"Test_{test}", "DesiredTipPosition.txt"))
    t, poses = numpy.array(t), numpy.array(vals).T
    return t[t >= 0], poses[t >= 0]


def generatedPoses(kind: str, size: float, z0: float, z1: float, mf: int = 4, taskTime: float = 30.0):
    t = numpy.arange(0, taskTime, STEP_TIME)
    w = 2*numpy.pi*mf/taskTime*t
    if kind == "circle":
        x, y, z = size*numpy.cos(w), size*numpy.sin(w), numpy.full_like(t, z0)
    elif kind == "figure 8":
        x = size*numpy.cos(w)/(1 + numpy.sin(w)**2)
        y = size*numpy.sin(w)*numpy.cos(w)/(1 + numpy.sin(w)**2)
        z = numpy.full_like(t, z0)
    else:
        x, y, z = size*numpy.cos(w), size*numpy.sin(w), z0 + (z1 - z0)*t/taskTime
    return numpy.c_[x, y, z, numpy.zeros((len(t), 3))]


def trainingRange():
    ''' Largest |measured motor angle| per motor over the training tests. '''
    import json
    C = json.load(open(polyModel.CONSTANTS_FILE))
    top = numpy.zeros(6)
    for tests in C["train_tests"].values():
        for t in tests:
            tt, vals = parse_log(os.path.join(REPO, "PCC", f"Test_{t}", "CurrentMotorPosition.txt"))
            q = numpy.array(vals).T[numpy.array(tt) >= 0]
            top = numpy.maximum(top, numpy.abs(q).max(axis=0))
    return top


def replay(method: str, poses: numpy.ndarray):
    robot = ContinuumRobot(method)
    q, times, iters, solve = [], [], [], []
    for pose in poses:
        t0 = time.perf_counter()
        cmd, _ = robot.FullIK(list(pose), [0]*6)
        times.append(time.perf_counter() - t0)
        q.append(numpy.array(cmd, dtype=float))
        if method == "Poly":
            iters.append(robot.lastIterations)
            solve.append(robot.lastSolveMs)
    return numpy.array(q), numpy.array(times)*1000, robot, numpy.array(iters), numpy.array(solve)


def main():
    C = polyModel.loadConstants()
    top = trainingRange()
    print("Training data motor range (max |angle|, rad) A..F: " + " ".join(f"{m} {v:.1f}" for m, v in zip("ABCDEF", top)))
    print("Predicted tip error = where the fitted Poly model puts the tip for each method's commands (mm).")
    print(" test | trajectory          | method    | predicted RMSE / max | max |motor| | beyond training | max speed  | IK time mean / max | notes")
    runs = [(f"{t} | {n:19s}", desiredPoses(t)[1]) for t, n in TESTS.items()]
    runs += [(f"gen  | {n:19s}", generatedPoses(*spec)) for n, spec in COMPARISON_SET.items()]
    for label, poses in runs:
        for method in METHODS:
            q, ms, robot, iters, solve = replay(method, poses)
            tip = polyModel.polyTip(C["poly"], q, C["gains"])
            err = numpy.linalg.norm(tip - poses[:, :3], axis=1)
            speed = numpy.abs(numpy.diff(q, axis=0)).max()/STEP_TIME
            notes = ""
            if method == "Poly":
                notes = (f"own solve {solve.mean():.1f} / {solve.max():.1f} ms, iterations mean {iters.mean():.1f} max {iters.max()},"
                         f" limited {robot.limitCount}, failed {robot.fallbackCount}, correction vs PCC_Cable max {numpy.abs(q - replay('PCC_Cable', poses)[0]).max():.2f} rad")
            if method != "PCC" and robot.clampCount:
                notes += f" clamped {robot.clampCount}x"
            print(f" {label} | {method:9s} | {numpy.sqrt(numpy.mean(err**2)):6.1f} / {err.max():5.1f}"
                  f"       | {numpy.abs(q).max():5.2f} rad  | {100*numpy.mean((numpy.abs(q) > top).any(axis=1)):5.1f}% of steps  |"
                  f" {speed:5.1f} rad/s | {ms.mean():5.1f} / {ms.max():5.1f} ms | {notes}")


if __name__ == "__main__":
    main()
