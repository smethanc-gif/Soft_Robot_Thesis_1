'''Processes the positional information stored using fileWriting.py and returns graphs and figures on the following information:\n
'''
import os
import sys
import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root, so this file can be run directly
from Results.analysis.graphing import parse_log, DATA_ROOT, GlobErrorMetrics as trackingErrorMetrics, plotGlobError as plotTrackingError
from Core.continuumRobot import ContinuumRobot
from Core.settings import loadRobotGeometry, loadDataLogging

GEOMETRY = loadRobotGeometry()
FILES = loadDataLogging()["files"]
TOOL_L = GEOMETRY["toolLength"]
HOME_Z = 2*GEOMETRY["sectL"] + TOOL_L   # straight tool-tip height of the PCC model (327 mm)


# ============================================================
# Loading / cleaning
# ============================================================

def loadTest(test: int, ctrl: str = "PCC", packup: bool = False) -> dict:
    ''' All logs of one test as numpy arrays [sample][value], trimmed to a
        common length. "cyl" and "lim" are only present for tests logged after
        CylinderParams.txt / LimitsHit.txt were added.
        Pack-up rows (logged after the task, from TestInfo.txt's packupStart
        on) are left out unless packup=True, so t >= 0 is always just the
        task.'''
    folder = f"{DATA_ROOT}/{ctrl}/Test_{test}"
    d = {}
    for key, name in [("des", "desiredGlob"), ("cal", "calculatedGlob"),
                      ("cur", "currentGlob"), ("dm", "desiredEgo"),
                      ("cm", "currentEgo"), ("cyl", "configParams")]:
        path = f"{folder}/{FILES[name]}"
        if os.path.exists(path):
            t, vals = parse_log(path)
            d[key] = numpy.array(vals).T
            if key == "des":
                d["t"] = numpy.array(t)
    if os.path.exists(f"{folder}/{FILES['limitsHit']}"):
        _, vals = parse_log(f"{folder}/{FILES['limitsHit']}", ncols=4)
        d["lim"] = numpy.array(vals).T
    n = min(len(v) for v in d.values())
    d = {k: v[:n] for k, v in d.items()}
    packupStart = loadMeta(test, ctrl).get("packupStart")
    if not packup and isinstance(packupStart, (int, float)):
        task = d["t"] < packupStart
        d = {k: v[task] for k, v in d.items()}
    return d


def cameraGate(d: dict, max_jump_mm: float = 200.0) -> numpy.ndarray:
    ''' True for every sample except camera readings that jump more than
        max_jump_mm from the last accepted reading - the tracker locking onto
        a false blob when the tip is hidden (e.g. ~550 mm jumps to
        (-70, 445, -40) at t~83-86 s in every step-return run). Everything
        else is kept raw: no motor or repeated-frame filtering.'''
    cur = d["cur"][:, :3]
    keep = numpy.ones(len(cur), dtype=bool)
    last = cur[0]
    for i in range(1, len(cur)):
        if numpy.linalg.norm(cur[i] - last) > max_jump_mm:
            keep[i] = False
        else:
            last = cur[i]
    return keep


def homeRef(d: dict, pre_t: float = -1.8, z0: float = HOME_Z) -> tuple[numpy.ndarray, numpy.ndarray]:
    ''' Camera tip positions shifted so the straight pre-start pose (t < pre_t,
        before CTRL_INIT has moved the motors) sits at (0, 0, z0). Removes the
        per-session marker pose-lock offset. Returns (shifted xyz, raw home).'''
    home = d["cur"][d["t"] < pre_t, :3].mean(axis=0)
    return d["cur"][:, :3] - home + numpy.array([0.0, 0.0, z0]), home


def _stepReturnBend(des: numpy.ndarray) -> numpy.ndarray:
    ''' Total commanded bend (rad) of main.py's step-return poses.
        C-shape (traj 9): the tool tilts with the bend, so bend = tilt.
        S-shape (traj 10): the tool stays vertical and the equal-and-opposite
        arcs put the tip at xy / (z - tool) = tan(bend/4).'''
    tilt = _bendFromPose(des)
    sBend = 4*numpy.arctan2(numpy.hypot(des[:, 0], des[:, 1]), des[:, 2] - TOOL_L)
    return numpy.where(tilt > 1e-6, tilt, sBend)


def _bendFromPose(pose: numpy.ndarray) -> numpy.ndarray:
    ''' Tool-axis tilt from vertical (rad) from logged Rx, Ry columns - same
        tangent convention as constantCurve._tangentFromAngles.'''
    return numpy.arccos(numpy.clip(numpy.cos(pose[:, 3])*numpy.cos(pose[:, 4]), -1, 1))


# ============================================================
# Error vs bend angle
# ============================================================

def bendParams(d: dict, step: int = 1) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    ''' Sample indices, per-sample arc parameters [k1,r1,s1,k2,r2,s2], model
        FK tool pose and IK limit-hit count. Uses the logged CylinderParams.txt when present,
        otherwise re-runs the PCC IK on the desired poses (warm-started in
        order, like the live control loop). Only every `step`th sample.'''
    idx = numpy.arange(0, len(d["t"]), step)
    if "cyl" in d:
        lim = d["lim"][idx].sum(axis=1) if "lim" in d else numpy.zeros(len(idx))
        return idx, d["cyl"][idx], d["cal"][idx], lim
    robot = ContinuumRobot("PCC")
    cyl, fk, lim = [], [], []
    for i in idx:
        robot.FullIK(list(d["des"][i]), [0]*6)
        cyl.append(robot.targCylPos)
        fk.append(robot.forwardKinematics())
        lim.append(sum(robot.limitFlags))
    return idx, numpy.array(cyl), numpy.array(fk), numpy.array(lim)


def errorVsBend(tests: list[int], ctrl: str = "PCC", bins=(0, 10, 20, 30, 45, 60, 90, 120, 180),
                step: int = 2, verbose: bool = True) -> dict:
    ''' Tip error binned by total commanded bend |k1*s1| + |k2*s2| (deg),
        pooled over `tests`, camera-gated (cameraGate) and home-referenced (homeRef).
        gain = actual tip displacement from straight / model displacement:
        <1 the robot under-bends, >1 it is further out than commanded
        (at low bend that means it hasn't come back - hysteresis).'''
    pooled = {k: [] for k in ("bend", "raw", "home", "dmod", "dact", "dz", "lim")}
    for t in tests:
        d = loadTest(t, ctrl)
        curh, home = homeRef(d)
        idx, cyl, fk, lim = bendParams(d, step)
        keep = cameraGate(d)[idx] & (d["t"][idx] >= 0)
        idx, cyl, fk, lim = idx[keep], cyl[keep], fk[keep], lim[keep]
        straight = numpy.array([0.0, 0.0, HOME_Z])
        pooled["bend"].append(numpy.degrees(numpy.abs(cyl[:, 0]*cyl[:, 2]) + numpy.abs(cyl[:, 3]*cyl[:, 5])))
        pooled["raw"].append(numpy.linalg.norm(fk[:, :3] - d["cur"][idx, :3], axis=1))
        pooled["home"].append(numpy.linalg.norm(fk[:, :3] - curh[idx], axis=1))
        pooled["dmod"].append(numpy.linalg.norm(fk[:, :3] - straight, axis=1))
        pooled["dact"].append(numpy.linalg.norm(curh[idx] - straight, axis=1))
        pooled["dz"].append(curh[idx, 2] - fk[:, 2])
        pooled["lim"].append(lim)
        if verbose:
            print(f"Test_{t}: {keep.sum()} samples ({(~keep).sum()} camera jumps rejected), session home offset {numpy.round(home - straight, 1)} mm")
    p = {k: numpy.concatenate(v) for k, v in pooled.items()}

    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p["bend"] >= lo) & (p["bend"] < hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": (lo, hi), "n": int(m.sum()),
                     "raw_err": float(p["raw"][m].mean()), "home_err": float(p["home"][m].mean()),
                     "model_disp": float(p["dmod"][m].mean()), "actual_disp": float(p["dact"][m].mean()),
                     "gain": float(numpy.median(p["dact"][m]/numpy.maximum(p["dmod"][m], 1.0))),
                     "z_bias": float(p["dz"][m].mean()), "limit_frac": float((p["lim"][m] > 0).mean())})
    if verbose:
        print(f"\nError vs bend, tests {list(tests)}")
        print(" bend (deg) |   n  | raw err | home err | model disp | actual disp | gain | z bias | limits hit")
        for r in rows:
            print(f"  {r['bin'][0]:3d}-{r['bin'][1]:3d}   | {r['n']:4d} | {r['raw_err']:7.1f} | {r['home_err']:8.1f} | "
                  f"{r['model_disp']:10.1f} | {r['actual_disp']:11.1f} | {r['gain']:4.2f} | {r['z_bias']:6.1f} | {100*r['limit_frac']:5.0f}%")
    return {"bins": rows, "samples": p}


def plotErrorVsBend(tests: list[int], ctrl: str = "PCC"):
    ''' Home-referenced error and gain vs commanded bend (see errorVsBend). '''
    res = errorVsBend(tests, ctrl)
    rows = res["bins"]
    mids = [(r["bin"][0] + r["bin"][1])/2 for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(res["samples"]["bend"], res["samples"]["home"], s=3, alpha=0.2, color="tab:gray")
    ax1.plot(mids, [r["home_err"] for r in rows], marker="o", color="tab:red", label="Home-referenced")
    ax1.plot(mids, [r["raw_err"] for r in rows], marker="s", linestyle="--", color="tab:blue", label="Raw")
    ax1.set_xlabel("Commanded bend |θ1|+|θ2| (deg)")
    ax1.set_ylabel("Tip error (mm)")
    ax1.set_title("Tip error vs bend")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(mids, [r["gain"] for r in rows], marker="o", color="tab:purple")
    ax2.axhline(1.0, color="black", linewidth=0.8)
    ax2.set_xlabel("Commanded bend |θ1|+|θ2| (deg)")
    ax2.set_ylabel("Actual / model displacement")
    ax2.set_title("Displacement gain vs bend")
    ax2.grid(True)
    fig.suptitle(f"{ctrl} tests {list(tests)}")
    fig.tight_layout()
    plt.show()


# ============================================================
# Step-return (hysteresis) test - main.py trajectory 9
# ============================================================

def stepReturn(test: int, ctrl: str = "PCC", settle_s: float = 1.0, verbose: bool = True) -> list[dict]:
    ''' For each "straight hold" of a step-return test, the tip residual left
        over from the preceding bend. Segments are found from the logged
        desired pose (no dependency on main.py's constants), for both the
        C-shape (traj 9) and S-shape (traj 10) versions: a rest is a run of
        desired bend ~0 after t=0, and it belongs to the bend just before it.
        The first settle_s of each rest is skipped.
          residual     - home-referenced actual - model tip, mean over the rest
          along        - residual component along the prior bend direction
                         (positive = still bent that way)
          slack_mm     - the residual converted to equivalent per-tendon cable
                         length error [A..F] via the PCC IK: how much each
                         tendon is effectively longer (+) or shorter (-) than
                         commanded. This is what backlash compensation needs.'''
    d = loadTest(test, ctrl)
    curh, _ = homeRef(d)
    ok = cameraGate(d)
    bend = _stepReturnBend(d["des"])
    run = d["t"] >= 0
    straight = run & (bend < numpy.radians(0.5))

    # contiguous straight runs
    edges = numpy.flatnonzero(numpy.diff(numpy.r_[0, straight.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]

    robot = ContinuumRobot("PCC")
    robot.FullIK([0, 0, HOME_Z, 0, 0, 0], [0]*6)
    straightCable = robot.targCablePos.copy()
    dt = numpy.median(numpy.diff(d["t"]))

    results = []
    prev_end = 0
    for s, e in zip(starts, ends):
        prior = numpy.flatnonzero(run[prev_end:s]) + prev_end   # the bend between the last rest and this one
        prev_end = e
        if len(prior) == 0:
            continue                                    # rest before the first bend
        k = prior[numpy.argmax(bend[prior])]            # peak of that bend
        phi = numpy.arctan2(d["des"][k, 1], d["des"][k, 0])
        m = numpy.arange(s + int(settle_s/dt), e)
        m = m[ok[m]]
        if len(m) < 3:
            continue
        resid = (curh[m] - d["cal"][m, :3]).mean(axis=0)
        u = numpy.array([numpy.cos(phi), numpy.sin(phi)])
        robot.FullIK([resid[0], resid[1], HOME_Z + resid[2], 0, 0, 0], [0]*6)
        results.append({"t": float(d["t"][s]), "dir_deg": float(numpy.degrees(phi)) % 360,
                        "bend_deg": float(numpy.degrees(bend[k])), "n": len(m),
                        "residual": resid, "residual_xy": float(numpy.linalg.norm(resid[:2])),
                        "along": float(resid[:2] @ u),
                        "slack_mm": robot.targCablePos - straightCable})
    if verbose:
        print(f"Step-return, Test_{test}")
        print("   t (s) | dir | bend |  n  | residual xyz (mm)    | |xy| | along | equiv. tendon slack A..F (mm)")
        for r in results:
            print(f"  {r['t']:6.1f} | {r['dir_deg']:3.0f} | {r['bend_deg']:4.0f} | {r['n']:3d} | "
                  f"{numpy.array2string(r['residual'], precision=1, suppress_small=True):20s} | {r['residual_xy']:4.1f} | "
                  f"{r['along']:5.1f} | {numpy.array2string(r['slack_mm'], precision=2, suppress_small=True)}")
    return results


def plotStepReturn(test: int, ctrl: str = "PCC", isCartesian: bool = True):
    ''' Where the tip rests (XY, mm, home-referenced) during each straight
        hold of a step-return test.
        Cartesian (default): one arrow per hold from straight (origin) to the
        rest position, coloured by the direction of the bend it just came
        back from (dashed line, same colour); marker shape = size of that
        bend (see legend). Arrows lying along their own dashed line =
        the robot stays bent the way it came from (hysteresis); arrows all
        pointing one way regardless of colour = drift.
        isCartesian=False: the original polar view (dot = prior bend
        direction at the residual's radius).'''
    res = stepReturn(test, ctrl)
    shape = {9: " (C-shape)", 10: " (S-shape)"}.get(loadMeta(test, ctrl).get("trajectory"), "")
    if isCartesian:
        fig, ax = plt.subplots(figsize=(7, 7))
        levels = sorted({round(r["bend_deg"]) for r in res})
        markers = dict(zip(levels, "o^sDvP*X"))
        dirs = sorted({round(r["dir_deg"]) % 360 for r in res})
        colours = dict(zip(dirs, plt.cm.hsv(numpy.linspace(0, 0.9, len(dirs)))))
        reach = max([r["residual_xy"] for r in res] + [1.0])*1.25
        for dd in dirs:                                      # prior bend directions
            u = numpy.radians(dd)
            ax.plot([0, reach*numpy.cos(u)], [0, reach*numpy.sin(u)], "--",
                    color=colours[dd], alpha=0.8, linewidth=1.5, label=f"after {dd}° direction", zorder=3)
        for r in res:
            c = colours[round(r["dir_deg"]) % 360]
            x, y = r["residual"][:2]
            ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, linewidth=1.5))
            ax.plot(x, y, markers[round(r["bend_deg"])], color=c, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5)
        for b in levels:
            ax.plot([], [], markers[b], color="gray", label=f"after {b}° bend")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_xlabel("X residual (mm)")
        ax.set_ylabel("Y residual (mm)")
        ax.set_title(f"Step-return{shape} rest position after each bend, {ctrl} Test {test}\n"
                     f"(arrow along its dashed line = hysteresis; all one way = drift)")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    bends = sorted({round(r["bend_deg"]) for r in res})
    colours = plt.cm.viridis(numpy.linspace(0, 0.85, max(len(bends), 1)))
    for r in res:
        c = colours[bends.index(round(r["bend_deg"]))]
        ang = numpy.arctan2(r["residual"][1], r["residual"][0])
        ax.annotate("", xy=(ang, r["residual_xy"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c))
        ax.plot(numpy.radians(r["dir_deg"]), r["residual_xy"], "o", color=c, markersize=4, alpha=0.5)
    for b, c in zip(bends, colours):
        ax.plot([], [], color=c, label=f"after {b}° bend")
    ax.set_title(f"Straight-hold residual (mm), {ctrl} Test {test}\n(dot = prior bend direction)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.12))
    fig.tight_layout()
    plt.show()




def phaseLag(test: int, ctrl: str = "PCC", axis: int = 0, timeStep: float = 0.04, max_lag_s: float = 2.0) -> float:
    ''' Cross-correlation estimate of the time lag (s) between desired and
        actual tip position on one axis (0=X, 1=Y, 2=Z). Positive means the
        actual position lags behind the desired one.

        These trajectories are periodic (sinusoidal), so an unrestricted
        search can lock onto a large lag that just happens to line up two
        DIFFERENT cycles instead of the true small shift - max_lag_s caps
        the search to lags that could plausibly be real control-loop lag
        (default 2s), not a fraction/multiple of the trajectory's period.'''
    _, desiredPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{test}/{FILES['desiredGlob']}")
    _, currentPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{test}/{FILES['currentGlob']}")
    d = numpy.array(desiredPos[axis])
    c = numpy.array(currentPos[axis])
    n = min(len(d), len(c))
    d, c = d[:n] - numpy.mean(d[:n]), c[:n] - numpy.mean(c[:n])

    if d.std() < 1.0:  # this axis isn't actually commanded to move in this test - lag is undefined, not zero
        return float("nan")

    correlation = correlate(c, d, mode="full")
    lags = correlation_lags(len(c), len(d), mode="full")

    max_lag_samples = max(1, int(round(max_lag_s / timeStep)))
    window = numpy.abs(lags) <= max_lag_samples
    best_lag = lags[window][numpy.argmax(correlation[window])]
    return float(best_lag * timeStep)


def speedSweep(tests: list[int], ctrl: str = "PCC") -> dict:
    ''' Runs trackingErrorMetrics + phaseLag across several tests (e.g. the
        same trajectory shape recorded at different movementfreq settings),
        to see whether tracking degrades with speed and whether that's
        specifically a growing time lag rather than just bigger error.'''
    results = {}
    for t in tests:
        m = trackingErrorMetrics(t, ctrl)
        lag = [phaseLag(t, ctrl, axis=a) for a in range(3)]
        results[t] = {"rmse": m["rmse"], "max_error": m["max_error"],
                       "phase_lag_s_xyz": lag}
    return results


def plotSpeedSweep(tests: list[int], ctrl: str = "PCC"):
    ''' RMSE and phase lag (X axis) vs. test, to visualise trend across a
        speed sweep. Assumes `tests` is already ordered by trajectory speed.'''
    sweep = speedSweep(tests, ctrl)
    rmse = [sweep[t]["rmse"] for t in tests]
    lag_x = [sweep[t]["phase_lag_s_xyz"][0] for t in tests]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(tests, rmse, marker="o", color="tab:red")
    ax1.set_xlabel("Test")
    ax1.set_ylabel("RMSE (mm)")
    ax1.set_title("Tracking error vs. speed")
    ax1.grid(True)

    ax2.plot(tests, lag_x, marker="o", color="tab:purple")
    ax2.set_xlabel("Test")
    ax2.set_ylabel("Phase lag, X axis (s)")
    ax2.set_title("Phase lag vs. speed")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


# ============================================================
# Point 5 - root-causing the desired/actual Cartesian offset
# ============================================================
# Three hypotheses from the supervisor's email: coordinate definitions,
# initial configuration, or the PCC FK/IK model. "Coordinate definitions" is
# a manual check, not code - cross-reference camera_feedback/robot_tracker.py's
# documented marker-frame axes (X=right, Y=up, Z=out-of-marker) against what
# main.py's MOVEMENT_FUNCTIONS/the PCC model treat X,Y,Z as, and re-verify
# ORIGIN_OFFSET_MM and MARKER_SIZE_MM were measured correctly. The other two
# hypotheses are what the functions below distinguish between.

def sessionOffset(test: int, ctrl: str = "PCC", tail_fraction: float = 0.4) -> numpy.ndarray:
    ''' Mean (Calculated - Actual) tip-position offset over the steady-state
        tail of one test. If this varies test-to-test rather than being a
        fixed constant, it points at "initial configuration" (e.g.
        TipTracker's marker pose-lock, re-solved fresh each run) rather than
        a one-off coordinate bug or a fixed model error.'''
    _, calcPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{test}/{FILES['calculatedGlob']}")
    _, curPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{test}/{FILES['currentGlob']}")
    calc = numpy.array(calcPos)[:3].T
    cur = numpy.array(curPos)[:3].T
    n = min(len(calc), len(cur))
    tail = slice(int(n*(1 - tail_fraction)), n)
    return (calc[tail] - cur[tail]).mean(axis=0)


def modelResidualVsBend(tests: list[int], ctrl: str = "PCC", tail_fraction: float = 0.4) -> dict:
    ''' Distinguishes "initial configuration" from "PCC model" for the
        Cartesian offset: removes each test's OWN session offset (see
        sessionOffset), then checks whether the leftover residual error
        correlates with how much the robot is commanded to bend.
          - High correlation  -> real shape/model error (hypothesis 3).
          - Low/no correlation, and the raw offset shrinks a lot once the
            per-session offset is removed -> mostly per-session calibration
            drift (hypothesis 2), not the model.
        Needs tests where the commanded bend amount actually varies within
        the test - a constant-radius trajectory (e.g. a circle held at one
        bend magnitude) can't answer this question, no matter how many
        timesteps it has.'''
    pooled_resid, pooled_bend = [], []
    per_test = {}
    for t in tests:
        _, calcPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{t}/{FILES['calculatedGlob']}")
        _, curPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{t}/{FILES['currentGlob']}")
        _, desPos = parse_log(f"{DATA_ROOT}/{ctrl}/Test_{t}/{FILES['desiredGlob']}")
        calc = numpy.array(calcPos)[:3].T
        cur = numpy.array(curPos)[:3].T
        des = numpy.array(desPos)[:3].T
        n = min(len(calc), len(cur), len(des))
        tail = slice(int(n*(1 - tail_fraction)), n)
        calc, cur, des = calc[tail], cur[tail], des[tail]

        offset = (calc - cur).mean(axis=0)
        resid = numpy.linalg.norm((calc - offset) - cur, axis=1)
        bend = numpy.linalg.norm(des[:, :2], axis=1)

        corr = float(numpy.corrcoef(bend, resid)[0, 1]) if bend.std() > 1e-6 else float("nan")
        per_test[t] = {"offset_mm": offset, "residual_mean_mm": float(resid.mean()),
                        "bend_range_mm": (float(bend.min()), float(bend.max())),
                        "corr_bend_residual": corr}
        pooled_resid.append(resid)
        pooled_bend.append(bend)

    pooled_resid = numpy.concatenate(pooled_resid)
    pooled_bend = numpy.concatenate(pooled_bend)
    pooled_corr = float(numpy.corrcoef(pooled_bend, pooled_resid)[0, 1])
    return {
        "per_test": per_test,
        "pooled_residual_mean_mm": float(pooled_resid.mean()),
        "pooled_corr_bend_residual": pooled_corr,
    }


# ============================================================
# Advisor deliverables: 3D tracking, metrics, speed / phase lag, error budget
# ============================================================

def loadMeta(test: int, ctrl: str = "PCC") -> dict:
    ''' Run settings from TestInfo.txt (see fileWriting.writeMeta), numbers
        converted. {} for tests logged before TestInfo.txt existed.'''
    path = f"{DATA_ROOT}/{ctrl}/Test_{test}/{FILES['testInfo']}"
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
            meta[key] = value
    return meta


def _period(des: numpy.ndarray, dt: float, meta: dict) -> float:
    ''' Trajectory period (s): from TestInfo.txt when the trajectory is a
        plain periodic one, otherwise the dominant FFT period of the desired
        path on its most-moving axis.'''
    mf, traj, T = meta.get("movementfreq"), meta.get("trajectory"), meta.get("taskTime")
    if isinstance(mf, (int, float)) and mf > 0 and T:
        if traj in (0, 4, 7, 8):
            return T/mf
        if traj == 1:                       # main.py doubles const_1 for the spiral
            return T/(2*mf)
    ax = numpy.argmax(des.std(axis=0))
    x = des[:, ax] - des[:, ax].mean()
    f = numpy.fft.rfftfreq(len(x), dt)
    P = numpy.abs(numpy.fft.rfft(x))
    return float(1/f[1 + numpy.argmax(P[1:])])


def _rmse(a: numpy.ndarray, b: numpy.ndarray) -> float:
    return float(numpy.sqrt(numpy.mean(numpy.sum((a - b)**2, axis=1))))


def _bestLag(des: numpy.ndarray, cur: numpy.ndarray, ok: numpy.ndarray, dt: float, max_s: float = 1.0) -> tuple[int, float]:
    ''' Delay (samples) of actual behind desired that minimises RMSE, and
        that RMSE. Unlike phaseLag's cross-correlation this can't lock onto a
        different cycle, and it reads directly as "error left if the robot
        weren't late".'''
    best = (0, numpy.inf)
    for L in range(int(max_s/dt) + 1):
        m = ok[L:] & ok[:len(ok) - L]
        r = _rmse(des[:len(des) - L][m], cur[L:][m])
        if r < best[1]:
            best = (L, r)
    return best


def _prep(test: int, ctrl: str):
    ''' Camera-gated, t>=0 arrays shared by the functions below. '''
    d = loadTest(test, ctrl)
    curh, home = homeRef(d)
    run = d["t"] >= 0
    ok = cameraGate(d)[run]
    dt = float(numpy.median(numpy.diff(d["t"][run])))
    return d, run, ok, dt, curh[run], home


def trackingMetrics(test: int, ctrl: str = "PCC") -> dict:
    ''' Cartesian tracking of one test, desired vs actual tip (camera-gated, t>=0):
        error norm over time, mean, RMSE and max - both raw and
        home-referenced - plus mean desired tip speed, time lag, phase lag
        and period.'''
    d, run, ok, dt, curh, home = _prep(test, ctrl)
    meta = loadMeta(test, ctrl)
    des, cur, t = d["des"][run, :3], d["cur"][run, :3], d["t"][run]
    e_raw = numpy.linalg.norm(des - cur, axis=1)[ok]
    e_home = numpy.linalg.norm(des - curh, axis=1)[ok]
    L, rmse_lag = _bestLag(des, curh, ok, dt)
    period = _period(des[ok], dt, meta)
    return {"test": test, "ctrl": ctrl, "meta": meta, "n": int(ok.sum()),
            "n_rejected": int((~ok).sum()), "kept": ok,
            "times": t[ok], "error": e_raw, "error_home": e_home,
            "mean_error": float(e_raw.mean()), "rmse": float(numpy.sqrt(numpy.mean(e_raw**2))),
            "max_error": float(e_raw.max()),
            "mean_error_home": float(e_home.mean()), "rmse_home": float(numpy.sqrt(numpy.mean(e_home**2))),
            "max_error_home": float(e_home.max()),
            "speed": float(numpy.linalg.norm(numpy.diff(des, axis=0), axis=1).mean()/dt),
            "lag_s": L*dt, "rmse_lag_removed": rmse_lag,
            "period_s": period, "phase_deg": 360*L*dt/period,
            "home_offset": home - numpy.array([0.0, 0.0, HOME_Z])}


def trackingTable(tests: list[int], ctrl: str = "PCC", verbose: bool = True) -> list[dict]:
    ''' trackingMetrics for several tests, printed as one table. '''
    rows = [trackingMetrics(t, ctrl) for t in tests]
    if verbose:
        print(f"\nTracking metrics ({ctrl}) - raw | home-referenced")
        print(" test | traj | speed mm/s |  n   | rejected | mean err  |   RMSE    |  max err   | lag s | phase deg")
        for r in rows:
            traj = r["meta"].get("trajectory", "?")
            print(f" {r['test']:4d} | {str(traj):>4s} | {r['speed']:10.1f} | {r['n']:4d} | {r['n_rejected']:8d} | "
                  f"{r['mean_error']:4.1f}|{r['mean_error_home']:4.1f} | {r['rmse']:4.1f}|{r['rmse_home']:4.1f} | "
                  f"{r['max_error']:5.1f}|{r['max_error_home']:5.1f} | {r['lag_s']:5.2f} | {r['phase_deg']:6.1f}")
    return rows


def plotTracking3D(test: int, ctrl: str = "PCC", home_ref: bool = True):
    ''' Desired vs actual (and model-calculated) tip path in 3D with equal
        axes, plus the error norm over time with RMSE and max marked. The raw
        camera path (including rejected jumps) is drawn in grey underneath;
        the actual path breaks at rejected samples instead of bridging them.'''
    m = trackingMetrics(test, ctrl)
    d, run, ok, dt, curh, _ = _prep(test, ctrl)
    des, cal = d["des"][run, :3], d["cal"][run, :3]
    raw = curh if home_ref else d["cur"][run, :3]
    cur = numpy.where(ok[:, None], raw, numpy.nan)
    err, rmse, mx = (m["error_home"], m["rmse_home"], m["max_error_home"]) if home_ref \
        else (m["error"], m["rmse"], m["max_error"])

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    ax.plot(*raw.T, color="tab:gray", alpha=0.35, linewidth=0.8, label="Raw camera (incl. rejected jumps)")
    ax.plot(*des.T, linestyle="--", color="tab:blue", label="Desired")
    ax.plot(*cur.T, color="tab:orange", label="Actual" + (" (home-referenced)" if home_ref else ""))
    ax.plot(*cal.T, color="tab:green", alpha=0.6, label="Model (calculated)")
    allp = numpy.vstack([des, raw[ok], cal])     # rejected jumps would blow out the axes
    centre = (allp.max(axis=0) + allp.min(axis=0))/2
    half = (allp.max(axis=0) - allp.min(axis=0)).max()/2 + 5
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{ctrl} Test {test}: RMSE {rmse:.1f} mm, max {mx:.1f} mm\n"
                 f"mean speed {m['speed']:.0f} mm/s, best-fit lag {m['lag_s']:.2f} s "
                 f"({m['phase_deg']:.0f}°), whole run")
    ax.legend(loc="upper left")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(m["times"], err, color="tab:red", label="Error norm")
    ax2.axhline(rmse, color="tab:blue", linestyle="--", label=f"RMSE = {rmse:.1f} mm")
    ax2.axhline(mx, color="tab:orange", linestyle=":", label=f"Max = {mx:.1f} mm")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cartesian tracking error (mm)")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    plt.show()


def plotSpeedStudy(tests: list[int], ctrl: str = "PCC", labels: dict = None):
    ''' Tracking vs mean tip speed, one colour per trajectory (from
        TestInfo.txt, or `labels` {test: name} for older tests).
          - RMSE and lag-removed RMSE: the gap is what the lag costs.
          - Time lag and phase lag: constant lag with rising phase = a fixed
            delay; rising lag = the robot can't keep up.
          - Gain: falling with speed would mean a bandwidth limit.'''
    rows = trackingTable(tests, ctrl, verbose=False)
    gains = [errorBudget(t, ctrl, verbose=False)["gain"] for t in tests]
    labels = labels or {}
    names = [labels.get(r["test"], f"traj {r['meta'].get('trajectory', '?')}") for r in rows]
    groups = list(dict.fromkeys(names))
    colours = dict(zip(groups, plt.cm.tab10(numpy.arange(len(groups)))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for g in groups:
        idx = sorted((i for i, n in enumerate(names) if n == g), key=lambda i: rows[i]["speed"])
        sp = [rows[i]["speed"] for i in idx]
        c = colours[g]
        axes[0].plot(sp, [rows[i]["rmse_home"] for i in idx], "o-", color=c, label=f"{g}")
        axes[0].plot(sp, [rows[i]["rmse_lag_removed"] for i in idx], "o--", color=c, alpha=0.5)
        axes[1].plot(sp, [rows[i]["lag_s"] for i in idx], "o-", color=c, label=f"{g} lag")
        axes[1].plot([], [], "s--", color=c, alpha=0.5, label=f"{g} phase")
        axes[2].plot(sp, [gains[i] for i in idx], "o-", color=c, label=g)
    ax1b = axes[1].twinx()
    for g in groups:
        idx = sorted((i for i, n in enumerate(names) if n == g), key=lambda i: rows[i]["speed"])
        ax1b.plot([rows[i]["speed"] for i in idx], [rows[i]["phase_deg"] for i in idx], "s--",
                  color=colours[g], alpha=0.5)
    axes[0].set_title("RMSE (solid) vs RMSE with lag removed (dashed)")
    axes[0].set_ylabel("RMSE, home-referenced (mm)")
    axes[1].set_title("Time lag (solid) and phase lag (dashed)")
    axes[1].set_ylabel("Lag (s)")
    ax1b.set_ylabel("Phase lag (deg)")
    axes[2].set_title("Displacement gain (actual / desired)")
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Gain")
    for ax in axes:
        ax.set_xlabel("Mean desired tip speed (mm/s)")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.suptitle(f"{ctrl} speed study")
    fig.tight_layout()
    plt.show()


def errorBudget(test: int, ctrl: str = "PCC", verbose: bool = True) -> dict:
    ''' Where the desired-vs-actual offset comes from (advisor's question:
        coordinate definitions, initial configuration, or the PCC model).
        Removes one candidate cause at a time; RMSE after each step:
          raw        - as logged
          session    - minus the straight-pose XY offset measured before the
                       run starts (initial configuration: marker pose lock,
                       robot not quite straight)
          z_datum    - also minus the straight-pose Z offset from the
                       model's 327 mm (coordinate definition: tip-marker
                       depth / straight length / marker plane)
          lag        - also shifted by the best time lag (dynamics + camera)
          gain       - also with the desired displacement from straight
                       scaled by one least-squares gain (PCC model
                       under-/over-bending)
          similarity - instead of "gain", starting again from the "lag"
                       stage: XY fitted as actual = s*R(psi)*desired + c
                       (scale, rotation, shift; Z as after "lag"). What's
                       left is error in the SHAPE of the path, not its
                       size/position. Returned as scale, rotation_deg,
                       shift_xy.
        What's left after "gain" / "similarity" is hysteresis, drift and
        anything else - see errorVsBend / stepReturn.'''
    d, run, ok, dt, curh, home = _prep(test, ctrl)
    des, cur = d["des"][run, :3], d["cur"][run, :3]
    straight = numpy.array([0.0, 0.0, HOME_Z])
    offset = home - straight
    cur_xy = cur - numpy.array([offset[0], offset[1], 0.0])
    steps = {"raw": _rmse(des[ok], cur[ok]),
             "session": _rmse(des[ok], cur_xy[ok]),
             "z_datum": _rmse(des[ok], curh[ok])}
    L, steps["lag"] = _bestLag(des, curh, ok, dt)
    m = ok[L:] & ok[:len(ok) - L]
    a = des[:len(des) - L][m] - straight
    b = curh[L:][m] - straight
    gain = float(numpy.sum(a*b)/numpy.sum(a*a))
    steps["gain"] = _rmse(gain*a, b)

    # Similarity fit in the XY plane as complex numbers: b = coef*a + shift
    za = a[:, 0] + 1j*a[:, 1]
    zb = b[:, 0] + 1j*b[:, 1]
    (coef, shift), *_ = numpy.linalg.lstsq(numpy.c_[za, numpy.ones_like(za)], zb, rcond=None)
    zfit = coef*za + shift
    fit = numpy.c_[zfit.real, zfit.imag, a[:, 2]]
    steps["similarity"] = _rmse(fit, b)

    out = {"test": test, "rmse": steps, "home_offset": offset, "lag_s": L*dt, "gain": gain,
           "scale": float(abs(coef)), "rotation_deg": float(numpy.degrees(numpy.angle(coef))),
           "shift_xy": numpy.array([shift.real, shift.imag])}
    if verbose:
        print(f"Test_{test}: " + "  ->  ".join(f"{k} {v:5.1f}" for k, v in steps.items())
              + f"   | home offset {numpy.round(offset, 1)} mm, lag {L*dt:.2f} s, gain {gain:.2f}"
              + f" | xy scale {out['scale']:.2f}, rotation {out['rotation_deg']:.1f}°, shift {numpy.round(out['shift_xy'], 1)} mm")
    return out


def plotErrorBudget(tests: list[int], ctrl: str = "PCC"):
    ''' RMSE after each errorBudget step, one group of bars per test. '''
    res = [errorBudget(t, ctrl) for t in tests]
    keys = list(res[0]["rmse"].keys())
    names = {"raw": "Raw", "session": "- session XY offset", "z_datum": "- Z datum",
             "lag": "- time lag", "gain": "- model gain (remaining)",
             "similarity": "- xy scale/rotation/shift (instead of gain)"}
    x = numpy.arange(len(tests))
    w = 0.8/len(keys)
    fig, ax = plt.subplots(figsize=(max(8, 1.2*len(tests)), 4.5))
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1)/2)*w, [r["rmse"][k] for r in res], w, label=names[k])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("RMSE (mm)")
    ax.set_title(f"{ctrl} error budget: RMSE left after removing each cause in turn")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    plt.show()


def main():
    # Run from the repo root (log paths are relative: PCC/Test_N/...)
    tests = [192,193,194,195,196,197,198,199,200,201,202,205,206,208,209,211,212,213,214,215,216,217,218] # From test day 3.1
    #trackingTable(tests)
    #plotSpeedStudy(tests)
    #plotErrorBudget(tests)
    # print("\n=== Error budget (RMSE mm after removing each cause) ===")
    # for t in tests:
    #     errorBudget(t)

    # print("\n=== Error vs bend, controller tests 136-137 ===")
    # errorVsBend([136, 137])


# Tests from before TestInfo.txt existed, for plotSpeedStudy's legend
OLD_LABELS = {**{t: "V-spiral" for t in (119, 120, 121, 122)}, **{t: "Figure 8" for t in (129, 130, 131, 132)}}

if __name__ == "__main__":
    main()
    plotTracking3D(194)                 # simple
    #plotTracking3D(162)                 # more curvature variation
    # plotSpeedStudy([119, 120, 121, 122, 129, 130, 131, 132], labels=OLD_LABELS)
    # plotErrorBudget([119, 120, 121, 122, 129, 130, 131, 132, 136])
    #plotErrorVsBend([136, 137])
    #plotStepReturn(test=203)
    #plotSpeedSweep(tests=range(91,102,1))
    #plotTrackingError(test=212)
