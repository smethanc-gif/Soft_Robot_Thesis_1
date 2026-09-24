'''Processes the positional information stored using fileWriting.py and returns graphs and figures on the following information:\n
'''
import os
import sys
import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

try:
    from .Graphing import parse_log, GlobErrorMetrics as trackingErrorMetrics, plotGlobError as plotTrackingError
except ImportError:
    from Graphing import parse_log, GlobErrorMetrics as trackingErrorMetrics, plotGlobError as plotTrackingError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, for PCC_control
from PCC_control.CTRL import ContinuumRobot

HOME_Z = 327.0  # straight tool-tip height of the PCC model: 2*sectL + toolLength


# ============================================================
# Loading / cleaning
# ============================================================

def loadTest(test: int, ctrl: str = "PCC") -> dict:
    ''' All logs of one test as numpy arrays [sample][value], trimmed to a
        common length. "cyl" and "lim" are only present for tests logged after
        CylinderParams.txt / LimitsHit.txt were added.'''
    folder = f"{ctrl}/Test_{test}"
    d = {}
    for key, name in [("des", "DesiredTipPosition"), ("cal", "CalculatedTipPosition"),
                      ("cur", "CurrentTipPosition"), ("dm", "DesiredMotorPosition"),
                      ("cm", "CurrentMotorPosition"), ("cyl", "CylinderParams")]:
        path = f"{folder}/{name}.txt"
        if os.path.exists(path):
            t, vals = parse_log(path)
            d[key] = numpy.array(vals).T
            if key == "des":
                d["t"] = numpy.array(t)
    if os.path.exists(f"{folder}/LimitsHit.txt"):
        _, vals = parse_log(f"{folder}/LimitsHit.txt", ncols=4)
        d["lim"] = numpy.array(vals).T
    n = min(len(v) for v in d.values())
    return {k: v[:n] for k, v in d.items()}


def cleanMask(d: dict, motor_tol: float = 0.5) -> numpy.ndarray:
    ''' True for samples worth trusting: every motor within motor_tol rad of
        its target (rules out the STM32 state freezes/jumps seen in e.g. Test
        108, 123, 137) and a fresh camera frame (main.py repeats the previous
        position on a missed frame).'''
    motor_ok = numpy.abs(d["dm"] - d["cm"]).max(axis=1) <= motor_tol
    fresh = numpy.r_[True, numpy.any(numpy.diff(d["cur"][:, :3], axis=0) != 0, axis=1)]
    return motor_ok & fresh


def homeRef(d: dict, pre_t: float = -1.8, z0: float = HOME_Z) -> tuple[numpy.ndarray, numpy.ndarray]:
    ''' Camera tip positions shifted so the straight pre-start pose (t < pre_t,
        before CTRL_INIT has moved the motors) sits at (0, 0, z0). Removes the
        per-session marker pose-lock offset. Returns (shifted xyz, raw home).'''
    home = d["cur"][d["t"] < pre_t, :3].mean(axis=0)
    return d["cur"][:, :3] - home + numpy.array([0.0, 0.0, z0]), home


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
        pooled over `tests`, cleaned (cleanMask) and home-referenced (homeRef).
        gain = actual tip displacement from straight / model displacement:
        <1 the robot under-bends, >1 it is further out than commanded
        (at low bend that means it hasn't come back - hysteresis).'''
    pooled = {k: [] for k in ("bend", "raw", "home", "dmod", "dact", "dz", "lim")}
    for t in tests:
        d = loadTest(t, ctrl)
        curh, home = homeRef(d)
        idx, cyl, fk, lim = bendParams(d, step)
        keep = cleanMask(d)[idx] & (d["t"][idx] >= 0)
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
            print(f"Test_{t}: {keep.sum()} clean samples, session home offset {numpy.round(home - straight, 1)} mm")
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
        desired pose (no dependency on main.py's constants): a rest is a run
        of desired bend ~0 after t=0, and it belongs to the bend just before
        it. The first settle_s of each rest is skipped.
          residual     - home-referenced actual - model tip, mean over the rest
          along        - residual component along the prior bend direction
                         (positive = still bent that way)
          slack_mm     - the residual converted to equivalent per-tendon cable
                         length error [A..F] via the PCC IK: how much each
                         tendon is effectively longer (+) or shorter (-) than
                         commanded. This is what backlash compensation needs.'''
    d = loadTest(test, ctrl)
    curh, _ = homeRef(d)
    ok = cleanMask(d)
    bend = _bendFromPose(d["des"])
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


def plotStepReturn(test: int, ctrl: str = "PCC"):
    ''' Polar plot of the straight-hold residual after each bend: the arrow
        from the origin shows where the tip is left, coloured by prior bend
        angle, with a marker at the prior bend direction.'''
    res = stepReturn(test, ctrl)
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
    _, desiredPos = parse_log(f"{ctrl}/Test_{test}/DesiredTipPosition.txt")
    _, currentPos = parse_log(f"{ctrl}/Test_{test}/CurrentTipPosition.txt")
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
    _, calcPos = parse_log(f"{ctrl}/Test_{test}/CalculatedTipPosition.txt")
    _, curPos = parse_log(f"{ctrl}/Test_{test}/CurrentTipPosition.txt")
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
        _, calcPos = parse_log(f"{ctrl}/Test_{t}/CalculatedTipPosition.txt")
        _, curPos = parse_log(f"{ctrl}/Test_{t}/CurrentTipPosition.txt")
        _, desPos = parse_log(f"{ctrl}/Test_{t}/DesiredTipPosition.txt")
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


def main():
    # Run from the repo root (log paths are relative: PCC/Test_N/...)
    print("=== Tracking error, Test 136 ===")
    m = trackingErrorMetrics(136)
    print(f"RMSE={m['rmse']:.2f}mm  mean={m['mean_error']:.2f}mm  max={m['max_error']:.2f}mm\n")

    print("=== Error vs bend, controller tests 136-137 ===")
    errorVsBend([136, 137])


if __name__ == "__main__":
    main()
    #plotErrorVsBend([136, 137])
    #plotStepReturn(test=138)
    #plotSpeedSweep(tests=range(91,102,1))
    #plotTrackingError(test=101)
