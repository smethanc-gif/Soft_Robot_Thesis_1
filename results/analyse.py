'''Processes the positional information stored using fileWriting.py and returns graphs and figures on the following information:\n
'''
import re
import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

try:
    from .Graphing import parse_log
except ImportError:
    from Graphing import parse_log




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
    tests = list(range(91, 102))
    print("=== Point 3: tracking error, Test 100 ===")
    m = trackingErrorMetrics(100)
    print(f"RMSE={m['rmse']:.2f}mm  mean={m['mean_error']:.2f}mm  max={m['max_error']:.2f}mm")

    print("\n=== Point 5: offset root-cause across Test 91-101 ===")
    result = modelResidualVsBend(tests)
    for t, r in result["per_test"].items():
        print(f"Test_{t}: offset={numpy.round(r['offset_mm'],1)}  "
              f"residual_mean={r['residual_mean_mm']:.1f}mm  "
              f"bend_range={r['bend_range_mm']}  corr={r['corr_bend_residual']:.2f}")
    print(f"\nPooled residual mean: {result['pooled_residual_mean_mm']:.1f}mm")
    print(f"Pooled corr(bend, residual): {result['pooled_corr_bend_residual']:.2f}")


if __name__ == "__main__":
    #main()
    #plotSpeedSweep(tests=range(91,102,1))
    plotTrackingError(test=101)
