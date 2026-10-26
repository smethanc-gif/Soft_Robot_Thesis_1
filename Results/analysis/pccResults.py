'''PCC results for the thesis and the oral presentation.

Every logged PCC run is classified (clean / camera fault / stopped early / motor trip) and the clean
runs are summarised by trajectory type: tip error, error budget, time lag, speed and loop timing.

Error metric (the same as calcTrackingMetrics in analyse.py):
    e(t) = || p_desired(t) - p_camera(t) ||     tip error norm per camera sample (mm)
    - camera positions home-referenced to the straight pose logged before each run (calcHomeRef)
    - camera samples that jump > 200 mm from the last accepted reading are rejected (getCameraGateMask)
    - only the task itself (t >= 0)
    The run is summarised as RMSE = sqrt(mean(e^2)), with the mean and maximum alongside.
    No individual samples are dropped for being large; a run is excluded as a whole only if the
    camera failed (see classifyRun).

Run from anywhere:
    python Results/analysis/pccResults.py
Prints the tables and writes Results/analysis/output/pccResults.json, which
presentationFigures.py reads.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import warnings
import numpy as np
from Results.logReader import loadTest, loadMeta, getTestFolder
from Results.analysis import analyse as A

OUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
CTRL = "PCC"
FIRST_TEST, LAST_TEST = 138, 300   # PCC tests from the final hardware build (earlier tests used older firmware/settings)
STEP_S = 0.04                      # Pi control step (25 Hz)
OVERRUN_S = 0.0405                 # a step counts as late above this
TRIP_RAD = 19.5                    # measured motor angle beyond this = the STM32's 20 rad safety trip
ROBOT_LENGTH_MM = A.HOME_Z         # straight tool-tip height, 327 mm (2 x 145 mm sections + 30 mm tool + base offset)

TRAJ_NAMES = {0: "Off-axis tilted circle", 1: "Vertical spiral", 2: "Vertical rotate", 3: "Controller",
              5: "Butterfly", 7: "Figure 8", 8: "Helix", 9: "Step-return (C)", 10: "Step-return (S)"}


# ---- One run ----

def describePath(_d: dict) -> dict:
    '''Size of the commanded path (t >= 0): mean radius from the z axis, height range and
        maximum tool tilt from vertical.'''
    des = _d["des"][_d["t"] >= 0]
    rxy = np.hypot(des[:, 0], des[:, 1])
    tilt = np.degrees(np.arccos(np.clip(np.cos(des[:, 3])*np.cos(des[:, 4]), -1, 1)))
    return {"r_mean": float(rxy.mean()), "z_min": float(des[:, 2].min()), "z_max": float(des[:, 2].max()),
            "tilt_max": float(tilt.max())}


def trajectoryType(_traj, _path: dict) -> str:
    '''Readable trajectory name. Trajectory 4 (centred circle) is split by tool tilt: tilted with the
        motion = C-shape (both sections bend the same way), vertical = S-shape.'''
    if _traj == 4:
        return "C-shape circle" if _path["tilt_max"] > 5 else "S-shape circle"
    return TRAJ_NAMES.get(_traj, f"Trajectory {_traj}")


def calcLoopTiming(_d: dict) -> dict:
    '''Pi control-step durations during the task (from the logged step times).'''
    t = _d["t"][_d["t"] >= 0]
    dt = np.diff(t)
    return {"dt_mean": float(dt.mean()), "dt_median": float(np.median(dt)), "dt_p95": float(np.percentile(dt, 95)),
            "dt_max": float(dt.max()), "frac_over": float((dt > OVERRUN_S).mean()), "rate_hz": float(1/dt.mean()),
            "n_steps": int(dt.size), "n_late": int((dt > OVERRUN_S).sum())}


def classifyRun(_test: int, _ctrl: str = CTRL) -> dict:
    '''Outcome and metrics of one run. Classes, checked in this order:
        motor trip     - a measured motor angle passed the 20 rad safety trip
        stopped early  - the log ends before 95 % of the task time
        camera: no data- under half the camera samples usable (rejected, all-zero or frozen)
        camera: offset - median error over 100 mm: the tracker followed the wrong object or the
                         marker pose lock was off for the whole run
        clean          - everything else; only clean runs are used for accuracy results.'''
    meta = loadMeta(_test, _ctrl)
    d = loadTest(_test, _ctrl)
    run = d["t"] >= 0
    path = describePath(d)
    row = {"test": _test, "ctrl": _ctrl, "traj": meta.get("trajectory"), "type": trajectoryType(meta.get("trajectory"), path),
           "movementFreq": meta.get("movementFreq"), "taskTime": meta.get("taskTime"), **path, **calcLoopTiming(d),
           "t_end": float(d["t"][run][-1])}
    taskTime = meta.get("taskTime") if isinstance(meta.get("taskTime"), (int, float)) else 30
    cam = d["cur"][run, :3]
    zero = float(np.all(np.abs(cam) < 1e-6, axis=1).mean())
    frozen = float((np.linalg.norm(np.diff(cam, axis=0), axis=1) < 1e-6).mean())
    maxMotor = float(np.abs(d["cm"][run]).max()) if "cm" in d else float("nan")   # nan: motor log missing
    tripped = bool(maxMotor > TRIP_RAD)

    m = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = A.calcTrackingMetrics(_test, _ctrl)
    except Exception as ex:          # e.g. no usable camera samples at all
        row["metrics_error"] = str(ex)
    coverage = m["n"]/(m["n"] + m["n_rejected"]) if m else 0.0
    medErr = float(np.median(m["error_home"])) if m else None

    if tripped:
        cls = "motor trip"
    elif row["t_end"] < 0.95*taskTime:
        cls = "stopped early"
    elif m is None or coverage < 0.5 or zero > 0.5 or frozen > 0.5:
        cls = "camera: no data"
    elif medErr > 100:
        cls = "camera: offset"
    else:
        cls = "clean"
    row.update({"class": cls, "coverage": coverage, "median_error": medErr, "max_motor_rad": maxMotor})

    if m is not None:
        row.update({"rmse": m["rmse_home"], "mean_error": m["mean_error_home"], "max_error": m["max_error_home"],
                    "rmse_raw": m["rmse"], "lag_s": m["lag_s"], "rmse_lag_removed": m["rmse_lag_removed"],
                    "phase_deg": m["phase_deg"], "period_s": m["period_s"], "speed": m["speed"]})
        if cls == "clean" and meta.get("trajectory") not in (9, 10):
            b = A.calcErrorBudget(_test, _ctrl, _verbose=False)
            row.update({"budget": b["rmse"], "reach": b["gain"]})
    return row


# ---- Groups of runs ----

def settingsKey(_row: dict) -> tuple:
    '''Runs with the same trajectory, speed and path size - repeats of one experiment.'''
    return (_row["type"], _row["movementFreq"], round(_row["r_mean"]/5)*5, round(_row["z_min"]/5)*5,
            round(_row["z_max"]/5)*5, round(_row["tilt_max"]/5)*5)


def summarise(_rows: list[dict], _key: str = "type") -> dict:
    '''Median and spread of the clean-run metrics per group (default: per trajectory type).'''
    groups = {}
    for r in _rows:
        if r["class"] == "clean" and "rmse" in r:
            groups.setdefault(r[_key] if _key != "settings" else str(settingsKey(r)), []).append(r)
    out = {}
    for k, v in groups.items():
        rm = np.array([r["rmse"] for r in v])
        s = {"n": len(v), "tests": [r["test"] for r in v],
             "rmse_median": float(np.median(rm)), "rmse_q1": float(np.percentile(rm, 25)), "rmse_q3": float(np.percentile(rm, 75)),
             "rmse_min": float(rm.min()), "rmse_max": float(rm.max()),
             "max_error_median": float(np.median([r["max_error"] for r in v])),
             "lag_median": float(np.median([r["lag_s"] for r in v])),
             "rmse_lag_removed_median": float(np.median([r["rmse_lag_removed"] for r in v])),
             "speed_median": float(np.median([r["speed"] for r in v])),
             "rmse_pct_length": float(100*np.median(rm)/ROBOT_LENGTH_MM)}
        if all("reach" in r for r in v):
            s["reach_median"] = float(np.median([r["reach"] for r in v]))
            s["budget_median"] = {b: float(np.median([r["budget"][b] for r in v])) for b in v[0]["budget"]}
        out[k] = s
    return out


def outcomeCounts(_rows: list[dict]) -> dict:
    '''How many runs ended in each class.'''
    out = {}
    for r in _rows:
        out[r["class"]] = out.get(r["class"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def timingSummary(_rows: list[dict]) -> dict:
    '''Loop timing over all runs that ran to the end (any class except stopped early / motor trip).'''
    v = [r for r in _rows if r["class"] not in ("stopped early", "motor trip")]
    return {"n_runs": len(v), "dt_median_ms": 1000*float(np.median([r["dt_median"] for r in v])),
            "dt_mean_ms": 1000*float(np.mean([r["dt_mean"] for r in v])),
            "dt_p95_ms": 1000*float(np.median([r["dt_p95"] for r in v])),
            "frac_over": sum(r["n_late"] for r in v)/sum(r["n_steps"] for r in v),   # pooled over every step
            "rate_hz": float(np.mean([r["rate_hz"] for r in v]))}


def repeatGroups(_rows: list[dict], _minRuns: int = 3) -> list[dict]:
    '''Clean runs with identical settings (settingsKey), for repeatability: RMSE spread per group.'''
    g = summarise(_rows, "settings")
    return sorted([{"settings": k, **v} for k, v in g.items() if v["n"] >= _minRuns], key=lambda x: -x["n"])


def main():
    rows = []
    for t in range(FIRST_TEST, LAST_TEST + 1):
        try:
            rows.append(classifyRun(t))
        except Exception as ex:
            print(f"Test_{t}: skipped ({ex})")
    res = {"metric": "RMSE of the home-referenced, camera-gated tip error norm (mm), t >= 0",
           "robot_length_mm": ROBOT_LENGTH_MM, "runs": rows, "outcomes": outcomeCounts(rows),
           "by_type": summarise(rows), "timing": timingSummary(rows), "repeats": repeatGroups(rows)}

    print(f"\n{len(rows)} PCC runs (Test_{FIRST_TEST}-{LAST_TEST}): {res['outcomes']}")
    print("\nClean runs by trajectory type (RMSE of tip error norm, mm):")
    print(f" {'type':24s}  n  median [IQR]          min-max     lag-rem  lag s  reach  speed mm/s  %L")
    for k, s in sorted(res["by_type"].items(), key=lambda kv: -kv[1]["n"]):
        print(f" {k:24s} {s['n']:2d}  {s['rmse_median']:5.1f} [{s['rmse_q1']:5.1f}-{s['rmse_q3']:5.1f}]  "
              f"{s['rmse_min']:5.1f}-{s['rmse_max']:5.1f}  {s['rmse_lag_removed_median']:6.1f}  {s['lag_median']:5.2f}  "
              f"{s.get('reach_median', float('nan')):5.2f}  {s['speed_median']:8.0f}  {s['rmse_pct_length']:4.1f}")
    tm = res["timing"]
    print(f"\nLoop timing over {tm['n_runs']} runs: median step {tm['dt_median_ms']:.1f} ms, mean {tm['dt_mean_ms']:.1f} ms, "
          f"p95 {tm['dt_p95_ms']:.1f} ms, {100*tm['frac_over']:.0f}% of steps late, {tm['rate_hz']:.1f} Hz")
    print("\nRepeated settings (>= 3 clean runs):")
    for g in res["repeats"]:
        print(f" {g['settings']}: n={g['n']}, RMSE {g['rmse_min']:.1f}-{g['rmse_max']:.1f} (median {g['rmse_median']:.1f}), tests {g['tests']}")
    print("\nRuns that were not clean:")
    for r in rows:
        if r["class"] != "clean":
            print(f" Test_{r['test']}: {r['class']:16s} {r['type']:22s} coverage {r['coverage']:.2f}, median error "
                  f"{r['median_error'] if r['median_error'] is None else round(r['median_error'], 1)}, max motor {r['max_motor_rad']:.1f} rad")
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "pccResults.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nWritten {OUT_DIR / 'pccResults.json'}")


if __name__ == "__main__":
    main()
