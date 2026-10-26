'''Comparison of the three control methods for the thesis: PCC, cable-compensated PCC (PCC_Cable)
and the kinematic piecewise affine curvature model (Poly, "PAC" in the thesis).

Covers three things, all from the logged runs:
    accuracy      - tip RMSE per path, on matched trajectories, split into paths each method was
                    tuned or fitted on and paths it never saw, and checked across test days
    cost          - Pi control-step durations per method (same test days, so the Pi's state is shared)
                    and an offline replay of each method's IK solve time
    reliability   - how each run ended (pccResults.classifyRun), with every motor trip broken down into
                    a runaway (measured angle leaving its command) or a run that reached the motor clamp,
                    plus the motor-loop measures that go with it: how far each motor fell behind its
                    command, how big each command step was, and how jagged the command was

Error metric and run classes are pccResults.py's, unchanged (home-referenced tip error norm,
camera jumps > 200 mm rejected, t >= 0, RMSE per run; runs are only excluded for a camera fault,
stopping early or a motor trip).

Run sets (see RUNS, DISTURBANCE, CABLE_TUNING, POLY_V1 below):
    PCC        Test_138-300   final hardware build (as in the oral presentation)
    PCC_Cable  Test_1-94      1-33 are the 28 Sep tuning session (tendon gains changed between runs),
                              34-94 use the final gains [0.75, 0.75, 0.80, 0.60, 0.80, 0.70]
    Poly       Test_1-43      1-10 used the earlier constants (28 Sep), 11-43 the final fit (12 Oct)
    The 16 Oct hammer-strike runs (PCC 301-302, PCC_Cable 95-96, Poly 44-49) are left out of everything,
    and so are the S-shaped step-return runs (EXCLUDED_TYPES), which the thesis does not consider.

Test days come from testDays.json (the commit that first added each Test_N folder); rerun
writeTestDays() after pulling new tests.

Run from anywhere:
    python Results/analysis/methodComparison.py
Prints the tables and writes Results/analysis/output/methodComparison.json, which thesisFigures.py reads.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import re
import subprocess
import time
import warnings
import numpy as np
from scipy.signal import savgol_filter
from Results.logReader import loadTest, loadMeta, parseLog, getLogPath
from Results.analysis import pccResults as P

OUT_DIR = P.OUT_DIR
DAYS_FILE = pathlib.Path(__file__).resolve().parent / "testDays.json"
METHODS = ("PCC", "PCC_Cable", "Poly")
NAMES = {"PCC": "PCC", "PCC_Cable": "Cable-compensated PCC", "Poly": "PAC"}
RUNS = {"PCC": range(138, 301), "PCC_Cable": range(1, 95), "Poly": range(1, 44)}
DISTURBANCE = {"PCC": (301, 302), "PCC_Cable": (95, 96), "Poly": (44, 45, 46, 47, 48, 49)}
CABLE_TUNING = range(1, 34)        # 28 Sep: tendon gains adjusted between runs, on the r100 z280 S-shape circle
CABLE_TUNED_PATH = ("S-shape circle", 100, 280, 280)
POLY_V1 = range(1, 11)             # 28 Sep: constants from the earlier fit (gains 0.82/0.72/...)
SAT_RAD = 15/11                    # tracking error at which the PD loop asks for the drivers' 15 A limit (Kp = 11 A/rad)
JUMP_RAD = 3.0                     # a measured motor angle moving this far in one Pi step marks a runaway
PRE_ONSET_S = 0.2                  # samples this close to a runaway are left out of the "before" measures
COMPARE_DAYS = ("2026-10-13", "2026-10-15")   # all three methods were run on both days
EXCLUDED_TYPES = ("Step-return (S)", "Butterfly")   # not in the thesis test set (S step-returns: PCC 175, 204, Poly 30-32; butterfly: PCC 234-235, Cable 41)


# ---- Test days ----

def writeTestDays(_repo: str = ".") -> dict:
    '''Writes testDays.json: the date of the commit that first added each Test_N folder (any branch).
        Needs git and the repo's history; run from the repo root after pulling new tests.'''
    out = subprocess.run(["git", "-C", _repo, "log", "--all", "--diff-filter=A", "--name-only", "--format=@@%ad",
                          "--date=format:%Y-%m-%d"], capture_output=True, text=True, check=True).stdout
    days, day = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            day = line[2:]
            continue
        m = re.search(r"(?:^|/)(PCC_Cable|Poly|PCC|PD_test)/Test_(\d+)/", line)
        if m:
            key = f"{m.group(1)}/{m.group(2)}"
            days[key] = min(days.get(key, day), day)
    with open(DAYS_FILE, "w") as f:
        json.dump(days, f, indent=1)
    return days


def loadTestDays() -> dict:
    '''{"ctrl/test": "YYYY-MM-DD"} from testDays.json.'''
    with open(DAYS_FILE) as f:
        return json.load(f)


# ---- Which runs count for what ----

def getRunSet(_ctrl: str, _test: int) -> str:
    '''"evaluation", "tuning" (PCC_Cable 1-33), "earlier constants" (Poly 1-10) or "disturbance".'''
    if _test in DISTURBANCE[_ctrl]:
        return "disturbance"
    if _ctrl == "PCC_Cable" and _test in CABLE_TUNING:
        return "tuning"
    if _ctrl == "Poly" and _test in POLY_V1:
        return "earlier constants"
    return "evaluation"


def calcPathSize(_d: dict) -> dict:
    '''Size of the commanded path from its largest radius (circle and helix radius, figure 8 half-width),
        which, unlike the mean radius, is the same for a run that stopped early.'''
    des = _d["des"][_d["t"] >= 0]
    return {**P.describePath(_d), "r_max": float(np.hypot(des[:, 0], des[:, 1]).max())}


def getPathKey(_row: dict) -> tuple:
    '''Path shape without speed: (type, largest radius, highest and lowest height, tilt), rounded to 5 mm / 5 deg.'''
    return (_row["type"], int(round(_row["r_max"]/5)*5), int(round(_row["z_max"]/5)*5), int(round(_row["z_min"]/5)*5),
            int(round(_row["tilt_max"]/5)*5))


def getPolyTrainingPaths() -> set:
    '''Path keys of the runs the final Poly constants were fitted on (train_tests in polyConstants.json).'''
    with open(pathlib.Path(__file__).resolve().parents[2] / "Core" / "Poly" / "polyConstants.json") as f:
        train = json.load(f)["train_tests"]
    keys = set()
    for tests in train.values():
        for ctrl, test in tests:
            size = calcPathSize(loadTest(test, ctrl))
            keys.add(getPathKey({"type": P.trajectoryType(loadMeta(test, ctrl).get("trajectory"), size), **size}))
    return keys


def getSeenStatus(_row: dict, _polyPaths: set) -> str:
    '''Whether the run's path was used to set the method's constants: "tuned on" (PCC_Cable's
        tuning circle, or a path in Poly's training set) or "unseen". PCC has no fitted constants.'''
    if _row["ctrl"] == "PCC_Cable" and getPathKey(_row)[:4] == CABLE_TUNED_PATH:
        return "tuned on"
    if _row["ctrl"] == "Poly" and getPathKey(_row) in _polyPaths:
        return "fitted on"
    return "unseen" if _row["ctrl"] != "PCC" else "n/a"


# ---- Motor loop ----

def calcMotorLoop(_test: int, _ctrl: str) -> dict:
    '''Motor-level measures of one run (task only, t >= 0):
        runaway       - first sample where a measured motor angle is more than JUMP_RAD from its command,
                        jumps more than JUMP_RAD in one step or passes 19.5 rad, with the motor, time,
                        command and measured angle there, which side of the command the motor went
                        ("direction": -1 = paying out more cable than commanded), and whether it had
                        already started before the task (during the move to the start pose)
        track_max/p95 - largest / 95th-percentile motor tracking error |command - measured| over all
                        six motors, up to PRE_ONSET_S before any runaway (rad; nan if a runaway started
                        before the task)
        n_sat         - samples where that error was above SAT_RAD (PD loop at the current limit)
        step_p95/max  - 95th-percentile / largest command change between consecutive Pi steps (rad)
        jitter_cmd/meas - RMS of the command / measured angle about a 0.6 s Savitzky-Golay smoothing of
                        itself, worst motor (rad): how jagged the motor motion is, independent of the path'''
    d = loadTest(_test, _ctrl)
    run = d["t"] >= 0
    t, dm, cm = d["t"][run], d["dm"][run], d["cm"][run]
    off = np.abs(cm - dm)
    jump = (off > JUMP_RAD).any(axis=1) | np.r_[False, (np.abs(np.diff(cm, axis=0)) > JUMP_RAD).any(axis=1)] | (np.abs(cm) > 19.5).any(axis=1)
    out = {"runaway": bool(jump.any())}
    end = len(t)
    if jump.any():
        i = int(np.argmax(jump))
        stepNow = np.abs(cm[i] - cm[max(i - 1, 0)])
        diverged = off[i].max() > JUMP_RAD or stepNow.max() > JUMP_RAD
        j = int(np.argmax(off[i] + stepNow)) if diverged else int(np.argmax(np.abs(cm[i])))    # else: a motor reached 19.5 rad on command
        out.update({"diverged": bool(diverged), "onset_t": float(t[i]), "onset_motor": "ABCDEF"[j], "onset_cmd": float(dm[i, j]), "onset_meas": float(cm[i, j]),
                    "direction": int(np.sign(cm[i, j] - dm[i, j])), "in_setup": bool(i == 0),
                    "cmd_at_limit": bool(np.abs(dm[:i + 1]).max() > 19.5)})
        end = int(np.searchsorted(t, t[i] - PRE_ONSET_S))
    nan = float("nan")
    if end < 3:
        out.update({"track_max": nan, "track_p95": nan, "n_sat": 0, "step_p95": nan, "step_max": nan,
                    "jitter_cmd": nan, "jitter_meas": nan, "max_cmd": float(np.abs(dm).max())})
        return out
    e = off[:end].max(axis=1)
    step = np.abs(np.diff(dm[:end], axis=0)).max(axis=1)
    dt = float(np.median(np.diff(t[:end])))
    win = max(5, int(round(0.6/dt)) | 1)
    jitter = {}
    for key, q in (("cmd", dm[:end]), ("meas", cm[:end])):
        jitter[key] = float(np.sqrt(np.mean((q - savgol_filter(q, win, 2, axis=0))**2, axis=0)).max()) if len(q) > win + 2 else nan
    out.update({"track_max": float(e.max()), "track_p95": float(np.percentile(e, 95)), "n_sat": int((e > SAT_RAD).sum()),
                "step_p95": float(np.percentile(step, 95)), "step_max": float(step.max()),
                "jitter_cmd": jitter["cmd"], "jitter_meas": jitter["meas"], "max_cmd": float(np.abs(dm).max())})
    return out


def getTripKind(_row: dict) -> str:
    '''For a "motor trip" run: "runaway" (the measured angle left its command, opposite to it) or
        "command at limit" (the command itself reached the clamp and the motor followed).'''
    if _row["class"] != "motor trip":
        return ""
    return "command at limit" if _row.get("cmd_at_limit") and not _row.get("diverged", True) else "runaway"


# ---- All runs ----

def classifyAll(_verbose: bool = True) -> list[dict]:
    '''Every run of every method in RUNS and DISTURBANCE, classified by pccResults.classifyRun, plus
        its test day, run set, whether its path was seen in tuning/fitting, and the motor-loop measures.'''
    days = loadTestDays()
    polyPaths = getPolyTrainingPaths()
    rows = []
    for ctrl in METHODS:
        for test in list(RUNS[ctrl]) + list(DISTURBANCE[ctrl]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = P.classifyRun(test, ctrl)
            except Exception as ex:
                if _verbose:
                    print(f"{ctrl} Test_{test}: skipped ({ex})")
                continue
            r.pop("budget", None); r.pop("reach", None)          # fitted reach is not used in the thesis
            r["r_max"] = calcPathSize(loadTest(test, ctrl))["r_max"]
            r.update({"day": days.get(f"{ctrl}/{test}", "?"), "set": getRunSet(ctrl, test),
                      "path": getPathKey(r), "seen": getSeenStatus(r, polyPaths), **calcMotorLoop(test, ctrl)})
            r["trip_kind"] = getTripKind(r)
            if r["type"] in EXCLUDED_TYPES:
                r["set"] = "excluded type"
            rows.append(r)
    return rows


def selectRuns(_rows: list[dict], _ctrl: str = None, _set: str = "evaluation", _clean: bool = False) -> list[dict]:
    '''Runs of one method (or all) in one run set (None = every set except disturbance).'''
    return [r for r in _rows if (_ctrl is None or r["ctrl"] == _ctrl)
            and (r["set"] == _set if _set else r["set"] not in ("disturbance", "excluded type"))
            and (not _clean or r["class"] == "clean")]


# ---- Accuracy ----

def summariseRmse(_runs: list[dict]) -> dict:
    '''Median, IQR, range, run count and test days of a group of clean runs.'''
    rm = np.array([r["rmse"] for r in _runs])
    return {"n": len(_runs), "tests": [r["test"] for r in _runs], "days": sorted({r["day"] for r in _runs}),
            "median": float(np.median(rm)), "q1": float(np.percentile(rm, 25)), "q3": float(np.percentile(rm, 75)),
            "min": float(rm.min()), "max": float(rm.max()),
            "pct_length": float(100*np.median(rm)/P.ROBOT_LENGTH_MM),
            "speed": float(np.median([r["speed"] for r in _runs]))}


def summariseByType(_rows: list[dict]) -> dict:
    '''Clean evaluation runs of each method, by path type, split by whether the path was seen in tuning.'''
    out = {}
    for ctrl in METHODS:
        for r in selectRuns(_rows, ctrl, _clean=True):
            out.setdefault(ctrl, {}).setdefault(r["type"], {}).setdefault(r["seen"], []).append(r)
    return {c: {t: {s: summariseRmse(v) for s, v in bySeen.items()} for t, bySeen in byType.items()} for c, byType in out.items()}


MATCHED = [  # (label, path type, largest radius, highest z, lowest z, tilt, movementFreq or None for any)
    ("S-shape circle, r 100, z 250", "S-shape circle", 100, 250, 250, 0, 4),
    ("S-shape circle, r 100, z 260", "S-shape circle", 100, 260, 260, 0, 4),
    ("S-shape circle, r 100, z 270", "S-shape circle", 100, 270, 270, 0, 4),
    ("S-shape circle, r 100, z 280", "S-shape circle", 100, 280, 280, 0, 4),
    ("S-shape circle, r 100, z 300", "S-shape circle", 100, 300, 300, 0, 4),
    ("C-shape circle, r 150, z 250, 45°", "C-shape circle", 150, 250, 250, 45, None),
    ("C-shape circle, r 140, z 220, 60°", "C-shape circle", 140, 220, 220, 60, None),
    ("Figure 8, a 100, z 270", "Figure 8", 100, 270, 270, 0, 3),
    ("Figure 8, a 100, z 280", "Figure 8", 100, 280, 280, 0, 4),
    ("Helix, r 90, z 300→250", "Helix", 90, 300, 250, 0, 4),
]


def summariseMatched(_rows: list[dict]) -> list[dict]:
    '''Clean evaluation runs of each method on the same path and speed (MATCHED), with every run's
        RMSE and day, the per-day medians, and how many runs of that setting ended in a motor trip.'''
    out = []
    for label, typ, r, z, z0, tilt, mf in MATCHED:
        entry = {"label": label, "methods": {}}
        for ctrl in METHODS:
            pool = [x for x in selectRuns(_rows, ctrl) if x["path"][:5] == (typ, r, z, z0, tilt) and (mf is None or x["movementFreq"] == mf)]
            clean = [x for x in pool if x["class"] == "clean"]
            m = {"n_runs": len(pool), "n_trips": sum(x["class"] == "motor trip" for x in pool),
                 "runs": [(x["test"], x["day"], round(x["rmse"], 1)) for x in clean],
                 "seen": sorted({x["seen"] for x in pool})}
            if clean:
                m.update(summariseRmse(clean))
                m["by_day"] = {dd: float(np.median([x["rmse"] for x in clean if x["day"] == dd])) for dd in m["days"]}
            entry["methods"][ctrl] = m
        out.append(entry)
    return out


def summariseDayConsistency(_rows: list[dict], _minDays: int = 2) -> list[dict]:
    '''Settings (path and speed) that a method ran clean on at least _minDays test days: the median of
        each day and the spread between days, against the spread within a day.'''
    groups = {}
    for r in selectRuns(_rows, _clean=True):
        groups.setdefault((r["ctrl"], r["path"], r["movementFreq"]), []).append(r)
    out = []
    for (ctrl, path, mf), v in groups.items():
        byDay = {}
        for r in v:
            byDay.setdefault(r["day"], []).append(r["rmse"])
        if len(byDay) < _minDays:
            continue
        meds = {d: float(np.median(x)) for d, x in sorted(byDay.items())}
        within = [max(x) - min(x) for x in byDay.values() if len(x) > 1]
        out.append({"ctrl": ctrl, "path": list(path), "movementFreq": mf, "n": len(v), "day_medians": meds,
                    "between_days": max(meds.values()) - min(meds.values()),
                    "within_day_max": max(within) if within else None})
    return sorted(out, key=lambda g: (g["ctrl"], -g["n"]))


# ---- Cost ----

def loadStepTimes(_test: int, _ctrl: str) -> np.ndarray:
    '''Pi control-step durations (ms) of one run's task.'''
    t, _ = parseLog(getLogPath(_test, _ctrl, "desiredGlob"))
    t = np.array(t)
    return 1000*np.diff(t[t >= 0])


def summariseStepTimes(_rows: list[dict], _days: tuple = COMPARE_DAYS) -> dict:
    '''Step durations pooled over the evaluation runs of each method that ran to the end, on _days
        (None = every day): median, 95th percentile, share of steps over 40.5 ms and the mean rate.'''
    out = {}
    for ctrl in METHODS:
        runs = [r for r in selectRuns(_rows, ctrl) if r["class"] not in ("stopped early", "motor trip")
                and (_days is None or r["day"] in _days)]
        if not runs:
            continue
        a = np.concatenate([loadStepTimes(r["test"], ctrl) for r in runs])
        out[ctrl] = {"n_runs": len(runs), "n_steps": int(a.size), "median_ms": float(np.median(a)),
                     "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
                     "late_frac": float(np.mean(a > 1000*P.OVERRUN_S)), "rate_hz": float(1000/np.mean(a)),
                     "days": sorted({r["day"] for r in runs})}
    return out


def calcSolveTimes(_cases: dict = None, _repeats: int = 3) -> dict:
    '''Offline replay: each method's full IK (ContinuumRobot.solveFullIK, exactly as main.py calls it)
        over the logged desired poses of a few runs, timed per step on this computer. The Pi is slower,
        so these show the relative cost of the methods, not the Pi's absolute times. Wall-clock times vary
        by a few per cent from one run of this script to the next; the thesis quotes the medians of one run
        (PCC 0.77 ms, cable-compensated PCC 0.85 ms, PAC 2.53 ms, PAC 95th percentile 5.2 ms).'''
    from Core.continuumRobot import ContinuumRobot
    _cases = _cases or {"S-shape circle": ("PCC", 253), "Figure 8": ("PCC", 299), "Helix": ("PCC", 293)}
    out = {}
    for ctrl in METHODS:
        times = []
        for name, (srcCtrl, test) in _cases.items():
            t, vals = parseLog(getLogPath(test, srcCtrl, "desiredGlob"))
            poses = np.array(vals).T[np.array(t) >= 0]
            for _ in range(_repeats):
                robot = ContinuumRobot(ctrl)
                for pose in poses:
                    t0 = time.perf_counter()
                    robot.solveFullIK(list(pose), [0.0]*6)
                    times.append(1000*(time.perf_counter() - t0))
        a = np.array(times)
        out[ctrl] = {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()), "n": int(a.size)}
    return out


# ---- Reliability ----

def calcWilson(_k: int, _n: int, _z: float = 1.96) -> tuple[float, float]:
    '''95 % Wilson score interval of a proportion _k/_n.'''
    if _n == 0:
        return (float("nan"), float("nan"))
    p = _k/_n
    c = (p + _z**2/(2*_n))/(1 + _z**2/_n)
    h = _z*np.sqrt(p*(1 - p)/_n + _z**2/(4*_n**2))/(1 + _z**2/_n)
    return (float(c - h), float(c + h))


def summariseReliability(_rows: list[dict]) -> dict:
    '''Run outcomes per method (evaluation runs, and every run except the disturbance session), the
        runaway rate with its 95 % interval, where in the run runaways start, and the motor-loop
        measures of runs that ran away against runs that did not.'''
    out = {}
    for ctrl in METHODS:
        for key, sets in (("evaluation", "evaluation"), ("all", None)):
            runs = selectRuns(_rows, ctrl, sets)
            counts = {}
            for r in runs:
                counts[r["class"]] = counts.get(r["class"], 0) + 1
            run = [r for r in runs if r["trip_kind"] == "runaway"]
            lo, hi = calcWilson(len(run), len(runs))
            clean = [r for r in runs if r["class"] == "clean"]
            out.setdefault(ctrl, {})[key] = {
                "n": len(runs), "outcomes": counts, "n_runaway": len(run),
                "n_cmd_limit": sum(r["trip_kind"] == "command at limit" for r in runs),
                "runaway_rate": len(run)/len(runs) if runs else float("nan"), "ci95": [lo, hi],
                "runaways": [{"test": r["test"], "type": r["type"], "day": r["day"], "t": r.get("onset_t"), "motor": r.get("onset_motor"),
                              "cmd": r.get("onset_cmd"), "meas": r.get("onset_meas"), "track_max": r["track_max"], "direction": r.get("direction"),
                              "in_setup": r.get("in_setup")} for r in run],
                "n_paying_out": sum(r.get("direction") == -1 for r in run), "n_in_setup": sum(bool(r.get("in_setup")) for r in run),
                "onset_first_3s": sum(1 for r in run if r.get("onset_t", 99) < 3.0),
                "motor_counts": {m: sum(r.get("onset_motor") == m for r in run) for m in "ABCDEF"},
                "clean_median": {k: float(np.nanmedian([r[k] for r in clean])) for k in ("track_p95", "track_max", "step_p95", "jitter_cmd", "jitter_meas")} if clean else {},
                "runaway_median": {k: float(np.nanmedian([r[k] for r in run])) for k in ("track_p95", "track_max", "step_p95", "jitter_cmd", "jitter_meas")} if run else {},
                "sat_runs": {"clean": sum(r["n_sat"] > 0 for r in clean), "n_clean": len(clean),
                             "runaway": sum(r["n_sat"] > 0 for r in run), "n_runaway": len(run)}}
    return out


def summariseMotorLoopMatched(_rows: list[dict]) -> list[dict]:
    '''Motor-loop measures (medians over the runs that ran to the end) of each method on the MATCHED
        paths, so path size and speed are the same for every method: Pi step time, command step size,
        motor tracking error and how jagged the command and measured motor angles are.'''
    out = []
    for label, typ, r, z, z0, tilt, mf in MATCHED:
        entry = {"label": label, "methods": {}}
        for ctrl in METHODS:
            v = [x for x in selectRuns(_rows, ctrl) if x["path"][:5] == (typ, r, z, z0, tilt) and (mf is None or x["movementFreq"] == mf)
                 and x["class"] not in ("stopped early", "motor trip")]
            if v:
                entry["methods"][ctrl] = {"n": len(v), **{k: float(np.nanmedian([x[k] for x in v])) for k in
                                          ("dt_median", "dt_p95", "step_p95", "track_p95", "jitter_cmd", "jitter_meas")}}
        out.append(entry)
    return out


def summariseRunawayVsError(_rows: list[dict], _bins: tuple = (0, 0.5*SAT_RAD, SAT_RAD, 99)) -> dict:
    '''Runaway rate of the evaluation runs that reached the trajectory, by the largest motor
        tracking error seen before any runaway (bins in rad; SAT_RAD is the PD current limit).'''
    runs = [r for r in selectRuns(_rows, None, "evaluation") if r["class"] in ("clean", "motor trip", "camera: no data", "camera: offset")
            and np.isfinite(r["track_max"])]
    out = []
    for lo, hi in zip(_bins[:-1], _bins[1:]):
        g = [r for r in runs if lo <= r["track_max"] < hi]
        k = sum(r["trip_kind"] == "runaway" for r in g)
        out.append({"lo": lo, "hi": hi, "n": len(g), "runaways": k, "rate": k/len(g) if g else float("nan"), "ci95": calcWilson(k, len(g)),
                    "by_method": {c: (sum(r["trip_kind"] == "runaway" for r in g if r["ctrl"] == c), sum(r["ctrl"] == c for r in g)) for c in METHODS}})
    return {"bins": out}


# ---- Main ----

def main():
    rows = classifyAll()
    res = {"metric": "RMSE of the home-referenced, camera-gated tip error norm (mm), t >= 0 (pccResults.py)",
           "robot_length_mm": P.ROBOT_LENGTH_MM, "runs": rows,
           "by_type": summariseByType(rows), "matched": summariseMatched(rows),
           "day_consistency": summariseDayConsistency(rows),
           "step_times": {"compare_days": summariseStepTimes(rows), "all_days": summariseStepTimes(rows, None)},
           "solve_times_offline": calcSolveTimes(),
           "reliability": summariseReliability(rows), "runaway_vs_error": summariseRunawayVsError(rows),
           "motor_loop_matched": summariseMotorLoopMatched(rows)}

    print("\nClean evaluation runs by path (median RMSE mm [n, days]):")
    for ctrl in METHODS:
        for typ, bySeen in sorted(res["by_type"][ctrl].items()):
            for seen, s in bySeen.items():
                print(f" {ctrl:9s} {typ:18s} {seen:9s} {s['median']:6.1f} [{s['q1']:.0f}-{s['q3']:.0f}]  n={s['n']:2d}  days={len(s['days'])}")
    print("\nMatched paths (median RMSE mm, n clean / n runs, trips):")
    for e in res["matched"]:
        cells = []
        for ctrl in METHODS:
            m = e["methods"][ctrl]
            cells.append(f"{ctrl}: {m.get('median', float('nan')):5.1f} ({m.get('n', 0)}/{m['n_runs']}, {m['n_trips']} trips, {len(m.get('days', []))} d)")
        print(f" {e['label']:34s} " + " | ".join(cells))
    print("\nSame settings on more than one day:")
    for g in res["day_consistency"]:
        print(f" {g['ctrl']:9s} {g['path']} mf {g['movementFreq']}: n={g['n']}, day medians {g['day_medians']}, between {g['between_days']:.1f}, within {g['within_day_max']}")
    print("\nPi step durations, 13 and 15 Oct:")
    for ctrl, s in res["step_times"]["compare_days"].items():
        print(f" {ctrl:9s} {s['n_runs']:3d} runs: median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f}, {100*s['late_frac']:.0f}% late, {s['rate_hz']:.1f} Hz")
    print("Pi step durations, all evaluation runs:")
    for ctrl, s in res["step_times"]["all_days"].items():
        print(f" {ctrl:9s} {s['n_runs']:3d} runs: median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f}, {100*s['late_frac']:.0f}% late, {s['rate_hz']:.1f} Hz")
    print("Offline IK solve time (this computer):")
    for ctrl, s in res["solve_times_offline"].items():
        print(f" {ctrl:9s} median {s['median_ms']:.2f} ms, p95 {s['p95_ms']:.2f}, max {s['max_ms']:.1f}")
    print("\nReliability:")
    for ctrl in METHODS:
        for key in ("evaluation", "all"):
            s = res["reliability"][ctrl][key]
            print(f" {ctrl:9s} {key:10s} n={s['n']:3d} runaways {s['n_runaway']:2d} ({100*s['runaway_rate']:.0f}%, 95% CI {100*s['ci95'][0]:.0f}-{100*s['ci95'][1]:.0f}%), "
                  f"cmd at limit {s['n_cmd_limit']}, in first 3 s {s['onset_first_3s']}, motors {s['motor_counts']}, outcomes {s['outcomes']}")
            print(f"            clean medians {np.round(list(s['clean_median'].values()), 3)}  runaway medians {np.round(list(s['runaway_median'].values()), 3)}  sat runs {s['sat_runs']}")
    print("\nMotor loop on matched paths (medians: step ms, p95 step ms, cmd step p95 rad, track p95 rad, jitter cmd/meas rad):")
    for e in res["motor_loop_matched"]:
        for ctrl, m in e["methods"].items():
            print(f" {e['label']:34s} {ctrl:9s} n={m['n']:2d}  {1000*m['dt_median']:4.0f} {1000*m['dt_p95']:4.0f}  {m['step_p95']:.3f}  {m['track_p95']:.3f}  {m['jitter_cmd']:.4f} {m['jitter_meas']:.4f}")
    print("\nRunaway rate by largest tracking error before onset:")
    for b in res["runaway_vs_error"]["bins"]:
        print(f" {b['lo']:.2f}-{b['hi']:.2f} rad: {b['runaways']}/{b['n']} ({100*b['rate']:.0f}%)  {b['by_method']}")
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "methodComparison.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nWritten {OUT_DIR / 'methodComparison.json'}")
    return res


if __name__ == "__main__":
    main()
