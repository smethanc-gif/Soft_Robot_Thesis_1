'''Reads the test logs written by Results/dataLogger.py.

Logs live in <logRoot>/<ctrl>/Test_N (Results/dataLogging.json), one file per
logged quantity, each row "t=<time>s  [v1, v2, ...]".
'''
import os
import re
import numpy as np
from Core.settings import loadDataLogging

LOGGING = loadDataLogging()
DATA_ROOT = LOGGING["logRoot"]
FILES = LOGGING["files"]
LINE_PATTERN = re.compile(r"t=(-?[\d.]+)s\s+\[([^\]]+)\]")


def getTestFolder(_test: int, _ctrl: str = "PCC") -> str:
    '''Folder holding one test's logs, e.g. Results/data/PCC/Test_196.'''
    return os.path.join(DATA_ROOT, _ctrl, f"{LOGGING['testFolderPrefix']}{_test}")


def getLogPath(_test: int, _ctrl: str, _fileKey: str) -> str:
    '''Path of one log file of a test, by its key in dataLogging.json (e.g. "desiredGlob").'''
    return os.path.join(getTestFolder(_test, _ctrl), FILES[_fileKey])


def parseLog(_filename: str, _ncols: int = 6) -> tuple[list[float], list[list[float]]]:
    '''Parses a log file into (times, [column0 values, column1 values, ...]).
        _ncols - values expected per line (6 for pose/motor logs, 4 for LimitsHit.txt).'''
    times = []
    columns = [[] for _ in range(_ncols)]

    with open(_filename, "r") as f:
        for line in f:
            match = LINE_PATTERN.match(line.strip())
            if not match:
                continue

            t = float(match.group(1))
            vals = [float(v.strip()) for v in match.group(2).split(",")]

            if len(vals) != _ncols:
                continue  # skip malformed lines

            times.append(t)
            for i in range(_ncols):
                columns[i].append(vals[i])

    return times, columns


def loadTest(_test: int, _ctrl: str = "PCC", _packup: bool = False) -> dict:
    '''All logs of one test as numpy arrays [sample][value], trimmed to a common length:
        "t" times, "des"/"cal"/"cur" desired/calculated/camera tip pose, "dm"/"cm" desired/measured
        motors, "config" configuration-space target and "lim" limit flags (the last two only for
        tests logged after CylinderParams.txt / LimitsHit.txt were added).
        Pack-up rows (logged after the task, from TestInfo.txt's packupStart on) are left out
        unless _packup=True, so t >= 0 is always just the task.'''
    d = {}
    for key, fileKey in [("des", "desiredGlob"), ("cal", "calculatedGlob"),
                         ("cur", "currentGlob"), ("dm", "desiredEgo"),
                         ("cm", "currentEgo"), ("config", "configParams")]:
        path = getLogPath(_test, _ctrl, fileKey)
        if os.path.exists(path):
            t, vals = parseLog(path)
            d[key] = np.array(vals).T
            if key == "des":
                d["t"] = np.array(t)
    limitsPath = getLogPath(_test, _ctrl, "limitsHit")
    if os.path.exists(limitsPath):
        _, vals = parseLog(limitsPath, _ncols=4)
        d["lim"] = np.array(vals).T
    n = min(len(v) for v in d.values())
    d = {k: v[:n] for k, v in d.items()}
    packupStart = loadMeta(_test, _ctrl).get("packupStart")
    if not _packup and isinstance(packupStart, (int, float)):
        task = d["t"] < packupStart
        d = {k: v[task] for k, v in d.items()}
    return d


def loadMeta(_test: int, _ctrl: str = "PCC") -> dict:
    '''Run settings from TestInfo.txt (see DataLogger.writeMeta), numbers converted.
        {} for tests logged before TestInfo.txt existed.'''
    path = getLogPath(_test, _ctrl, "testInfo")
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
    if "movementfreq" in meta:                  # key name in logs from before the rename
        meta["movementFreq"] = meta.pop("movementfreq")
    return meta
