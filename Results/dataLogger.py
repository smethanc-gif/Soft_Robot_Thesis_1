import os
import re
from Core.settings import loadDataLogging

## FILES - Results/dataLogging.json
LOGGING = loadDataLogging()
DATA_ROOT = LOGGING["logRoot"]      # logs go in <logRoot>/<controlType>/Test_N
FILES = LOGGING["files"]
DESIRED_EGO_FILE = FILES["desiredEgo"]
CURRENT_EGO_FILE = FILES["currentEgo"]
DESIRED_GLOB_FILE = FILES["desiredGlob"]
CALCULATED_GLOB_FILE = FILES["calculatedGlob"]
CURRENT_GLOB_FILE = FILES["currentGlob"]
CYL_FILE = FILES["configParams"]
LIMITS_FILE = FILES["limitsHit"]
META_FILE = FILES["testInfo"]

## FOLDERS

controlType = "default"  # set from the calling script, e.g. W.controlType = "PID"

_runFolder = None  # resolved on first write, then reused for the rest of the run
nextNum = 0
def _getRunFolder():
    """Return this run's 'Test_N' folder inside controlType, creating it on first call."""
    global _runFolder
    global nextNum
    if _runFolder is None:
        os.makedirs(os.path.join(DATA_ROOT, controlType), exist_ok=True)
        testNums = []
        for name in os.listdir(os.path.join(DATA_ROOT, controlType)):
            if os.path.isdir(os.path.join(DATA_ROOT, controlType, name)):
                m = re.fullmatch(re.escape(LOGGING["testFolderPrefix"]) + r"(\d+)", name)
                if m:
                    testNums.append(int(m.group(1)))
        nextNum = max(testNums) + 1 if testNums else 1
        
        _runFolder = os.path.join(DATA_ROOT, controlType, f"{LOGGING['testFolderPrefix']}{nextNum}")
        os.makedirs(_runFolder)
    return _runFolder, nextNum

def _path(filename: str, testing=False) -> str:
    runFolder, fileNumber = _getRunFolder()
    if testing:
        return os.path.join(runFolder, filename), fileNumber
    if not testing:
        return os.path.join(runFolder, filename)

## FUNCTIONS
def writeDesEgo(positions: list[float], timestamp: float):
    filepath, fileNumber = _path(DESIRED_EGO_FILE, testing=True)
    with open(filepath, "a") as f:
        
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")
    return fileNumber

def writeCurrEgo(positions: list[float], timestamp: float):
    filepath = _path(CURRENT_EGO_FILE)
    with open(filepath, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeDesGlob(positions: list[float], timestamp: float):
    filepath = _path(DESIRED_GLOB_FILE)
    with open(filepath, "a") as f:
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeCalcGlob(positions: list[float], timestamp: float):
    filepath = _path(CALCULATED_GLOB_FILE)
    with open(filepath, "a") as f:
    
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeCurrGlob(positions: list[float], timestamp: float):
    filepath = _path(CURRENT_GLOB_FILE)
    with open(filepath, "a") as f:
    
        vals = ", ".join(f"{p:.4f}" for p in positions)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeCyl(params: list[float], timestamp: float):
    filepath = _path(CYL_FILE)
    with open(filepath, "a") as f:
        vals = ", ".join(f"{p:.6f}" for p in params)   # curvature is ~1e-3, needs the extra digits
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeLimits(flags: list[int], timestamp: float):
    filepath = _path(LIMITS_FILE)
    with open(filepath, "a") as f:
        vals = ", ".join(f"{int(p)}" for p in flags)
        f.write(f"t={timestamp:.3f}s  [{vals}]\n")

def writeMeta(info: dict, append: bool = False):
    ''' Settings that are fixed for the whole run (trajectory, speed, ...),
        written once per test as "key = value" lines. Adds to the per-timestep
        logs from writeAll, doesn't replace them. append=True adds lines to
        the existing file instead (e.g. the pack-up start time, only known at
        the end of the run).'''
    filepath = _path(META_FILE)
    with open(filepath, "a" if append else "w") as f:
        for key, value in info.items():
            f.write(f"{key} = {value}\n")

def writeAll(_desEgo: list[float], _currEgo: list[float], _desGlob: list[float], _calcGlob: list[float], _currGlob: list[float], _timestamp: float):
    working = True
    if _desEgo is None:
        working = False
    elif _currEgo is None:
        working = False
    elif _desGlob is None:
        working = False
    elif _calcGlob is None:
        working = False
    elif _currGlob is None:
        working = False
    else:
        fileNumber = writeDesEgo(_desEgo, _timestamp)
        writeCurrEgo(_currEgo, _timestamp)
        writeDesGlob(_desGlob, _timestamp)
        writeCalcGlob(_calcGlob, _timestamp)
        writeCurrGlob(_currGlob, _timestamp)

    if not working:
        print(f"Failed to write for timestamp {_timestamp}s.")
    return fileNumber

def clear_files():
    open(_path(DESIRED_EGO_FILE), "w").close()
    open(_path(CURRENT_EGO_FILE), "w").close()
    open(_path(DESIRED_GLOB_FILE), "w").close()
    open(_path(CALCULATED_GLOB_FILE), "w").close()
    open(_path(CURRENT_GLOB_FILE), "w").close()

if __name__ == "__main__":
    controlType = "test_test"
    writeAll([0, 0],[0, 0],[0, 0],[0, 0],[0, 0],0)
    clear_files()