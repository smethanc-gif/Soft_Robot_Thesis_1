import os
import re
## FILES

DESIRED_EGO_FILE = "DesiredMotorPosition.txt"
CURRENT_EGO_FILE = "CurrentMotorPosition.txt"
DESIRED_GLOB_FILE = "DesiredTipPosition.txt"
CALCULATED_GLOB_FILE = "CalculatedTipPosition.txt"
CURRENT_GLOB_FILE = "CurrentTipPosition.txt"

## FOLDERS

controlType = "default"  # set from the calling script, e.g. W.controlType = "PID"

_runFolder = None  # resolved on first write, then reused for the rest of the run
nextNum = 0
def _getRunFolder():
    """Return this run's 'Test_N' folder inside controlType, creating it on first call."""
    global _runFolder
    global nextNum
    if _runFolder is None:
        os.makedirs(controlType, exist_ok=True)
        testNums = []
        for name in os.listdir(controlType):
            if os.path.isdir(os.path.join(controlType, name)):
                m = re.fullmatch(r"Test_(\d+)", name)
                if m:
                    testNums.append(int(m.group(1)))
        nextNum = max(testNums) + 1 if testNums else 1
        
        _runFolder = os.path.join(controlType, f"Test_{nextNum}")
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

def writeAll(_desEgo: list[float], _currEgo: list[float], _desGlob: list[float], _calcGlob: list[float], _currGlob: list[float], _timestamp: float):
    fileNumber = writeDesEgo(_desEgo, _timestamp)
    writeCurrEgo(_currEgo, _timestamp)
    writeDesGlob(_desGlob, _timestamp)
    writeCalcGlob(_calcGlob, _timestamp)
    writeCurrGlob(_currGlob, _timestamp)
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