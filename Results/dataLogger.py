'''Writes the logs of one run on the robot (used by main.py).

Each run gets the next free folder <logRoot>/<controlType>/Test_N, with one
file per logged quantity; folder and file names come from
Results/dataLogging.json. Every row is "t=<time>s  [v1, v2, ...]".
Results/logReader.py reads them back.
'''
import os
import re
from Core.settings import loadDataLogging

LOGGING = loadDataLogging()
FILES = LOGGING["files"]


class DataLogger:
    def __init__(self, _controlType: str):
        '''Logger for one run of the given control method (moveMode), e.g. "PCC".'''
        self.controlType = _controlType
        self.runFolder = None               # resolved on first write, then reused for the rest of the run
        self.testNumber = None

    def getRunFolder(self) -> str:
        '''Returns this run's Test_N folder, creating it (with the next free N) on the first call.'''
        if self.runFolder is None:
            controlFolder = os.path.join(LOGGING["logRoot"], self.controlType)
            os.makedirs(controlFolder, exist_ok=True)
            testNums = []
            for name in os.listdir(controlFolder):
                if os.path.isdir(os.path.join(controlFolder, name)):
                    m = re.fullmatch(re.escape(LOGGING["testFolderPrefix"]) + r"(\d+)", name)
                    if m:
                        testNums.append(int(m.group(1)))
            self.testNumber = max(testNums) + 1 if testNums else 1
            self.runFolder = os.path.join(controlFolder, f"{LOGGING['testFolderPrefix']}{self.testNumber}")
            os.makedirs(self.runFolder)
        return self.runFolder

    def _writeRow(self, _fileKey: str, _values: list[str], _timestamp: float):
        '''Appends one "t=<time>s  [values]" row to the log file named by _fileKey in dataLogging.json.'''
        with open(os.path.join(self.getRunFolder(), FILES[_fileKey]), "a") as f:
            f.write(f"t={_timestamp:.3f}s  [{', '.join(_values)}]\n")

    def writeAll(self, _desEgo: list[float], _currEgo: list[float], _desGlob: list[float], _calcGlob: list[float], _currGlob: list[float], _timestamp: float) -> int:
        '''Logs one control step: desired and measured motor angles, desired, calculated and measured tip pose. Returns the test number.'''
        rows = (("desiredEgo", _desEgo), ("currentEgo", _currEgo), ("desiredGlob", _desGlob),
                ("calculatedGlob", _calcGlob), ("currentGlob", _currGlob))
        if any(values is None for _, values in rows):
            print(f"Failed to write for timestamp {_timestamp}s.")
        else:
            for fileKey, values in rows:
                self._writeRow(fileKey, [f"{p:.4f}" for p in values], _timestamp)
        return self.testNumber

    def writeConfig(self, _params: list[float], _timestamp: float):
        '''Logs the configuration-space target [k1, r1, s1, k2, r2, s2] (curvature is ~1e-3, so 6 decimals).'''
        self._writeRow("configParams", [f"{p:.6f}" for p in _params], _timestamp)

    def writeLimits(self, _flags: list[int], _timestamp: float):
        '''Logs which PCC limits clamped the target [s1, s2, bend1, bend2] as 0/1.'''
        self._writeRow("limitsHit", [f"{int(p)}" for p in _flags], _timestamp)

    def writeMeta(self, _info: dict, _append: bool = False):
        '''Writes settings that are fixed for the whole run (trajectory, speed, ...) as "key = value"
           lines. _append=True adds to the existing file instead of starting it.'''
        with open(os.path.join(self.getRunFolder(), FILES["testInfo"]), "a" if _append else "w") as f:
            for key, value in _info.items():
                f.write(f"{key} = {value}\n")
