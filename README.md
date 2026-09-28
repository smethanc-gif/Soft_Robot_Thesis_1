# Soft_Robot_Thesis
Control codebase for a 2-section, 6-tendon soft continuum robot. It compares three control methods: pure PCC, PCC with cable compensation, and a fitted polynomial-curvature model.

A Raspberry Pi runs `main.py` and talks to an STM32 motor controller over UART. A RealSense camera with an ArUco marker tracks the tip.

## Layout
```
main.py                          runs one test on the robot (Pi): setup -> moveToStartPos -> runControlLoop -> packUp
Core/                            control
  controlSettings.json           what to run: moveMode, trajectory, taskTime, movementFreq, timeStep, setup/pack-up timings
  robotGeometry.json             the robot's dimensions and limits
  settings.py                    loads the JSON settings files
  continuumRobot.py              ContinuumRobot: picks the control method, tool-pose -> motor targets
  trajectories.py                the tip trajectories (calcTargGlob), step-return schedule, PD_test profiles
  PCC/constantCurvature.py       "PCC"       - piecewise constant curvature
  PCC_Cable/cableCompensation.py "PCC_Cable" - PCC with each cable's travel corrected by its fitted gain
  Poly/polyCurvature.py          "Poly"      - fitted polynomial-curvature model, inverted for control
  Poly/polyModel.py              the fitted forward model (shared by PCC_Cable, Poly and polyFit)
  Poly/polyConstants.json        fitted constants (written by Results/calibration/polyFit.py)
  testing/                       offline checks and simulations: checkIK, robotSim, workspaceSim
Results/                         logging and analysis
  dataLogging.json               where logs go and what each log file is called
  dataLogger.py                  DataLogger: writes one run's logs (used by main.py)
  logReader.py                   parseLog, loadTest, loadMeta: reads the logs back
  analysis/                      analyse.py (tracking, error vs bend, step-return, error budget), graphing.py
  calibration/                   polyFit.py, polyFigures.py, polyConstants.md (how the Poly constants were found)
  data/<moveMode>/Test_N/        the test logs
STM32_comms/                     Pi <-> STM32 UART link (piComm.py) and connection tests
camera_feedback/                 tip tracker (robotTracker.py) and the HSV tuner
controller/                      Xbox controller input (ctrlInput.py, trajectory 3)
```

## Running a test (on the Pi)
1. Set `moveMode`, `trajectory` and the timings in `Core/controlSettings.json`. Each value is explained in the file's `notes`.
2. Run `python main.py` from the repo root.
3. The logs are written to `Results/data/<moveMode>/Test_N/`, with every setting recorded in `TestInfo.txt`.

## Analysis (on the PC)
Run from the repo root, e.g. in a Python session:
```python
from Results.analysis import analyse as A
A.printTrackingTable([196, 199])
A.plotTracking3D(194)
A.plotStepReturn(203, _isCartesian=False)
A.calcErrorBudget(163, _ctrl="Poly")
```
Any script with a `main` can also be run directly, from any folder:
```
python Core/testing/checkIK.py              # offline check of the three control methods (also run it on the Pi for timing)
python Results/calibration/polyFit.py       # refit the Poly constants from the logs
python Results/calibration/polyFigures.py   # redraw the calibration figures
```

## Code conventions
- Variables and attributes: `variableName`. Classes: `PascalCase`. Module constants: `UPPER_CASE`. Files: `camelCase.py`.
- Functions: `functionNameDoesAThing(_paramA, _paramB)`, verb first (`calc`, `solve`, `get`, `load`, `write`, `plot`, `print`, `move`, `apply`).
  - Every parameter starts with `_`, including keyword options, e.g. `plotStepReturn(203, _isCartesian=False)`.
  - A leading `_` on a function name marks an internal helper.
  - Acronyms stay in capitals: `solveFullIK`, `calcArcFK`, `getPCCParams`.
  - Every function has at least a one-line docstring, and the `def` line is always a single line.
- Space names: `Ego` means motor space (rad), `Glob` means Cartesian space (mm, rad), and `Config` means configuration space [k1, r1, s1, k2, r2, s2].
- Imports are absolute from the repo root (`from Core.PCC.constantCurvature import ConstantCurvature`). Scripts in subfolders start with one line that puts the repo root on the path, so they can be run directly.
- Settings and constants live in the JSON files. Use the loaders in `Core/settings.py`; don't copy the values into code.
