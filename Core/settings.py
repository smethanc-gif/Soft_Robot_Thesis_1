'''Loads the JSON settings files shared by the whole codebase:
    Core/controlSettings.json  - control method, trajectory and timings for main.py
    Core/robotGeometry.json    - the robot's physical dimensions and limits
    Results/dataLogging.json   - where test logs are written and what each file is called
Each file has a "notes" block explaining its values and units.
'''
import os
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_SETTINGS_FILE = os.path.join(REPO_ROOT, "Core", "controlSettings.json")
ROBOT_GEOMETRY_FILE = os.path.join(REPO_ROOT, "Core", "robotGeometry.json")
DATA_LOGGING_FILE = os.path.join(REPO_ROOT, "Results", "dataLogging.json")


def loadJson(_path: str) -> dict:
    '''Reads one JSON file into a dict.'''
    with open(_path) as f:
        return json.load(f)


def loadControlSettings() -> dict:
    '''Control method, trajectory and timings (Core/controlSettings.json).'''
    return loadJson(CONTROL_SETTINGS_FILE)


def loadRobotGeometry() -> dict:
    '''Physical dimensions and limits of the robot (Core/robotGeometry.json).'''
    return loadJson(ROBOT_GEOMETRY_FILE)


def loadDataLogging() -> dict:
    '''Log folder and file names (Results/dataLogging.json), with logRoot made absolute.'''
    logging = loadJson(DATA_LOGGING_FILE)
    logging["logRoot"] = os.path.join(REPO_ROOT, *logging["logRoot"].split("/"))
    return logging
