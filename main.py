'''CONTINUUM ROBOT MAIN SCRIPT

Runs one test on the robot: waits for the camera, moves into the start
position, follows the trajectory with the chosen control method, then ramps
back to zero. What to run (moveMode, trajectory, timings) is set in
Core/controlSettings.json; every step is logged to Results/data/<moveMode>/Test_N.
'''
import time
from STM32_comms.piComm import STM32Comm
from Core.continuumRobot import ContinuumRobot
from Core.settings import loadControlSettings
from Core import trajectories
from Core.trajectories import calcTargGlob, calcPDTestEgo
from Results.dataLogger import DataLogger
from camera_feedback.robotTracker import TipTracker, SMOOTHING
from Results.analysis import analyse as Ana

# ---- Comms ----
PORT    = "/dev/ttyS0"
BAUD    = 115200

# ---- Settings (Core/controlSettings.json) ----
CONTROL = loadControlSettings()
MODEL_MODES = ("PCC", "PCC_Cable", "Poly")              # modes that run a kinematic model (see Core/continuumRobot.py)
moveMode    = CONTROL["moveMode"]                       # "PCC", "PCC_Cable", "Poly", "PD_test" - DISCONNECT TENDONS FOR PD_test
trajectory  = CONTROL["trajectory"]                     # see Core/trajectories.py calcTargGlob
pdTestMode  = CONTROL["pdTestMode"]                     # ramp or sine
robot       = ContinuumRobot(moveMode) if moveMode in MODEL_MODES else None

setupTime       = CONTROL["setupTime"]                  # time to move into the initial position (logged, t = -2 -> 0)
setupHoldTime   = CONTROL["setupHoldTime"]              # held at the start of setupTime (motors at zero) and at its end (at the t = 0 motor positions)
timeStep        = CONTROL["timeStep"]
freq            = 1/timeStep
taskTime        = trajectories.taskTime                 # set per trajectory in Core/trajectories.py
packupTime      = CONTROL["packupTime"]                 # time to ramp back to zero after the task
packupHoldTime  = CONTROL["packupHoldTime"]             # time held at zero after the pack-up ramp, before disabling
testNumber = 1

def readCamera(_tracker: TipTracker, _prevGlobPos: list[float]) -> list[float]:
    '''Latest camera tip position as [x, y, z, 0, 0, 0], or the previous one if the frame was missed.'''
    cameraData = _tracker.update()
    if cameraData["position"] is None:
        print("Missed Camera Frame.\n")
        return _prevGlobPos
    return list(cameraData["position"]) + [0]*3


def waitForStepEnd(_tStepStart: float) -> bool:
    '''Sleeps out the rest of the control step. Returns False if the step already went overtime.'''
    tStep = time.monotonic() - _tStepStart
    if tStep <= timeStep:
        time.sleep(timeStep - tStep)
        return True
    return False


def moveToStartPos(_stm: STM32Comm, _tracker: TipTracker, _logger: DataLogger) -> tuple[float, bool]:
    '''Moves the motors to the positions for the trajectory at t = 0: hold straight (motors
       at zero) for setupHoldTime, ramp linearly to the t = 0 positions, then hold there for
       setupHoldTime before the trajectory starts. All logged, t = -2 -> 0.
       Returns (last logged time, True) on success, (0.0, False) for an unknown moveMode.'''
    if moveMode in MODEL_MODES:
        targGlobPos, targGlobVel = calcTargGlob(0, trajectory)
        robot.setTarget(targGlobPos, targGlobVel)
        targEgoPos, _ = robot.solveFullIK(targGlobPos, targGlobVel)
        targEgoVel = [x/timeStep for x in targEgoPos]
    elif moveMode == "PD_test":
        targEgoPos, targEgoVel = calcPDTestEgo(0, pdTestMode)
        targGlobPos = [0]*6
    else:
        return 0.0, False
    prevGlobPos = [0]*6
    nSetup = round(setupTime/timeStep)
    nHold = round(setupHoldTime/timeStep)
    nRamp = nSetup - 2*nHold
    tGlob = time.monotonic()

    for i in range(nSetup):
        tStepStart = time.monotonic()

        if i < nHold:               # Hold straight (motors at zero)
            initEgoPos, initEgoVel = [0.0]*6, [0.0]*6
        elif i < nHold + nRamp:     # Ramp linearly to the motor positions for t = 0, reaching them on the last ramp step
            initEgoPos = [x * (i - nHold + 1)/nRamp for x in targEgoPos]
            initEgoVel = targEgoVel
        else:                       # Hold at the motor positions for t = 0 until the trajectory starts
            initEgoPos, initEgoVel = targEgoPos, [0.0]*6

        # Data collection for logging
        currEgoPos = _stm.getState()
        calcGlobPos = robot.calcForwardKinematics() if robot is not None else [0.0]*6
        currGlobPos = readCamera(_tracker, prevGlobPos)
        prevGlobPos = currGlobPos
        tNow = (tStepStart - tGlob) - 2
        global testNumber 
        testNumber = _logger.writeAll(initEgoPos, currEgoPos, targGlobPos, calcGlobPos, currGlobPos, tNow)
        if moveMode in MODEL_MODES:
            _logger.writeConfig(robot.targConfigPos, tNow)
            _logger.writeLimits(robot.limitFlags, tNow)
        _stm.setTargets(initEgoPos, initEgoVel)

        waitForStepEnd(tStepStart)
    return tNow, True


def runControlLoop(_stm: STM32Comm, _tracker: TipTracker, _logger: DataLogger, _tOffset: float, _state: dict):
    '''Follows the trajectory for taskTime, one control step per timeStep: read the motors and
       camera, solve the next motor targets, send them and log everything. Keeps the latest
       motor positions and overtime count in _state for packUp.'''
    currGlobPos = [0.0]*6
    tStart = time.monotonic()
    print(f"Time to start task took {tStart - _state['tInit']}")
    while time.monotonic() <= (tStart + taskTime):
        tStepStart = time.monotonic()
        tNow = tStepStart - tStart

        # Feedback
        prevEgoPos = _state["currEgoPos"]
        currEgoPos = _stm.getState()
        if currEgoPos is None:
            print("Failed to Contact STM32\n")
            currEgoPos = prevEgoPos
        _state["currEgoPos"] = currEgoPos
        currGlobPos = readCamera(_tracker, currGlobPos)

        # Control
        targGlobPos, targGlobVel = calcTargGlob(tNow, trajectory)
        if moveMode in MODEL_MODES:
            calcGlobPos = robot.calcForwardKinematics()
            targEgoPos, targEgoVel = robot.solveFullIK(targGlobPos, targGlobVel)
        elif moveMode == "PD_test":
            targEgoPos, targEgoVel = calcPDTestEgo(tNow, pdTestMode)
            targGlobPos, calcGlobPos, currGlobPos = [0]*6, [0]*6, [0]*6
        else:
            print("No Control Mode Given.\n")
            break
        _stm.setTargets(targEgoPos, targEgoVel)

        # Document
        _logger.writeAll(targEgoPos, currEgoPos, targGlobPos, calcGlobPos, currGlobPos, tNow + _tOffset)
        if moveMode in MODEL_MODES:
            _logger.writeConfig(robot.targConfigPos, tNow + _tOffset)
            _logger.writeLimits(robot.limitFlags, tNow + _tOffset)

        if trajectory == 3 and trajectories.gamepad.yHeld == 1:
            break

        if not waitForStepEnd(tStepStart):
            _state["overflowCount"] += 1


def packUp(_stm: STM32Comm, _tracker: TipTracker, _state: dict):
    '''Ramps the motors from their last position back to zero over packupTime, holds zero for
       packupHoldTime, then stops the camera and disables and closes the STM32 link.'''
    print("Disabling motors and cleaning up.")
    print(f"Over the task, {_state['overflowCount']} loops went overTime.")
    if _state["currEgoPos"] is not None and _state["motorsEnabled"]:
        endPoint = _state["currEgoPos"]
        for t in range(round(packupTime*freq)):
            _stm.setTargets([pos * (1 - t/(freq * packupTime)) for pos in endPoint], [0] * 6)
            time.sleep(timeStep)
        for t in range(round(packupHoldTime*freq)):
            _stm.setTargets([0]*6, [0] * 6)
            time.sleep(timeStep)
    if _tracker is not None:
        _tracker.stop()
    if _stm is not None:
        _stm.disable()
        _stm.close()


def main():
    '''Runs one test: camera pose lock, enable motors, log the settings, move to the start
       position, run the trajectory, and always pack up at the end (even after an error).'''
    stm = tracker = None
    state = {"currEgoPos": [0.0]*6, "motorsEnabled": False, "overflowCount": 0, "tInit": time.monotonic()}
    try:
        logger = DataLogger(moveMode)
        print("Waiting for camera pose lock...\n")
        stm = STM32Comm(_port=PORT, _baud=BAUD, _timeout=0.2)
        tracker = TipTracker()
        state["tInit"] = time.monotonic()
        while not tracker.getPoseLocked():      # Camera lock check
            tracker.update()
            if (time.monotonic() - state["tInit"]) > 10:
                print("Lost Reference\n")
                return 0

        stm.enable()                            # Enable motors
        state["motorsEnabled"] = True
        stm.setTargets([0]*6, [0]*6)
        print("Motors Enabled\n")

        logger.writeMeta({"moveMode": moveMode, "trajectory": trajectory, "taskTime": taskTime,
                          "movementFreq": trajectories.movementFreq, "timeStep": timeStep, "smoothing": SMOOTHING,
                          "setupTime": setupTime, "setupHoldTime": setupHoldTime,
                          "packupTime": packupTime, "packupHoldTime": packupHoldTime, "pdTestMode": pdTestMode})
        tOffset, startStatus = moveToStartPos(stm, tracker, logger)
        if not startStatus:
            print("Something is wrong!\n")
            return None
        runControlLoop(stm, tracker, logger, tOffset, state)

    finally:
        packUp(stm, tracker, state)


if __name__ == "__main__":
    main()
    print(f"Yoooo! Check this out! I can make Test {testNumber} into a graph.")
    Ana.plotTracking3D(_ctrl=CONTROL["moveMode"],_test=testNumber)