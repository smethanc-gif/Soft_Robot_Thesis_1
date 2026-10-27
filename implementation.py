'''CONTINUUM ROBOT EXHIBITION SCRIPT

Stripped-down version of main.py for demos: no camera, no logging, no analysis.
Two modes, picked by cycleTrajectories in Core/controlSettings.json:
    []          - single run: move into the start position, follow "trajectory", ramp back to zero.
    [4, 7, 8]   - cycle: run each trajectory in turn, returning to zero and resting between
                  each, looping until stopped (Ctrl+C). taskTime and movementFreq apply to each one.
Stopping at any point ramps the motors back to zero and disables them.
'''
import time
from STM32_comms.piComm import STM32Comm
from Core.continuumRobot import ContinuumRobot
from Core.settings import loadControlSettings
from Core import trajectories
from Core.trajectories import calcTargGlob, calcPDTestEgo, setTrajectoryTiming

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

setupTime       = CONTROL["setupTime"]                  # time to move into the initial position
setupHoldTime   = CONTROL["setupHoldTime"]              # held at the start of setupTime (motors at zero) and at its end (at the t = 0 motor positions)
timeStep        = CONTROL["timeStep"]
freq            = 1/timeStep
packupTime      = CONTROL["packupTime"]                 # time to ramp back to zero after the task
packupHoldTime  = CONTROL["packupHoldTime"]             # time held at zero after the pack-up ramp
cycleTrajectories = CONTROL["cycleTrajectories"]        # trajectories looped through in cycle mode, [] for a single run
cycleRestTime   = CONTROL["cycleRestTime"]              # time resting at zero between cycled trajectories
# taskTime is set per trajectory in Core/trajectories.py - read it as trajectories.taskTime so it follows setTrajectoryTiming


def waitForStepEnd(_tStepStart: float) -> bool:
    '''Sleeps out the rest of the control step. Returns False if the step already went overtime.'''
    tStep = time.monotonic() - _tStepStart
    if tStep <= timeStep:
        time.sleep(timeStep - tStep)
        return True
    return False


def moveToStartPos(_stm: STM32Comm, _state: dict, _trajectory: int) -> bool:
    '''Moves the motors to the positions for _trajectory at t = 0: hold straight for
       setupHoldTime, ramp linearly to the t = 0 positions, then hold there for setupHoldTime.
       Returns False for an unknown moveMode.'''
    if moveMode in MODEL_MODES:
        targGlobPos, targGlobVel = calcTargGlob(0, _trajectory)
        robot.setTarget(targGlobPos, targGlobVel)
        targEgoPos, _ = robot.solveFullIK(targGlobPos, targGlobVel)
        targEgoVel = [x/timeStep for x in targEgoPos]
    elif moveMode == "PD_test":
        targEgoPos, targEgoVel = calcPDTestEgo(0, pdTestMode)
    else:
        return False
    nSetup = round(setupTime/timeStep)
    nHold = round(setupHoldTime/timeStep)
    nRamp = nSetup - 2*nHold

    for i in range(nSetup):
        tStepStart = time.monotonic()

        if i < nHold:               # Hold straight (motors at zero)
            initEgoPos, initEgoVel = [0.0]*6, [0.0]*6
        elif i < nHold + nRamp:     # Ramp linearly to the motor positions for t = 0, reaching them on the last ramp step
            initEgoPos = [x * (i - nHold + 1)/nRamp for x in targEgoPos]
            initEgoVel = targEgoVel
        else:                       # Hold at the motor positions for t = 0 until the trajectory starts
            initEgoPos, initEgoVel = targEgoPos, [0.0]*6

        _stm.setTargets(initEgoPos, initEgoVel)
        _state["currEgoPos"] = initEgoPos
        waitForStepEnd(tStepStart)
    return True


def runControlLoop(_stm: STM32Comm, _state: dict, _trajectory: int) -> bool:
    '''Follows _trajectory for taskTime, one control step per timeStep. Keeps the latest
       motor positions in _state for returnToZero. Returns True if it ran the full taskTime,
       False if it stopped early (no control mode, or Y held in controller mode).'''
    tStart = time.monotonic()
    while time.monotonic() <= (tStart + trajectories.taskTime):
        tStepStart = time.monotonic()
        tNow = tStepStart - tStart

        # Feedback
        currEgoPos = _stm.getState()
        if currEgoPos is None:
            print("Failed to Contact STM32\n")
        else:
            _state["currEgoPos"] = currEgoPos

        # Control
        targGlobPos, targGlobVel = calcTargGlob(tNow, _trajectory)
        if moveMode in MODEL_MODES:
            targEgoPos, targEgoVel = robot.solveFullIK(targGlobPos, targGlobVel)
        elif moveMode == "PD_test":
            targEgoPos, targEgoVel = calcPDTestEgo(tNow, pdTestMode)
        else:
            print("No Control Mode Given.\n")
            return False
        _stm.setTargets(targEgoPos, targEgoVel)

        if _trajectory == 3 and trajectories.gamepad.yHeld == 1:
            return False

        waitForStepEnd(tStepStart)
    return True


def returnToZero(_stm: STM32Comm, _state: dict):
    '''Ramps the motors from their last position back to zero over packupTime, then holds zero for packupHoldTime.
       Keeps _state up to date on every step, so an interrupted ramp carries on from where it got to.'''
    endPoint = _state["currEgoPos"]
    for t in range(round(packupTime*freq)):
        rampEgoPos = [pos * (1 - t/(freq * packupTime)) for pos in endPoint]
        _stm.setTargets(rampEgoPos, [0] * 6)
        _state["currEgoPos"] = rampEgoPos
        time.sleep(timeStep)
    _state["currEgoPos"] = [0.0]*6
    for t in range(round(packupHoldTime*freq)):
        _stm.setTargets([0]*6, [0] * 6)
        time.sleep(timeStep)


def runSingle(_stm: STM32Comm, _state: dict):
    '''Runs the settings trajectory once. main() ramps back to zero afterwards.'''
    if not moveToStartPos(_stm, _state, trajectory):
        print("Something is wrong!\n")
        return
    runControlLoop(_stm, _state, trajectory)


def runCycle(_stm: STM32Comm, _state: dict):
    '''Runs each of cycleTrajectories in turn, returning to zero and resting between each,
       until Ctrl+C. Only stops by itself if a trajectory can't be run (bad moveMode).'''
    cycle = 1
    while True:
        for traj in cycleTrajectories:
            setTrajectoryTiming(traj)
            print(f"Cycle {cycle}: trajectory {traj} ({trajectories.taskTime} s)")
            if not moveToStartPos(_stm, _state, traj):
                print("Something is wrong!\n")
                return
            if not runControlLoop(_stm, _state, traj):
                return
            returnToZero(_stm, _state)
            time.sleep(cycleRestTime)
        cycle += 1


def main():
    '''Enables the motors and runs either cycle or single mode until it finishes or Ctrl+C,
       then always ramps back to zero and disables the motors.'''
    if 3 in cycleTrajectories:
        print("Trajectory 3 (controller input) can't be cycled - remove it from cycleTrajectories.\n")
        return
    stm = None
    state = {"currEgoPos": [0.0]*6, "motorsEnabled": False}
    try:
        stm = STM32Comm(_port=PORT, _baud=BAUD, _timeout=0.2)
        stm.enable()
        state["motorsEnabled"] = True
        stm.setTargets([0]*6, [0]*6)
        print("Motors Enabled. Ctrl+C to stop.\n")

        if cycleTrajectories:
            runCycle(stm, state)
        else:
            runSingle(stm, state)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        print("Disabling motors and cleaning up.")
        if stm is not None:
            if state["motorsEnabled"]:
                returnToZero(stm, state)
            stm.disable()
            stm.close()


if __name__ == "__main__":
    main()
