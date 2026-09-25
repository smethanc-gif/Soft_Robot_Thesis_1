'''
CONTINUUM ROBOT MAIN SCRIPT
'''
## IMPORTS ##
import time
import math
from math import sin as S
from math import cos as C
from math import pi as PI
import numpy as np
from STM32_comms.pi_comm import STM32Comm
from PCC_control.CTRL import ContinuumRobot
from results import fileWriting as W
from camera_feedback.robot_tracker import TipTracker, SMOOTHING
from controller.ctrlInput import Controller as INPUT
import results.Graphing as G

## COMMS ##
PORT    = "/dev/ttyS0"
BAUD    = 115200

## DATALOGGING ##
DESIRED_EGO_FILE        = "DesiredMotorPosition.txt"
CURRENT_EGO_FILE        = "CurrentMotorPosition.txt"
DESIRED_GLOB_FILE       = "DesiredTipPosition.txt"
CALCULATED_GLOB_FILE    = "CalculatedTipPosition.txt"
CURRENT_GLOB_FILE       = "CurrentTipPosition.txt"


fileNumber = 55 # Filler so graphing doesn't fail

## CTRL PROPERTIES ##
moveMode    = "PCC"                                     # "PCC", "Poly", "PD_test"  
                                                        # DISCONNECT TENDONS FOR PD_test
trajectory  = 9                                         # See MOVEMENT_FUNCTIONS()
Robot     = ContinuumRobot(method=moveMode) if moveMode in ("PCC", "Poly") else None
PD_test_mode = "sine"           # ramp or sine

## MOVEMENT TIMINGS ##
setupTime   = 2 # Time expected to move into initial position (logged, t = -2 -> 0)
setupHoldTime = 0.2 # Held at the start of setupTime (motors at zero) and at its end (at the t = 0 motor positions)

timeStep = 0.04
freq = 1/timeStep

packupTime  = 2 # Time expected to move back to starting position
packupHoldTime = 2 # Time held at zero after the pack-up ramp, before disabling (logged)

## STEP-RETURN TESTS (trajectory 9 = C-shape, 10 = S-shape)
# Bend out to each (direction, bend) pair, come back to straight and hold, to
# measure how far the robot fails to return (hysteresis / tendon slack), and to
# give the Poly fit a spread of shapes. Each section takes half the bend:
# C-shape - both sections bend the same way (one arc, tool tilts with the bend)
# S-shape - section 2 bends back by the same amount, tool stays vertical (as in traj 4, 7, 8)
SR_DIRS     = [i*PI/4 for i in range(8)]            # bend directions (rad), same 8 as Tests 138-159
SR_BENDS    = [PI/18, PI/6, PI/3, PI/2]             # total bend |theta1|+|theta2|: 10, 30, 60, 90 deg
SR_RAMP     = 2                                     # s, ramp out and ramp back each
SR_HOLD     = 2                                     # s, hold bent
SR_REST     = 3                                     # s, hold straight - residual measured here
SR_CYCLE    = 2*SR_RAMP + SR_HOLD + SR_REST
SR_BEND_L   = 2*148.5                               # mm, both sections as one arc (sectL in constantCurve.py)
SR_TOOL_L   = 30                                    # mm, toolLength in constantCurve.py

## COMMAND FUNCTIONS SETUP
if trajectory == 3:
    ctrl = INPUT()
    taskTime = 1000
elif trajectory == 5:
    taskTime = 100
elif trajectory in (9, 10):
    taskTime = SR_CYCLE*len(SR_DIRS)*len(SR_BENDS)
else:
    taskTime    = 30
    movementfreq = 6 # Amount of Repetitions of movement in the taskTime
    const_1 = PI*2*movementfreq/taskTime

if trajectory == 1:
    const_1 *= 2

def MOVEMENT_FUNCTIONS(t_now: float, _trajectory: int):
    ''' Returns the Cartesian position of the tip point for a chosen trajectory at a given time.\n
    0 - Horizontal Circle\n
    1 - Vertical Spiral\n
    2 - Vertical Rotate\n
    3 - Controller Input\n
    4 - Full Circle\n
    5 - Butterfly\n
    7 - Figure 8\n
    8 - 3D Helix\n
    9 - Step-return, C-shape (hysteresis test)\n
    10 - Step-return, S-shape'''
    match _trajectory:
        case 0:                                         ## Horizontal Circle
            x   , y   , z   = 150, 30*S(const_1*t_now), 220 + 30*C(const_1*t_now)
            Rx  , Ry  , Rz  = 0,  PI/2, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0
        case 1:                                         ## Vertical Spiral
            x   , y   , z   = 90* S(const_1*t_now) *S(const_1*t_now/8), 90*C(const_1*t_now)*S(const_1*t_now/8), 300
            Rx  , Ry  , Rz  = 0, 0, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0
            
        case 2:                                         ## Vertical Rotate
            x, y, z = 0, 0, 275 + 20*C(const_1*t_now)
            Rx, Ry, Rz = PI* C(const_1*t_now) *S(const_1*t_now/8)/8, PI*S(const_1*t_now)*S(const_1*t_now/8)/8, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 3:                                         ## Controller Input?
            ctrl.update()
            x, y, z, Rx, Ry, Rz = ctrl.process()
            dx= dy= dz = 0
            dRx= dRy= dRz =0

        case 4: # Full Circle - centred, untilted; complements case 0's off-axis tilted loop
            x, y, z = 120*C(const_1*t_now), 120*S(const_1*t_now), 270
            Rx, Ry, Rz = 0, 0, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 5: # Butterfly curve (Temple Fay). Own timing, not const_1 - this
                # curve needs a long, fixed sweep of theta to trace its full
                # shape (the loops only resolve over ~24pi), unlike a circle
                # which looks identical after any number of revolutions, so
                # tying it to movementfreq wouldn't make sense here.
            theta = t_now * (24*PI/taskTime)
            rad = math.exp(C(theta)) - 2*C(4*theta) + S(theta/12)**5
            x, y, z = 24*rad*C(theta), 24*rad*S(theta), 270
            Rx, Ry, Rz = 0, 0, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 6: # linear between two points DO NOT USE
            p1 = np.array([0,0,320,0,0,0])
            p2 = np.array([100,100,100, PI/2, PI/2,0])

            x, y, z, Rx, Ry, Rz = p1 + (t_now/taskTime)*p2
            dx,dy,dz,dRx,dRy,dRz = [0]*6

        case 7: # Figure 8
            a = 100
            x = a*C(const_1*t_now)/(1+(S(const_1*t_now))**2)
            y = a*S(const_1*t_now)*C(const_1*t_now)/(1+(S(const_1*t_now)**2))
            z = 300
            Rx, Ry, Rz = [0]*3
            dx,dy,dz,dRx,dRy,dRz = [0]*6

        case 8: # 3D Helix
            radius = 90
            x = radius *C(const_1*t_now)
            y = radius *S(const_1*t_now)
            z = 300 - 50*t_now/taskTime
            Rx, Ry, Rz = [0]*3
            dx,dy,dz,dRx,dRy,dRz = [0]*6

        case 9 | 10: # Step-return, C-shape (9) or S-shape (10) - see stepReturnSchedule()
            phi, bend, _, _ = stepReturnSchedule(t_now)
            x, y, z, Rx, Ry, Rz = stepReturnPose(phi, bend, "C" if _trajectory == 9 else "S")
            dx,dy,dz,dRx,dRy,dRz = [0]*6
        case _:
            x   = y     = z     = 0
            Rx  = Ry    = Rz    = 0
            dx  = dy    = dz    = 0
            dRx = dRy   = dRz   = 0
            
    positions = [x,y, z, Rx, Ry, Rz]
    velocities = [dx, dy, dz, dRx, dRy, dRz]
    return positions, velocities


def stepReturnSchedule(t_now: float) -> tuple[float, float, str, int]:
    ''' Step-return test schedule at time t_now.\n
        Returns (direction, current bend angle, phase, cycle index), where
        phase is "out", "hold", "back", "rest" or "done".'''
    n = len(SR_DIRS)*len(SR_BENDS)
    i = int(max(t_now, 0) // SR_CYCLE)
    if i >= n:
        return 0.0, 0.0, "done", n
    phi = SR_DIRS[i // len(SR_BENDS)]
    bend = SR_BENDS[i % len(SR_BENDS)]
    tc = max(t_now, 0) - i*SR_CYCLE
    if tc < SR_RAMP:
        return phi, bend*(1 - C(PI*tc/SR_RAMP))/2, "out", i
    tc -= SR_RAMP
    if tc < SR_HOLD:
        return phi, bend, "hold", i
    tc -= SR_HOLD
    if tc < SR_RAMP:
        return phi, bend*(1 + C(PI*tc/SR_RAMP))/2, "back", i
    return phi, 0.0, "rest", i

def stepReturnPose(phi: float, bend: float, shape: str = "C") -> list[float]:
    ''' Tool pose for a total bend of `bend` rad in direction `phi`, each
        section taking half of it.\n
        "C" - one constant-curvature arc over the whole bending length.\n
        "S" - section 2 bends back by the same amount as section 1: the shape is
        point-symmetric about the joint, so the tip sits at twice the joint's
        position and the tool comes back to vertical.'''
    if shape == "S":
        if bend < 1e-9:
            return [0.0, 0.0, SR_BEND_L + SR_TOOL_L, 0.0, 0.0, 0.0]
        R = SR_BEND_L/bend                                    # radius of each section's arc
        r_xy = 2*R*(1 - C(bend/2))
        return [r_xy*C(phi), r_xy*S(phi), 2*R*S(bend/2) + SR_TOOL_L, 0.0, 0.0, 0.0]

    t = [S(bend)*C(phi), S(bend)*S(phi), C(bend)]            # tool axis
    if bend < 1e-9:
        p = [0.0, 0.0, SR_BEND_L]
    else:
        R = SR_BEND_L/bend
        p = [R*(1 - C(bend))*C(phi), R*(1 - C(bend))*S(phi), R*S(bend)]
    Rx = math.asin(t[1])                                      # inverse of _tangentFromAngles
    Ry = math.atan2(t[0], t[2])
    return [p[0] + SR_TOOL_L*t[0], p[1] + SR_TOOL_L*t[1], p[2] + SR_TOOL_L*t[2], Rx, Ry, 0]

def PD_TEST(t_now: float, tMode)-> tuple[list[float], list[float]]:
    match tMode:
        case "ramp":
            rampTime = taskTime/8
            peak = 5 # radians
            if (t_now <= rampTime):
                _targEgoPos, _targEgoVel = [t_now*peak/rampTime]*6, [peak/rampTime] *6
            elif (t_now >= (taskTime-rampTime)):
                _targEgoPos, _targEgoVel = [peak - (t_now - taskTime + rampTime)*peak/rampTime]*6, [-peak/rampTime] *6
            else:
                _targEgoPos, _targEgoVel = [peak]*6, [0]*6
            return _targEgoPos, _targEgoVel
        case "sine":
            A = 3
            f = 2 * taskTime /PI
            _targEgoPos, _targEgoVel = [A * S(t_now * f)]*6, [A*(f)*C(t_now * f)]*6
            return _targEgoPos, _targEgoVel

def CTRL_INIT(mode, _trajectory: int, stm: STM32Comm, tracker: TipTracker):
    '''Move Motors to positions that correlate to a given trajectory at when t = 0:
       hold straight (motors at zero) for setupHoldTime, ramp linearly to the motor
       positions for t = 0, then hold there for setupHoldTime before the trajectory
       starts. All logged, t = -2 -> 0.
       Returns 0 on fail, 1 on success'''
    

    
    if mode in ("PCC", "Poly"):

        targGlobPos, targGlobVel = MOVEMENT_FUNCTIONS(0, _trajectory)
        Robot.setTarget(targGlobPos, targGlobVel)
        targEgoPos, _ = Robot.FullIK(targGlobPos, targGlobVel)
        targEgoVel =[x /timeStep for x in targEgoPos]

    elif mode == "PD_test":
        targEgoPos, targEgoVel = PD_TEST(0, PD_test_mode)
    else:
        return False
    prevGlobPos = [0]*6
    nSetup = round(setupTime/timeStep)
    nHold = round(setupHoldTime/timeStep)
    nRamp = nSetup - 2*nHold
    t_glob = time.monotonic()

    for i in range(nSetup):
        t_init_start = time.monotonic() # Start RealTime Process

        if i < nHold:               # Hold straight (motors at zero)
            initEgoPos, initEgoVel = [0.0]*6, [0.0]*6
        elif i < nHold + nRamp:     # Ramp linearly to the motor positions for t = 0, reaching them on the last ramp step
            initEgoPos = [x * (i - nHold + 1)/nRamp for x in targEgoPos]
            initEgoVel = targEgoVel
        else:                       # Hold at the motor positions for t = 0 until the trajectory starts
            initEgoPos, initEgoVel = targEgoPos, [0.0]*6

        # Data Collection for Logging
        currEgoPos = stm.get_state()
        calcGlobPos = Robot.forwardKinematics() if Robot is not None else [0.0]*6

        cameraData = tracker.update()
        if cameraData["position"] is None:
            print("Missed Camera Frame.\n")
            currGlobPos = prevGlobPos
        else:
            currGlobPos = list(cameraData["position"]) + [0]*3
            prevGlobPos = currGlobPos
        t_now = (t_init_start - t_glob) - 2
        W.writeAll(initEgoPos, currEgoPos, targGlobPos,  calcGlobPos, currGlobPos, t_now)
        if mode == "PCC":
            W.writeCyl(Robot.targCylPos, t_now)
            W.writeLimits(Robot.limitFlags, t_now)
        stm.set_targets(initEgoPos, initEgoVel)

        t_init_end = time.monotonic()
        timeTaken = t_init_end - t_init_start
        if timeTaken <= timeStep:
            time.sleep(timeStep-timeTaken)
        #else:
            #print(f"Loop {i} of the Initialisation Process went overtime by {timeTaken-timeStep}s!")
        
    return t_now, True


# Main
def main():
    
    try:
        ##  INITIALISATION  ##
        W.controlType = moveMode
        motors_enabled = False
        taskFinished = False    # pack-up is only logged after a task that ran to its end
        print("Waiting for camera pose lock...\n")
        stm     = STM32Comm(port=PORT, baud=BAUD, timeout=0.2)
        
        tracker = TipTracker()
        
        
        currEgoPos = [0.0]*6
        currGlobPos = [0.0] * 6 
        taskInitTime = time.monotonic()
        overflowTracker = 0
        while not tracker.getPoseLocked():    # Camera Lock Check
            tracker.update()
            #print("Here\n")
            if (time.monotonic() - taskInitTime) > 10:
                print("Lost Reference\n")
                return 0
            
        if not motors_enabled:  # Enable Motors
            stm.enable()
            motors_enabled = True
            stm.set_targets([0]*6, [0]*6)
            print("Motors Enabled\n")
        
        W.writeMeta({"moveMode": moveMode, "trajectory": trajectory, "taskTime": taskTime,
                     "movementfreq": globals().get("movementfreq", "n/a"),
                     "timeStep": timeStep, "smoothing": SMOOTHING,
                     "setupTime": setupTime, "setupHoldTime": setupHoldTime,
                     "packupTime": packupTime, "packupHoldTime": packupHoldTime})
        t_offset, startStatus = CTRL_INIT(moveMode, trajectory, stm, tracker)
        if not startStatus:
            print("Something is wrong!\n")
            return None
        t_start = time.monotonic()
        print(f"Time to start task took {t_start - taskInitTime}")
        
        ##  CONTROL LOOP START  ##
        while time.monotonic() <= (t_start + taskTime): # Task Completion code
            # TIMING
            t_loop_start = time.monotonic()
            t_now = t_loop_start - t_start

            ## FEEDBACK
            prevEgoPos = currEgoPos
            currEgoPos = stm.get_state()
            if currEgoPos is None:
                print("Failed to Contact STM32\n")
                currEgoPos  = prevEgoPos
            
            prevGlobPos = currGlobPos
            cameraData = tracker.update()   

            if cameraData["position"] is None:
                print("Missed Camera Frame.\n")
                currGlobPos = prevGlobPos
                W._path(DESIRED_EGO_FILE)
            else:
                currGlobPos = list(cameraData["position"]) + [0]*3
            
            ## CONTROL
            targGlobPos, targGlobVel = MOVEMENT_FUNCTIONS(t_now, trajectory)
            
            if moveMode in ("PCC", "Poly"):
                calcGlobPos = Robot.forwardKinematics()
                targEgoPos, targEgoVel  = Robot.FullIK(targGlobPos, targGlobVel)

            elif moveMode == "PD_test":
                targEgoPos, targEgoVel = PD_TEST(t_now, PD_test_mode)
                targGlobPos, calcGlobPos, currGlobPos = [0]*6, [0]*6, [0]*6

            else:
                print("No Control Mode Given.\n")
                break
            stm.set_targets(targEgoPos, targEgoVel)
            
            ## DOCUMENT
            fileNumber = W.writeAll(targEgoPos, currEgoPos, targGlobPos,  calcGlobPos, currGlobPos, t_now +t_offset)
            if moveMode == "PCC":
                W.writeCyl(Robot.targCylPos, t_now +t_offset)
                W.writeLimits(Robot.limitFlags, t_now +t_offset)

            if trajectory == 3:
                if ctrl.y_held == 1:
                    break

            ## TIMING 2 
            t_loop = time.monotonic() - t_loop_start
            if t_loop <= timeStep:
                time.sleep(timeStep-t_loop)
            else:
                #print(f"Went overtime at t_now = {t_now}s!")
                overflowTracker += 1
        taskFinished = True

    finally:
        print("Disabling motors and cleaning up.")
        print(f"Over the task, {overflowTracker} loops went overTime.")
        if currEgoPos is not None and motors_enabled:
            endPoint = currEgoPos
            for t in range(round(packupTime*freq)):
                stm.set_targets([pos * (1 - t/(freq * packupTime)) for pos in endPoint], [0] * 6)
                time.sleep(timeStep)
            for t in range(round(2*freq)):
                stm.set_targets([0]*6, [0] * 6)
                time.sleep(timeStep)
        tracker.stop()
        stm.disable()
        stm.close()   
        #G.EgoGraph(test=fileNumber)
        #G.GlobGraph3D(test=fileNumber)
        #G.EgoGraph(test = fileNumber, vel=True)
        #G.GlobGraph(test=fileNumber)


if __name__ == "__main__":
    main()
