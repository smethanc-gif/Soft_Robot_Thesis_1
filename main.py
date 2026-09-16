'''
CONTINUUM ROBOT TESTING SCRIPT
'''

import time

from math import sin as S
from math import cos as C
from math import pi as PI
from STM32_comms.pi_comm import STM32Comm
from PCC_control.PCC_control import ContinuumRobot as control1
from results import fileWriting as W
from camera_feedback.robot_tracker import TipTracker
from PCC_control.ctrlInput import Controller
import results.Graphing as G
# Config and Comms
PORT    = "/dev/ttyS0"
BAUD    = 115200

DESIRED_EGO_FILE        = "DesiredMotorPosition.txt"
CURRENT_EGO_FILE        = "CurrentMotorPosition.txt"
DESIRED_GLOB_FILE       = "DesiredTipPosition.txt"
CALCULATED_GLOB_FILE    = "CalculatedTipPosition.txt"
CURRENT_GLOB_FILE       = "CurrentTipPosition.txt"

Frequency = 30 # Hz
Period = 1/Frequency


# Robot Properties
moveMode    = "PCC"                                     # "PCC"     = PCC Control
trajectory  = 1                                         # "Rod"     = Cosserat Rod Control
PCC       = control1()                                  # "Poly"    = Polynomial Curvature Case
                                                        # "Order"   = Fractional Order Control
                                                        # "PD_test" = Test PD Stuff 
PD_test_mode = "sine"           # ramp or sine

# Movement Timing
setupTime   = 1

packupTime  = 1
if trajectory == 3:
    ctrl = Controller()
    taskTime = 1000
else:
    taskTime    = 10
    movementFrequency = 2 # Amount of Repetitions of movement in the taskTime
    const_1 = PI*2*movementFrequency/taskTime

if trajectory == 1:
    const_1 *= 2
def MOVEMENT_FUNCTIONS(t_now: float, _trajectory: int):
    ''' Returns the Cartesian position of the tip point for a chosen trajectory at a given time.\n
    0 - Horizontal Circle\n
    1 - Vertical Spiral\n
    2 - Vertical Rotate\n
    3 - Controller Input'''
    match _trajectory:
        case 0:                                         ## Horizontal Circle
            x   , y   , z   = 200, 20*S(const_1*t_now), 180 + 0*C(const_1*t_now) 
            Rx  , Ry  , Rz  = 0,  PI/2, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0
        case 1:                                         ## Vertical Spiral
            x   , y   , z   = 50* S(const_1*t_now) *S(const_1*t_now/8), 50*C(const_1*t_now)*S(const_1*t_now/8), 300
            Rx  , Ry  , Rz  = 0, 0, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0
            
        case 2:                                         ## Vertical Rotate
            x, y, z = 0, 0, 250 + 40*C(const_1*t_now) 
            Rx, Ry, Rz = PI* C(const_1*t_now) *S(const_1*t_now/8)/4, PI*S(const_1*t_now)*S(const_1*t_now/8)/4, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 3:                                         ## Controller Input?
            ctrl.update()
            x, y, z, Rx, Ry, Rz = ctrl.output()
            dx= dy= dz = 0
            dRx= dRy= dRz =0

        case _: 
            x   = y     = z     = 0
            Rx  = Ry    = Rz    = 0
            dx  = dy    = dz    = 0
            dRx = dRy   = dRz   = 0
            
    positions = [x,y, z, Rx, Ry, Rz]
    velocities = [dx, dy, dz, dRx, dRy, dRz]
    return positions, velocities

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

def CTRL_INIT(mode, _trajectory: int, stm: STM32Comm)-> bool: 
    '''Move Motors to positions that correlate to a given trajectory at when t = 0
       Returns 0 on fail, 1 on success'''
    

    
    if mode == "PCC":
        
        targGlobPos, targGlobVel = MOVEMENT_FUNCTIONS(0, _trajectory)
        PCC.setTarget(targGlobPos, targGlobVel)
        targEgoPos, targEgoVel = PCC.FullIK(targGlobPos, targGlobVel)
    elif mode == "Rod":
        return False
        
    elif mode == "Poly":
        return False
    elif mode == "Order":
        return False
    elif mode == "PD_test":
        targEgoPos, targEgoVel = PD_TEST(0, PD_test_mode)
    else:
        return False
    for i in range(round(setupTime*Frequency)):
        initEgoPos = [ x * (i/(Frequency*setupTime)) for x in targEgoPos]
        stm.set_targets(initEgoPos, targEgoVel)
        time.sleep(1/Frequency)
    return True


# Main
def main():
    
    try:
        ##  INITIALISATION
        W.controlType = moveMode
        motors_enabled = False
        print("Waiting for camera pose lock...\n")
        stm     = STM32Comm(port=PORT, baud=BAUD, timeout=0.2)
        
        tracker = TipTracker()
        
        
        currEgoPos = [0.0]*6
        currGlobPos = [0.0] * 6 
        trackInitTime = time.monotonic()

        while not tracker.getPoseLocked():    # Camera Lock Check
            tracker.update()
            print("Here\n")
            if (time.monotonic() - trackInitTime) > 10:
                print("Lost Reference\n")
                return 0
            
        if not motors_enabled:  # Enable Motors
            stm.enable()
            motors_enabled = True
            stm.set_targets([0]*6, [0]*6)
            print("Motors Enabled\n")
        
        startStatus = CTRL_INIT(moveMode, trajectory, stm)
        if not startStatus:
            print("Something is wrong!\n")
            return None
        t_start = time.monotonic()
        next_wake = t_start + Period # For a real-time approximation

        while time.monotonic() <= (t_start + taskTime): # Task Completion code
            t_now = time.monotonic() - t_start
            ## FEEDBACK
            prevEgoPos = currEgoPos
            currEgoPos = stm.get_state()
            if currEgoPos is None:
                print("Failed to Contact STM32\n")
                currEgoPos  = prevEgoPos
            
            
            prevGlobPos = currGlobPos
            cameraData = tracker.update()   # Camera Update is doing time

            if cameraData["position"] is None:
                print("Missed Camera Frame.\n")
                currGlobPos = prevGlobPos
                W._path(DESIRED_EGO_FILE)
            else:
                currGlobPos = list(cameraData["position"]) + [0]*3
            
            ## CONTROL
            targGlobPos, targGlobVel = MOVEMENT_FUNCTIONS(t_now, trajectory)
            
            if moveMode == "PCC":
                calcGlobPos = PCC.forwardKinematics()
                targEgoPos, targEgoVel  = PCC.FullIK(targGlobPos, targGlobVel)
            elif moveMode == "PD_test":
                targEgoPos, targEgoVel = PD_TEST(t_now, PD_test_mode)
                targGlobPos, calcGlobPos, currGlobPos = [0]*6, [0]*6, [0]*6
            else:
                print("No Control Mode Given.\n")
                break
            stm.set_targets(targEgoPos, targEgoVel)
            
            ## DOCUMENT
            fileNumber = W.writeAll(targEgoPos, currEgoPos, targGlobPos,  calcGlobPos, currGlobPos, t_now)

            if trajectory == 3:
                if ctrl.y_held == 1:
                    break
            #sleep_time = next_wake - time.monotonic()
            #if sleep_time > 0:
            #    time.sleep(sleep_time)
            #    
            #else:
            #    print("Went Overtime!")
            #    next_wake = time.monotonic()
            #next_wake += Period
            time.sleep(0.01) ## TODO Fix Real time stuff
            

    finally:
        print("Disabling motors and cleaning up.")
        if currEgoPos is not None and motors_enabled:
            endPoint = currEgoPos
            for t in range(round(packupTime*Frequency)):
                stm.set_targets([pos * (1 - t/(Frequency * packupTime)) for pos in endPoint], [0] * 6)
                time.sleep(1/Frequency)
        tracker.stop()
        stm.disable()
        stm.close()   
        G.EgoGraph(test=fileNumber)
        #G.GlobGraph(test=fileNumber)


if __name__ == "__main__":
    main()
