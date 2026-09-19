'''
CONTINUUM ROBOT MAIN SCRIPT
'''
## IMPORTS ##
import time
from math import sin as S
from math import cos as C
from math import pi as PI
from STM32_comms.pi_comm import STM32Comm
from PCC_control.PCC_control import ContinuumRobot as control1
from results import fileWriting as W
from camera_feedback.robot_tracker import TipTracker
from controller.ctrlInput import Controller
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
trajectory  = 0                                         # See MOVEMENT_FUNCTIONS()
PCC       = control1()                                  
                                         
                                                         
PD_test_mode = "sine"           # ramp or sine

## MOVEMENT TIMINGS ##
setupTime   = 2 # Time expected to move into initial position

timeStep = 0.04
freq = 1/timeStep

packupTime  = 1 # Time expected to move back to starting position

## COMMAND FUNCTIONS SETUP
if trajectory == 3:
    ctrl = Controller()
    taskTime = 1000
else:
    taskTime    = 10
    movementfreq = 2 # Amount of Repetitions of movement in the taskTime
    const_1 = PI*2*movementfreq/taskTime

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
            x   , y   , z   = 200, 20*S(const_1*t_now), 180 + 20*C(const_1*t_now) 
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

def CTRL_INIT(mode, _trajectory: int, stm: STM32Comm, tracker: TipTracker): 
    '''Move Motors to positions that correlate to a given trajectory at when t = 0
       Returns 0 on fail, 1 on success'''
    

    
    if mode == "PCC":
        
        targGlobPos, targGlobVel = MOVEMENT_FUNCTIONS(0, _trajectory)
        PCC.setTarget(targGlobPos, targGlobVel)
        targEgoPos, _ = PCC.FullIK(targGlobPos, targGlobVel)
        targEgoVel =[x /timeStep for x in targEgoPos]
    
    elif mode == "Poly":
        return False
    
    elif mode == "PD_test":
        targEgoPos, targEgoVel = PD_TEST(0, PD_test_mode)
    else:
        return False
    prevGlobPos = [0]*6
    t_glob = time.monotonic()
    
    for i in range(round(setupTime/timeStep)):
        t_init_start = time.monotonic() # Start RealTime Process

        if i < 0.9*round(setupTime/timeStep): # Linearly reach starting position within 90% of the starting time
            initEgoPos = [ x * (i/(0.9*freq*setupTime)) for x in targEgoPos]
            initEgoVel =[x /setupTime for x in targEgoPos]
        else:
            initEgoPos = targEgoPos
            initEgoVel = [0]*6

        # Data Collection for Logging
        currEgoPos = stm.get_state()
        calcGlobPos = PCC.forwardKinematics()
        if tracker.update()["position"] is None:
            print("Missed Camera Frame.\n")
            currGlobPos = prevGlobPos
        else:
            currGlobPos = list(tracker.update()["position"]) + [0]*3
        prevGlobPos = currGlobPos

        t_now = (t_init_start - t_glob) - 2
        W.writeAll(initEgoPos, currEgoPos, targGlobPos,  calcGlobPos, currGlobPos, t_now)
        stm.set_targets(initEgoPos, initEgoVel)

        t_init_end = time.monotonic()
        timeTaken = t_init_end - t_init_start
        if timeTaken <= timeStep:
            time.sleep(timeStep-timeTaken)
        else:
            print(f"Loop {i} of the Initialisation Process went overtime by {timeTaken-timeStep}s!")
        
    return t_now, True


# Main
def main():
    
    try:
        ##  INITIALISATION  ##
        W.controlType = moveMode
        motors_enabled = False
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
            fileNumber = W.writeAll(targEgoPos, currEgoPos, targGlobPos,  calcGlobPos, currGlobPos, t_now +t_offset)

            if trajectory == 3:
                if ctrl.y_held == 1:
                    break

            ## TIMING 2 
            t_loop = time.monotonic() - t_loop_start
            if t_loop <= timeStep:
                time.sleep(timeStep-t_loop)
            else:
                print(f"Went overtime at t_now = {t_now}s!")
                overflowTracker += 1

    finally:
        print("Disabling motors and cleaning up.")
        print(f"Over the task, {overflowTracker} loops went overTime.")
        if currEgoPos is not None and motors_enabled:
            endPoint = currEgoPos
            for t in range(round(packupTime*freq)):
                stm.set_targets([pos * (1 - t/(freq * packupTime)) for pos in endPoint], [0] * 6)
                time.sleep(1/freq)
        tracker.stop()
        stm.disable()
        stm.close()   
        G.EgoGraph(test=fileNumber)
        #G.EgoGraph(test = fileNumber, vel=True)
        #G.GlobGraph(test=fileNumber)


if __name__ == "__main__":
    main()
