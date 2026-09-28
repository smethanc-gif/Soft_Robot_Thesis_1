'''Target trajectories for main.py (moved out of main.py unchanged).

calcTargGlob gives the Cartesian tool-tip pose at a given time for each
trajectory number; the trajectory, taskTime and movementFreq come from
Core/controlSettings.json. Also holds the step-return test schedule
(trajectories 9 and 10) and the PD_test motor profiles.
'''
import math
from math import sin, cos, pi
from Core.settings import loadControlSettings, loadRobotGeometry

CONTROL = loadControlSettings()
GEOMETRY = loadRobotGeometry()
trajectory = CONTROL["trajectory"]

# ---- Step-return tests (trajectory 9 = C-shape, 10 = S-shape) ----
# Bend out to each (direction, bend) pair, come back to straight and hold, to
# measure how far the robot fails to return (hysteresis / tendon slack), and to
# give the Poly fit a spread of shapes. Each section takes half the bend:
# C-shape - both sections bend the same way (one arc, tool tilts with the bend)
# S-shape - section 2 bends back by the same amount, tool stays vertical (as in traj 4, 7, 8)
SR_DIRS     = [i*pi/4 for i in range(8)]                # bend directions (rad), same 8 as Tests 138-159
SR_BENDS    = [pi/18, pi/6, pi/3, pi/2, 3*pi/4]         # total bend |theta1|+|theta2|: 10, 30, 60, 90, 135 deg
SR_RAMP     = 2                                         # s, ramp out and ramp back each
SR_HOLD     = 2                                         # s, hold bent
SR_REST     = 3                                         # s, hold straight - residual measured here
SR_CYCLE    = 2*SR_RAMP + SR_HOLD + SR_REST
SR_BEND_L   = 2*GEOMETRY["sectL"]                       # mm, both sections as one arc
SR_TOOL_L   = GEOMETRY["toolLength"]                    # mm

# ---- Task timing for the chosen trajectory ----
movementFreq = "n/a"                                    # only used by the periodic trajectories
if trajectory == 3:
    from controller.ctrlInput import Controller        # pygame - only needed for controller input
    gamepad = Controller()
    taskTime = 1000
elif trajectory == 5:
    taskTime = 100
elif trajectory in (9, 10):
    taskTime = SR_CYCLE*len(SR_DIRS)*len(SR_BENDS)
else:
    taskTime = CONTROL["taskTime"]
    movementFreq = CONTROL["movementFreq"]              # repetitions of the movement in taskTime
    angularSpeed = pi*2*movementFreq/taskTime

if trajectory == 1:
    angularSpeed *= 2


def calcTargGlob(_tNow: float, _trajectory: int) -> tuple[list[float], list[float]]:
    '''Returns the Cartesian position and velocity of the tool tip for a chosen trajectory at a given time.\n
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
            x   , y   , z   = 150, 30*sin(angularSpeed*_tNow), 220 + 30*cos(angularSpeed*_tNow)
            Rx  , Ry  , Rz  = 0,  pi/2, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0
        case 1:                                         ## Vertical Spiral
            x   , y   , z   = 90* sin(angularSpeed*_tNow) *sin(angularSpeed*_tNow/8), 90*cos(angularSpeed*_tNow)*sin(angularSpeed*_tNow/8), 300
            Rx  , Ry  , Rz  = 0, 0, 0
            dx  , dy  , dz  = 0, 0, 0
            dRx , dRy , dRz = 0, 0, 0

        case 2:                                         ## Vertical Rotate
            x, y, z = 0, 0, 275 + 20*cos(angularSpeed*_tNow)
            Rx, Ry, Rz = pi* cos(angularSpeed*_tNow) *sin(angularSpeed*_tNow/8)/8, pi*sin(angularSpeed*_tNow)*sin(angularSpeed*_tNow/8)/8, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 3:                                         ## Controller Input
            gamepad.update()
            x, y, z, Rx, Ry, Rz = gamepad.calcTargGlob()
            dx= dy= dz = 0
            dRx= dRy= dRz =0

        case 4: # Full Circle - centred, untilted; complements case 0's off-axis tilted loop
            x, y, z = 100*cos(angularSpeed*_tNow), 100*sin(angularSpeed*_tNow), 280
            Rx, Ry, Rz = 0, 0, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 5: # Butterfly curve (Temple Fay). Own timing, not angularSpeed - this
                # curve needs a long, fixed sweep of theta to trace its full
                # shape (the loops only resolve over ~24pi), unlike a circle
                # which looks identical after any number of revolutions, so
                # tying it to movementFreq wouldn't make sense here.
            theta = _tNow * (24*pi/taskTime)
            rad = math.exp(cos(theta)) - 2*cos(4*theta) + sin(theta/12)**5
            x, y, z = 24*rad*cos(theta), 24*rad*sin(theta), 250
            Rx, Ry, Rz = 0, 0, 0
            dx, dy, dz = 0, 0, 0
            dRx, dRy, dRz = 0, 0, 0

        case 7: # Figure 8
            a = 100
            x = a*cos(angularSpeed*_tNow)/(1+(sin(angularSpeed*_tNow))**2)
            y = a*sin(angularSpeed*_tNow)*cos(angularSpeed*_tNow)/(1+(sin(angularSpeed*_tNow)**2))
            z = 260
            Rx, Ry, Rz = [0]*3
            dx,dy,dz,dRx,dRy,dRz = [0]*6

        case 8: # 3D Helix
            radius = 90
            x = radius *cos(angularSpeed*_tNow)
            y = radius *sin(angularSpeed*_tNow)
            z = 300 - 80*_tNow/taskTime
            Rx, Ry, Rz = [0]*3
            dx,dy,dz,dRx,dRy,dRz = [0]*6

        case 9 | 10: # Step-return, C-shape (9) or S-shape (10) - see getStepReturnPhase()
            phi, bend, _, _ = getStepReturnPhase(_tNow)
            x, y, z, Rx, Ry, Rz = calcStepReturnPos(phi, bend, "C" if _trajectory == 9 else "S")
            dx,dy,dz,dRx,dRy,dRz = [0]*6
        case _:
            x   = y     = z     = 0
            Rx  = Ry    = Rz    = 0
            dx  = dy    = dz    = 0
            dRx = dRy   = dRz   = 0

    positions = [x,y, z, Rx, Ry, Rz]
    velocities = [dx, dy, dz, dRx, dRy, dRz]
    return positions, velocities


def getStepReturnPhase(_tNow: float) -> tuple[float, float, str, int]:
    '''Step-return test schedule at time _tNow.\n
        Returns (direction, current bend angle, phase, cycle index), where
        phase is "out", "hold", "back", "rest" or "done".'''
    n = len(SR_DIRS)*len(SR_BENDS)
    i = int(max(_tNow, 0) // SR_CYCLE)
    if i >= n:
        return 0.0, 0.0, "done", n
    phi = SR_DIRS[i // len(SR_BENDS)]
    bend = SR_BENDS[i % len(SR_BENDS)]
    tc = max(_tNow, 0) - i*SR_CYCLE
    if tc < SR_RAMP:
        return phi, bend*(1 - cos(pi*tc/SR_RAMP))/2, "out", i
    tc -= SR_RAMP
    if tc < SR_HOLD:
        return phi, bend, "hold", i
    tc -= SR_HOLD
    if tc < SR_RAMP:
        return phi, bend*(1 + cos(pi*tc/SR_RAMP))/2, "back", i
    return phi, 0.0, "rest", i


def calcStepReturnPos(_phi: float, _bend: float, _shape: str = "C") -> list[float]:
    '''Tool pose for a total bend of _bend rad in direction _phi, each
        section taking half of it.\n
        "C" - one constant-curvature arc over the whole bending length.\n
        "S" - section 2 bends back by the same amount as section 1: the shape is
        point-symmetric about the joint, so the tip sits at twice the joint's
        position and the tool comes back to vertical.'''
    if _shape == "S":
        if _bend < 1e-9:
            return [0.0, 0.0, SR_BEND_L + SR_TOOL_L, 0.0, 0.0, 0.0]
        R = SR_BEND_L/_bend                                   # radius of each section's arc
        rXY = 2*R*(1 - cos(_bend/2))
        return [rXY*cos(_phi), rXY*sin(_phi), 2*R*sin(_bend/2) + SR_TOOL_L, 0.0, 0.0, 0.0]

    t = [sin(_bend)*cos(_phi), sin(_bend)*sin(_phi), cos(_bend)]    # tool axis
    if _bend < 1e-9:
        p = [0.0, 0.0, SR_BEND_L]
    else:
        R = SR_BEND_L/_bend
        p = [R*(1 - cos(_bend))*cos(_phi), R*(1 - cos(_bend))*sin(_phi), R*sin(_bend)]
    Rx = math.asin(t[1])                                      # inverse of _calcTangentFromAngles
    Ry = math.atan2(t[0], t[2])
    return [p[0] + SR_TOOL_L*t[0], p[1] + SR_TOOL_L*t[1], p[2] + SR_TOOL_L*t[2], Rx, Ry, 0]


def calcPDTestEgo(_tNow: float, _mode: str) -> tuple[list[float], list[float]]:
    '''Motor position and velocity targets for the PD_test mode (all six motors
        the same, tendons disconnected): "ramp" up to 5 rad, hold, ramp down, or
        a 3 rad "sine".'''
    match _mode:
        case "ramp":
            rampTime = taskTime/8
            peak = 5 # radians
            if (_tNow <= rampTime):
                targEgoPos, targEgoVel = [_tNow*peak/rampTime]*6, [peak/rampTime] *6
            elif (_tNow >= (taskTime-rampTime)):
                targEgoPos, targEgoVel = [peak - (_tNow - taskTime + rampTime)*peak/rampTime]*6, [-peak/rampTime] *6
            else:
                targEgoPos, targEgoVel = [peak]*6, [0]*6
            return targEgoPos, targEgoVel
        case "sine":
            A = 3
            f = 2 * taskTime /pi
            targEgoPos, targEgoVel = [A * sin(_tNow * f)]*6, [A*(f)*cos(_tNow * f)]*6
            return targEgoPos, targEgoVel
