## Goal - Control-method dispatcher: turns a target Cartesian pose into motor targets
'''
Owns which control method is active and the generic control-loop bookkeeping
(last commanded target, tool-offset removal). All model-specific math (FK,
IK, cable-length conversion) is delegated to the active model:
    "PCC"       - constantCurve.py: piecewise constant curvature
    "PCC_Cable" - cableComp.py: PCC with each cable's travel corrected by its fitted gain
    "Poly"      - poly.py: fitted polynomial-curvature model, built on PCC_Cable
All three share PCC's target-shape solve (midpointCalc + PCC_Control).
'''
import numpy as np


from Core.PCC import constantCurvature as constantCurve
from Core.PCC_Cable import cableCompensation as cableComp
from Core.Poly import polyCurvature as poly

METHODS = {"PCC": constantCurve.ConstantCurvature,
           "PCC_Cable": cableComp.CableCompensatedPCC,
           "Poly": poly.PolynomialCurvature}


class ContinuumRobot:
    def __init__(self, method: str = "PCC"):
        self.method = method
        if method not in METHODS:
            raise ValueError(f"Unknown control method: {method!r} (choose from {list(METHODS)})")
        self.model = METHODS[method]()

    def __getattr__(self, name):
        '''Fall through to the active model for anything not defined here
            (e.g. sectL, toolLength, targCylPos, arcFK, midpointCalc).'''
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)

    def setTarget(self, _targGlobPos: list[float], _targGlobVel: list[float]) -> bool:
        if len(_targGlobPos) != 6 or len(_targGlobVel) != 6:
            print(f"Error:\n {len(_targGlobPos)} positions provided.\n {len(_targGlobVel)} velocities provided.\n")
            return False
        self.targGlobPos = np.array(_targGlobPos)
        self.targGlobVel = np.array(_targGlobVel)
        #print(_targGlobPos,"\n")
        return True

    def toolTransform(self, _toolGlobPos: np.ndarray) -> np.ndarray:
        '''Calc the endpoint for the bending sections'''
        x, y, z, Rx, Ry, Rz = _toolGlobPos
        t = self._tangentFromAngles(Rx, Ry)
        L = self.toolLength
        return np.array([x - L*t[0], y - L*t[1], z - L*t[2], Rx, Ry, Rz])

    def FullIK(self, _targGlobPos: list[float], _targGlobVel: list[float]) -> tuple[np.ndarray, np.ndarray]:
        ''' Runs toolTransform, setTarget, then the active method's own IK
            and cable/motor conversion.\n
            returns the motor positions necessary to get to the desired Cartesian Coordinates'''
        targBendPos = self.toolTransform(_targGlobPos)
        self.setTarget(targBendPos, _targGlobVel)
        self.model.midpointCalc(self.targGlobPos)
        return self.model.PCC_Control(self.targGlobPos)

    def forwardKinematics(self) -> np.ndarray:
        ''' Tool pose the active model predicts for its last command. '''
        return self.model.forwardKinematics()
