'''Control-method dispatcher: turns a target Cartesian tool pose into motor targets.

Owns which control method is active and the generic control-loop bookkeeping
(last commanded target, tool-offset removal). All model-specific maths (FK,
IK, cable-length conversion) is delegated to the active model:
    "PCC"       - PCC/constantCurvature.py: piecewise constant curvature
    "PCC_Cable" - PCC_Cable/cableCompensation.py: PCC with each cable's travel corrected by its fitted gain
    "Poly"      - Poly/polyCurvature.py: fitted polynomial-curvature model, built on PCC_Cable
All three share PCC's target-shape solve (calcMidpoint + calcEgoTargets).
'''
import numpy as np
from Core.PCC.constantCurvature import ConstantCurvature
from Core.PCC_Cable.cableCompensation import CableCompensatedPCC
from Core.Poly.polyCurvature import PolynomialCurvature

METHODS = {"PCC": ConstantCurvature,
           "PCC_Cable": CableCompensatedPCC,
           "Poly": PolynomialCurvature}


class ContinuumRobot:
    def __init__(self, _method: str = "PCC"):
        '''Robot controlled by one of the METHODS ("PCC", "PCC_Cable" or "Poly").'''
        self.method = _method
        if _method not in METHODS:
            raise ValueError(f"Unknown control method: {_method!r} (choose from {list(METHODS)})")
        self.model = METHODS[_method]()

    def __getattr__(self, _name):
        '''Falls through to the active model for anything not defined here
            (e.g. sectL, toolLength, targConfigPos, calcArcFK, calcMidpoint).'''
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(_name) from None
        return getattr(model, _name)

    def setTarget(self, _targGlobPos: list[float], _targGlobVel: list[float]) -> bool:
        '''Stores the target pose and velocity [x, y, z, Rx, Ry, Rz]. Returns False if either isn't 6 long.'''
        if len(_targGlobPos) != 6 or len(_targGlobVel) != 6:
            print(f"Error:\n {len(_targGlobPos)} positions provided.\n {len(_targGlobVel)} velocities provided.\n")
            return False
        self.targGlobPos = np.array(_targGlobPos)
        self.targGlobVel = np.array(_targGlobVel)
        return True

    def calcBendGlobPos(self, _toolGlobPos: np.ndarray) -> np.ndarray:
        '''Pose of the end of the bending sections for a tool-tip pose (removes the straight tool).'''
        x, y, z, Rx, Ry, Rz = _toolGlobPos
        t = self._calcTangentFromAngles(Rx, Ry)
        L = self.toolLength
        return np.array([x - L*t[0], y - L*t[1], z - L*t[2], Rx, Ry, Rz])

    def solveFullIK(self, _targGlobPos: list[float], _targGlobVel: list[float]) -> tuple[np.ndarray, np.ndarray]:
        '''Motor positions and velocities that put the tool tip at _targGlobPos: removes the tool
            (calcBendGlobPos), stores the target, then runs the active method's own IK and cable/motor conversion.'''
        targBendPos = self.calcBendGlobPos(_targGlobPos)
        self.setTarget(targBendPos, _targGlobVel)
        self.model.calcMidpoint(self.targGlobPos)
        return self.model.calcEgoTargets(self.targGlobPos)

    def calcForwardKinematics(self) -> np.ndarray:
        '''Tool pose the active model predicts for its last command.'''
        return self.model.calcForwardKinematics()
