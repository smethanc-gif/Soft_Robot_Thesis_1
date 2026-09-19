## Goal - Control-method dispatcher: turns a target Cartesian pose into motor targets
'''
Owns which curvature model is active (e.g. "PCC" or, later, "Poly") and the
generic control-loop bookkeeping (last commanded target, tool-offset removal).
All curvature-specific math (FK, IK, cable-length conversion) is delegated to
the active model - see constantCurve.py for the "PCC" model.
'''
import numpy as np

try:
    from . import constantCurve
except ImportError:
    import constantCurve


class ContinuumRobot:
    def __init__(self, method: str = "PCC"):
        self.method = method
        if method == "PCC":
            self.model = constantCurve.ConstantCurvature()
        elif method == "Poly":
            raise NotImplementedError("Poly control method is not implemented yet")
        else:
            raise ValueError(f"Unknown control method: {method!r}")

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
        ''' Runs toolTransform, setTarget, midpointCalc, and the active model's
            cable/motor conversion.\n
            returns the motor positions necessary to get to the desired Cartesian Coordinates'''
        targBendPos = self.toolTransform(_targGlobPos)
        self.setTarget(targBendPos, _targGlobVel)
        self.model.midpointCalc(self.targGlobPos)
        return self.model.PCC_Control(self.targGlobPos)

    def forwardKinematics(self) -> np.ndarray:
        return self.model.forwardKinematics()
