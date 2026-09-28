## Goal - "PCC_Cable" method: PCC with each cable's travel corrected by its fitted gain
'''
Same target shape as PCC (same midpoint solve, coupling and limits), but the
motor angles account for the fact that each cable only delivers a fraction of
the length its motor winds (the tendon gains, fitted in results/polyFit.py -
see results/POLY_CONSTANTS.md):

    motor angle = PCC-geometry angle for the target shape / tendon gain

plus the fitted length gain, so the sections shorten as PCC intends. This
isolates "calibrating the cables" from "changing the curvature model" (Poly,
poly.py, builds on this class).
'''
import math
import time
import numpy as np

from Core.PCC.constantCurvature import ConstantCurvature
from Core.Poly import polyModel
from Core.settings import loadRobotGeometry


class CableCompensatedPCC(ConstantCurvature):
    def __init__(self, constantsFile: str = polyModel.CONSTANTS_FILE):
        super().__init__()
        C = polyModel.loadConstants(constantsFile)
        self.tendonGains = C["gains"]                 # A..F: fraction of wound cable that reaches the robot
        self.modelParams = C["pcc"]                   # forward model used by forwardKinematics
        self.lengthGain = self.modelParams[-2]
        self.motorLimit = loadRobotGeometry()["motorLimit"]   # rad - clamp below the STM32's 20 rad safety trip
        self.clampCount = 0                           # how many commands have hit motorLimit
        self.targMotorPos = np.zeros(6)

    def compensatedMotors(self) -> np.ndarray:
        ''' Motor angles that make the robot take PCC's target shape
            (self.targCylPos), given the fitted tendon and length gains. '''
        k1, r1, s1, k2, r2, s2 = self.targCylPos
        v1 = k1*s1*np.exp(1j*r1)                      # bend of each section (rad, direction as a complex number)
        v2 = k2*s2*np.exp(1j*r2)
        # The model scales length changes by lengthGain, so ask for 1/lengthGain of PCC's
        L = self.sectL
        s1e, s2e = L + (s1 - L)/self.lengthGain, L + (s2 - L)/self.lengthGain
        effective = polyModel.motorsFromSections(s1e, v1, s2e, v2)[0]
        return effective/self.tendonGains

    def _command(self, q: np.ndarray):
        ''' Clamp, store, and add the finite-difference velocity reference
            (same scheme as ConstantCurvature.PCC_Control). '''
        if np.any(np.abs(q) > self.motorLimit):
            self.clampCount += 1
            q = np.clip(q, -self.motorLimit, self.motorLimit)
        now = time.monotonic()
        vel = np.zeros(6)
        if hasattr(self, "_lastCmdTime"):
            dt = now - self._lastCmdTime
            if 0 < dt <= 0.5:                         # only when streaming targets
                vel = (q - self.targMotorPos)/dt
        self.targMotorPos, self._lastCmdTime = q, now
        return q, vel

    def PCC_Control(self, targGlobPos: np.ndarray):
        ConstantCurvature.PCC_Control(self, targGlobPos)     # PCC target shape -> self.targCylPos (limits applied)
        return self._command(self.compensatedMotors())

    def forwardKinematics(self) -> np.ndarray:
        ''' Tool pose the fitted model predicts for the last motor command. '''
        tip, t = polyModel.polyPose(self.modelParams, self.targMotorPos[None, :], self.tendonGains)
        tip, t = tip[0], t[0]
        Rx = math.atan2(t[1], math.hypot(t[0], t[2]))
        Ry = math.atan2(t[0], t[2])
        return np.array([tip[0], tip[1], tip[2], Rx, Ry, 0.0])
