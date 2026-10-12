'''"PCC_Cable" method: PCC with each cable's travel corrected by its fitted gain.

Same target shape as PCC (same midpoint solve, coupling and limits), but the
motor angles account for the fact that each cable only delivers a fraction of
the length its motor winds (the tendon gains, fitted in
Results/calibration/polyFit.py - see Results/calibration/polyConstants.md):

    motor angle = PCC-geometry angle for the target shape / tendon gain

plus the fitted length gain, so the sections shorten as PCC intends. This
isolates "calibrating the cables" from "changing the curvature model" (Poly,
Core/Poly/polyCurvature.py, builds on this class).
'''
import math
import time
import numpy as np
from Core.PCC.constantCurvature import ConstantCurvature
from Core.Poly import polyModel
from Core.settings import loadRobotGeometry


class CableCompensatedPCC(ConstantCurvature):
    def __init__(self, _constantsFile: str = polyModel.CONSTANTS_FILE):
        '''PCC robot plus the fitted tendon and length gains from _constantsFile.'''
        super().__init__()
        constants = polyModel.loadConstants(_constantsFile)
        self.tendonGains = constants["gains"]         # A..F: fraction of wound cable that reaches the robot
        self.modelParams = constants["pcc"]           # forward model used by calcForwardKinematics
        self.lengthGain = self.modelParams[-2]
        self.motorLimit = loadRobotGeometry()["motorLimit"]   # rad - clamp below the STM32's 20 rad safety trip
        self.clampCount = 0                           # how many commands have hit motorLimit
        self.targEgoPos = np.zeros(6)

    def calcCompensatedEgo(self) -> np.ndarray:
        '''Motor angles that make the robot take PCC's target shape (self.targConfigPos), given the
            fitted tendon and length gains.'''
        k1, r1, s1, k2, r2, s2 = self.targConfigPos
        v1 = k1*s1*np.exp(1j*r1)                      # bend of each section (rad, direction as a complex number)
        v2 = k2*s2*np.exp(1j*r2)
        # The model scales length changes by lengthGain, so ask for 1/lengthGain of PCC's
        effective = polyModel.calcMotorsFromSections(s1, v1, s2, v2)[0]
        return effective/self.tendonGains

    def _applyEgoCommand(self, _q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Clamps the motor command to motorLimit, stores it, and adds the finite-difference velocity
            reference (same scheme as ConstantCurvature.calcEgoTargets).'''
        if np.any(np.abs(_q) > self.motorLimit):
            self.clampCount += 1
            _q = np.clip(_q, -self.motorLimit, self.motorLimit)
        now = time.monotonic()
        vel = np.zeros(6)
        if hasattr(self, "_lastCmdTime"):
            dt = now - self._lastCmdTime
            if 0 < dt <= 0.5:                         # only when streaming targets
                vel = (_q - self.targEgoPos)/dt
        self.targEgoPos, self._lastCmdTime = _q, now
        return _q, vel

    def calcEgoTargets(self, _targBendPos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Motor positions and velocities: PCC's target shape, with the motors corrected for the
            tendon and length gains.'''
        ConstantCurvature.calcEgoTargets(self, _targBendPos)     # PCC target shape -> self.targConfigPos (limits applied)
        return self._applyEgoCommand(self.calcCompensatedEgo())

    def calcForwardKinematics(self) -> np.ndarray:
        '''Tool pose [x, y, z, Rx, Ry, 0] the fitted model predicts for the last motor command.'''
        tip, t = polyModel.calcPolyPose(self.modelParams, self.targEgoPos[None, :], self.tendonGains)
        tip, t = tip[0], t[0]
        Rx = math.atan2(t[1], math.hypot(t[0], t[2]))
        Ry = math.atan2(t[0], t[2])
        return np.array([tip[0], tip[1], tip[2], Rx, Ry, 0.0])
