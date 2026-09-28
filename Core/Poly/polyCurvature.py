## Goal - "Poly" method: polynomial-curvature model fitted to the robot, inverted for control
'''
Forward model: polyModel.polyPose with the fitted constants - tendon gains,
then linearly varying curvature along each section whose size depends on
bend size and compression (see results/POLY_CONSTANTS.md).

Inverse kinematics, each control step:
  1. Start from the cable-compensated PCC solution q0 (cableComp.py) for the
     same target - PCC's shape, corrected for the tendon gains.
  2. Find the smallest motor correction d (so q = q0 + d) that puts the Poly
     model's predicted tool tip on the target tip position: damped
     Gauss-Newton, Jacobian recomputed each iteration. d is always measured
     from q0, so parts of it that don't move the tip can't build up over
     time.
  3. Warm start from the previous step's correction; the best iterate is
     kept even if the tolerance isn't reached, so the next step starts
     closer.
  4. Safety: the correction is limited to maxCorrection per motor and may
     change by at most maxCorrectionStep per control step, so the motor
     commands never jump. limitCount counts steps where either limit acted.
Only the tip position is targeted: the model was fitted to camera tip
positions, so its tool-orientation predictions aren't validated.
'''
import time
import numpy as np

from Core.PCC_Cable.cableCompensation import CableCompensatedPCC
from Core.PCC.constantCurvature import ConstantCurvature
from Core.Poly import polyModel


class PolynomialCurvature(CableCompensatedPCC):
    def __init__(self, constantsFile: str = polyModel.CONSTANTS_FILE):
        super().__init__(constantsFile)
        self.modelParams = polyModel.loadConstants(constantsFile)["poly"]
        self.maxIterations = 4
        self.tolerance = 0.1          # mm - stop once the predicted tip is this close to the target
        self.maxCorrection = 3.0      # rad per motor away from the cable-compensated solution
        self.maxCorrectionStep = 0.25 # rad per motor per control step (6 rad/s at 25 Hz)
        self.limitCount = 0           # steps where the correction was limited
        self.fallbackCount = 0        # steps where the solve failed outright (non-finite)
        self.lastTipError = 0.0       # mm, predicted tip vs target at the best iterate
        self.lastIterations = 0
        self.lastSolveMs = 0.0        # time spent in solveMotors on the last step
        self._correction = np.zeros(6)
        self._lastSolveTime = None

    def _tipsAndJacobian(self, q: np.ndarray, h: float = 1e-4):
        ''' Predicted tip at q and its 3x6 Jacobian (one batched model call). '''
        Q = np.vstack([q, q + h*np.eye(6)])
        tips = polyModel.polyTip(self.modelParams, Q, self.tendonGains)
        return tips[0], (tips[1:] - tips[0]).T/h

    def solveMotors(self, tipTarget: np.ndarray, q0: np.ndarray) -> np.ndarray:
        ''' Motor angles whose predicted tip is tipTarget, as close as possible
            to q0 (the cable-compensated solution). '''
        t0 = time.perf_counter()
        now = time.monotonic()
        streaming = self._lastSolveTime is not None and 0 < now - self._lastSolveTime <= 0.5
        self._lastSolveTime = now
        dPrev = self._correction if streaming else np.zeros(6)

        d, best = dPrev.copy(), None
        for it in range(self.maxIterations + 1):
            tip, J = self._tipsAndJacobian(q0 + d)
            err = tipTarget - tip
            e = float(np.linalg.norm(err))
            if best is None or e < best[0]:
                best = (e, d.copy(), it)
            if e < self.tolerance or it == self.maxIterations:
                break
            JJt = J @ J.T
            # smallest d with J d = err + J d_current (linearised tip equation, measured from q0)
            d = J.T @ np.linalg.solve(JJt + 1e-3*np.trace(JJt)/3*np.eye(3), err + J @ d)
        self.lastTipError, d, self.lastIterations = best

        if not np.all(np.isfinite(d)):
            self.fallbackCount += 1
            d = dPrev
        limited = d.copy()
        hit = False
        if streaming and np.any(np.abs(d - dPrev) > self.maxCorrectionStep):
            limited = dPrev + np.clip(d - dPrev, -self.maxCorrectionStep, self.maxCorrectionStep)
            hit = True
        if np.any(np.abs(limited) > self.maxCorrection):
            limited = np.clip(limited, -self.maxCorrection, self.maxCorrection)
            hit = True
        self.limitCount += hit
        self._correction = limited
        self.lastSolveMs = (time.perf_counter() - t0)*1000
        return q0 + limited

    def PCC_Control(self, targGlobPos: np.ndarray):
        ''' targGlobPos is the bend-end target from ContinuumRobot.toolTransform;
            the tool tip target is recovered by adding the tool back on. '''
        ConstantCurvature.PCC_Control(self, targGlobPos)     # PCC target shape -> self.targCylPos
        q0 = self.compensatedMotors()                         # cable-compensated PCC solution
        x, y, z, Rx, Ry, _ = targGlobPos
        tipTarget = np.array([x, y, z]) + self.toolLength*self._tangentFromAngles(Rx, Ry)
        return self._command(self.solveMotors(tipTarget, q0))
