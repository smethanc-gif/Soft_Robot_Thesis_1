## Goal - Polynomial-curvature kinematics + cable-length model for a 2-section continuum robot
'''
Curvature within each section varies along the normalized arc length s' in [0,1],
instead of being constant like PCC. Each section is described by two linear
curvature components (no torsion):
    kx(s') = a0 + a1*s'
    ky(s') = b0 + b1*s'
so the bend MAGNITUDE (sqrt(kx^2+ky^2)) and DIRECTION (atan2(ky,kx)) can both
vary continuously along the section - unlike PCC's fixed (k, r) pair.

There is no closed form for a general (non-constant) curvature section, so FK
and the cable-length relationship are both computed by chopping the section
into N small pieces, treating each as constant-curvature (exact for that one
piece via constantCurve.ConstantCurvature.arcFK), and composing them. With
a1=b1=0 (curvature reduces to constant), this composition is mathematically
EXACT and reproduces PCC's own arcFK/PCC_Control output regardless of N -
that's the main regression check used to validate this file.
'''
import math
from math import pi as PI
import time
import numpy as np

try:
    from . import constantCurve
except ImportError:
    import constantCurve


class PolynomialCurvature:
    def __init__(self):
        '''Same physical robot as ConstantCurvature - see that class for the
            hardware constants. Kept separate (not shared) so this file can be
            read on its own, and because Poly's own math (cable-length in
            particular) uses these constants differently to PCC's closed form.'''
# GEOMETRIC DESCRIPTION
        self.nodes              = 14
        self.sections           = 2
        self.robotRadius        = 15.25 # mm
        self.sectL              = 148.5 #mm
        self.initCableLen       = np.array([self.sectL*2, self.sectL] * 3)
        self.motorShaftDiameter = 10
        self.spoolRadius        = self.motorShaftDiameter/2    # mm of cable per rad of shaft
        self.toolLength         = 30 # mm
# CONTROL OFFSETS
        self.rotationalOffset   = 2.8623 # radians

# PHYSICAL LIMITS
        self.sMin               = 0.8*self.sectL          # section arc length range (mm)
        self.sMax               = self.sectL
        self.thetaMax           = PI/2                # max bend angle along a section
        self.limitsHit          = []                 # which limits clamped on the last solve

# POLY-SPECIFIC CONFIG
        self.N                  = 8      # sub-segments per section for FK/cable-length integration
        self.regWeight          = 0.05   # penalty weight on the curvature-slope terms (a1, b1) - biases
                                          # the IK solve towards the smoothest shape that still hits the target

    @staticmethod
    def _tangentFromAngles(Rx: float, Ry: float) -> np.ndarray:
        ''' Unit vector in the direction the tooltip is pointing, [x,y,z].
            Same pure vector math as ConstantCurvature's copy - duplicated
            rather than cross-referenced so this file reads on its own.'''
        return np.array([math.cos(Rx)*math.sin(Ry), math.sin(Rx), math.cos(Rx)*math.cos(Ry)])

    # ---------------- forward kinematics (per section) ----------------

    def polyFK(self, coeffs: np.ndarray, s: float, N: int = None):
        ''' One section's tip position + rotation, given curvature-vector
            coefficients [a0,a1,b0,b1] and arc length s.\n
            No closed form exists for varying curvature, so this composes N
            small constant-curvature pieces (each exact via arcFK) - this is
            NOT an approximation scheme with its own error on top of the
            piecewise-constant approximation of the real polynomial; each
            piece is solved exactly for its own (locally constant) curvature.'''
        if N is None:
            N = self.N
        a0, a1, b0, b1 = coeffs
        p_acc = np.zeros(3)
        R_acc = np.eye(3)
        ds = s / N
        for i in range(N):
            s_mid = (i + 0.5) / N
            kx = a0 + a1*s_mid
            ky = b0 + b1*s_mid
            k_i = math.hypot(kx, ky)
            r_i = math.atan2(ky, kx) if k_i > 0.0 else 0.0
            p_i, R_i = constantCurve.ConstantCurvature.arcFK(k_i, r_i, ds)
            p_acc = p_acc + R_acc @ p_i
            R_acc = R_acc @ R_i
        return p_acc, R_acc

    def forwardKinematics(self) -> np.ndarray:
        ''' Recompute the tool pose from the last solved shape, used to
            verify the inverse kinematics - mirrors ConstantCurvature's
            version, just composing polyFK instead of arcFK per section.'''
        coeffs1, s1, coeffs2, s2 = self._unpackShape(self.targShapePos)
        p1, R1 = self.polyFK(coeffs1, s1)
        p2, R2 = self.polyFK(coeffs2, s2)
        tip = p1 + R1 @ p2
        t2 = R2[:, 2]
        t = R1 @ t2
        tip = tip + self.toolLength * t
        Rx = math.atan2(t[1], math.hypot(t[0], t[2]))
        Ry = math.atan2(t[0], t[2])
        return np.array([tip[0], tip[1], tip[2], Rx, Ry, 0.0])

    # ---------------- cable length (per section) ----------------

    def cableLength(self, coeffs: np.ndarray, s: float, offsets: np.ndarray, N: int = None) -> np.ndarray:
        ''' Tendon lengths for one section (one offset angle per tendon,
            same convention as PCC_Control's offsets1/offsets2), found the
            same way as polyFK: sum the exact closed-form tendon-length
            contribution of each of N constant-curvature sub-segments.
            The node count (self.nodes) is split proportionally across the N
            pieces so this reduces EXACTLY to PCC's own closed-form tendon
            formula when a1=b1=0, for any N.'''
        if N is None:
            N = self.N
        a0, a1, b0, b1 = coeffs
        OFS = self.rotationalOffset
        d = self.robotRadius
        n_local = self.nodes / N
        ds = s / N
        total = np.zeros(len(offsets))
        for i in range(N):
            s_mid = (i + 0.5) / N
            kx = a0 + a1*s_mid
            ky = b0 + b1*s_mid
            k_i = math.hypot(kx, ky)
            if k_i == 0.0:
                total += ds
                continue
            r_i = math.atan2(ky, kx)
            n_calc = 2*n_local*math.sin(ds*k_i/(2*n_local))
            total += n_calc*(1/k_i - d*np.sin(r_i + OFS + offsets))
        return total

    # ---------------- shape-vector packing ----------------
    # Unknown vector x = [a0_1,a1_1,b0_1,b1_1,s1, a0_2,a1_2,b0_2,b1_2,s2]

    @staticmethod
    def _unpackShape(x: np.ndarray):
        coeffs1 = x[0:4]
        s1 = x[4]
        coeffs2 = x[5:9]
        s2 = x[9]
        return coeffs1, s1, coeffs2, s2

    def _defaultShapeGuess(self) -> np.ndarray:
        ''' Straight robot: zero curvature everywhere, nominal section length. '''
        return np.array([0.0, 0.0, 0.0, 0.0, self.sectL,
                          0.0, 0.0, 0.0, 0.0, self.sectL])

    # ---------------- inverse kinematics ----------------

    def _couplingError(self, s1: float, s2: float) -> float:
        ''' Mechanical coupling from the tendon routing (section 2's tendons
            pass through section 1, so pulling them elongates/shortens both
            sections together): the section-2 length change from nominal
            must be half the section-1 length change. Same relationship as
            ConstantCurvature._couplingError, assumed to carry over to Poly
            until experimental data says otherwise.'''
        return (s2 - self.sectL) - 0.5*(s1 - self.sectL)

    def _shapeError(self, x: np.ndarray, p_target: np.ndarray, t_target: np.ndarray) -> np.ndarray:
        ''' 11-component residual for the LM solve:
              [0:3] tip position error
              [3:6] tip tangent error (full 3-vector, same "don't normalize,
                    let the pole take care of itself" trick PCC's
                    _tangentError uses)
              [6] section-length mechanical coupling (see _couplingError) -
                    a real physical constraint, weighted at full strength
                    (same mm scale as the position error), not a soft
                    preference like the regularization below.
              [7:11] regularization on the 4 curvature-slope terms (a1_1,
                    b1_1, a1_2, b1_2) - pulls the solve towards the smoothest
                    shape (closest to constant curvature) that still reaches
                    the target, instead of wandering the redundant null space.'''
        coeffs1, s1, coeffs2, s2 = self._unpackShape(x)
        p1, R1 = self.polyFK(coeffs1, s1)
        p2, R2 = self.polyFK(coeffs2, s2)
        tip = p1 + R1 @ p2
        tangent = R1 @ R2[:, 2]

        pos_err = tip - p_target
        tan_err = tangent - t_target
        coupling_err = np.array([self._couplingError(s1, s2)])
        reg_err = math.sqrt(self.regWeight) * np.array([coeffs1[1], coeffs1[3], coeffs2[1], coeffs2[3]])
        return np.concatenate([pos_err, tan_err, coupling_err, reg_err])

    def _shapeJacobian(self, x: np.ndarray, p_target: np.ndarray, t_target: np.ndarray, F: np.ndarray) -> np.ndarray:
        ''' Finite-difference Jacobian (11 residuals x 10 unknowns) of
            _shapeError, same one-coordinate-at-a-time pattern as
            ConstantCurvature's _tangentJacobian.'''
        n = len(x)
        J = np.zeros((len(F), n))
        for j in range(n):
            h = 1e-6*max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            Fp = self._shapeError(xp, p_target, t_target)
            J[:, j] = (Fp - F) / h
        return J

    def solveShape(self, targGlobPos: np.ndarray) -> np.ndarray:
        ''' Levenberg-Marquardt solve for both sections' curvature
            coefficients + arc lengths at once (no separate "solve section 1
            then couple section 2" step like PCC needs - composing both
            sections' polyFK directly ties them together in one residual).'''
        x, y, z, Rx, Ry, Rz = targGlobPos
        p_target = np.array([x, y, z])
        t_target = self._tangentFromAngles(Rx, Ry)

        x0 = self.targShapePos.copy() if hasattr(self, "targShapePos") else self._defaultShapeGuess()

        F = self._shapeError(x0, p_target, t_target)
        fn = math.sqrt(np.sum(F**2))
        lam = 1e-3

        for _ in range(80):
            if fn < 1e-8:
                break
            J = self._shapeJacobian(x0, p_target, t_target, F)
            # Marquardt's original SCALED damping (lam*diag(J^T J), not lam*I):
            # the 10 unknowns span wildly different scales - curvature terms
            # are ~1e-3, arc length is ~150 - so uniform damping either barely
            # restrains the curvature terms (unstable, huge first steps) or
            # over-restrains length (no progress). Scaling damping by each
            # parameter's own JtJ diagonal (its local sensitivity) fixes both.
            diagA = np.maximum(np.diag(J.T @ J), 1e-12)
            A = J.T @ J + lam*np.diag(diagA)
            b = -J.T @ F
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                lam *= 10.0
                if lam > 1e6:
                    break
                continue

            n = np.linalg.norm(step)
            if n > 40.0:
                step = step*40.0/n
            cand = x0 + step
            cand[4] = max(cand[4], 1.0)    # arc lengths must stay positive
            cand[9] = max(cand[9], 1.0)
            Fc = self._shapeError(cand, p_target, t_target)
            fc = math.sqrt(np.sum(Fc**2))
            if fc < fn:
                x0, F, fn = cand, Fc, fc
                lam = max(lam*0.2, 1e-9)
            else:
                lam *= 10.0
                if lam > 1e6:
                    break

        self.targShapePos = x0
        return x0

    def _clampBend(self, coeffs: np.ndarray, s: float) -> np.ndarray:
        ''' Scale curvature down (never up) so the peak bend magnitude along
            the section - curvature is linear, so its extreme is at one of
            the section's two ends - stays within thetaMax.'''
        a0, a1, b0, b1 = coeffs
        k_end0 = math.hypot(a0, b0)
        k_end1 = math.hypot(a0 + a1, b0 + b1)
        k_peak = max(k_end0, k_end1)
        if k_peak*s > self.thetaMax and k_peak > 0:
            return coeffs*(self.thetaMax/(k_peak*s))
        return coeffs

    def _applyLimits(self, coeffs1: np.ndarray, s1: float, coeffs2: np.ndarray, s2: float):
        ''' Clamp both sections' arc lengths to [sMin, sMax] and their peak
            bend to thetaMax - same structure as
            ConstantCurvature._applyLimits, including propagating half of
            section 1's length correction onto section 2 (the same
            _couplingError relationship, enforced again here in case the
            LM solve left a small residual and the s1 clamp then broke it).'''
        hits = []
        s1c = min(max(s1, self.sMin), self.sMax)
        if s1c != s1:
            hits.append(f"sect1 s {s1:.1f} -> {s1c:.1f}")
        s2c = s2 + 0.5*(s1c - s1)
        s2cl = min(max(s2c, self.sMin), self.sMax)
        if s2cl != s2c:
            hits.append(f"sect2 s {s2c:.1f} -> {s2cl:.1f}")
        s2c = s2cl

        coeffs1c = self._clampBend(coeffs1, s1c)
        if not np.array_equal(coeffs1c, coeffs1):
            hits.append(f"sect1 bend clamped to {self.thetaMax:.3f} rad")
        coeffs2c = self._clampBend(coeffs2, s2c)
        if not np.array_equal(coeffs2c, coeffs2):
            hits.append(f"sect2 bend clamped to {self.thetaMax:.3f} rad")

        return coeffs1c, s1c, coeffs2c, s2c, hits

    # ---------------- cable / motor conversion ----------------

    def computeMotorTargets(self, targGlobPos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Mirrors ConstantCurvature.PCC_Control: shape -> cable lengths ->
            motor positions/velocities.'''
        coeffs1, s1, coeffs2, s2 = self._unpackShape(self.targShapePos)
        coeffs1, s1, coeffs2, s2, self.limitsHit = self._applyLimits(coeffs1, s1, coeffs2, s2)
        self.targShapePos = np.concatenate([coeffs1, [s1], coeffs2, [s2]])

        offsets1 = np.array([0, PI/3, 2*PI/3, PI, -2*PI/3, -PI/3])
        offsets2 = np.array([0, 2*PI/3, -2*PI/3])          # only 3 cables run through section 2

        l1 = self.cableLength(coeffs1, s1, offsets1)
        l2 = self.cableLength(coeffs2, s2, offsets2)
        la, lc, le = l1[0]+l2[0], l1[2]+l2[1], l1[4]+l2[2]

        _targCablePos = np.array([la, l1[1], lc, l1[3], le, l1[5]])
        self.targCablePos = _targCablePos

        _targEgoPos = (self.initCableLen - _targCablePos)/self.spoolRadius

        now = time.monotonic()
        if hasattr(self, "_lastEgoPos"):
            dt = now - self._lastEgoTime
            if 0 < dt <= 0.5:
                _targEgoVel = (_targEgoPos - self._lastEgoPos)/dt
            else:
                _targEgoVel = np.zeros(6)
        else:
            _targEgoVel = np.zeros(6)
        self._lastEgoPos = _targEgoPos
        self._lastEgoTime = now

        return _targEgoPos, _targEgoVel
