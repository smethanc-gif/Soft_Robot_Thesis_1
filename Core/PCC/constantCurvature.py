'''"PCC" method: piecewise-constant-curvature kinematics and cable-length model for the
2-section continuum robot.

Transforms a Cartesian target into actuator space: solves for the midpoint
between the two bent sections (so the tool tilt comes out right as well as
the tool position), turns both sections into arc parameters, then into cable
lengths and motor angles. The robot's dimensions and limits come from
Core/robotGeometry.json.
'''
import math
import time
import numpy as np
from Core.settings import loadRobotGeometry


class ConstantCurvature:
    def __init__(self):
        '''2-section robot for PCC control: geometry, control offsets and physical limits.'''
        geometry = loadRobotGeometry()          # Core/robotGeometry.json - units and meanings in its "notes"
        # ---- Geometric description ----
        self.nodes              = geometry["nodes"]
        self.sections           = geometry["sections"]
        self.robotRadius        = geometry["robotRadius"]
        self.sectL              = geometry["sectL"]
        self.initCableLen       = np.array([self.sectL*2, self.sectL] * 3)
        self.spoolRadius        = geometry["spoolRadius"]
        self.toolLength         = geometry["toolLength"]
        # ---- Control offsets ----
        self.rotationalOffset   = geometry["rotationalOffset"]
        self.tendonAngles1      = np.radians(geometry["tendonAnglesDeg"]["section1"])
        self.tendonAngles2      = np.radians(geometry["tendonAnglesDeg"]["section2"])
        # ---- Physical limits ----
        self.sMin               = geometry["sMinRatio"]*self.sectL
        self.sMax               = self.sectL
        self.thetaMax           = math.radians(geometry["thetaMaxDeg"])
        self.limitsHit          = []                 # which limits clamped on the last solve
        self.limitFlags         = [0, 0, 0, 0]       # same, as 0/1 for logging: [s1, s2, bend1, bend2]

    def calcMidpoint(self, _targBendPos: np.ndarray) -> np.ndarray:
        '''Solves for the midpoint position that makes the tool tilt (Rx, Ry) come out at the
            target as well as the tool position. The tangent constraints only fix 2 of the 3
            midpoint coordinates; the leftover freedom is used to satisfy the section coupling
            ds2 = ds1/2. Warm-started from the previous midpoint.'''
        x, y, z, Rx, Ry, Rz = _targBendPos
        p2 = np.array([x, y, z])
        tT = self._calcTangentFromAngles(Rx, Ry)

        # Start from the last midpoint if there is one, otherwise from the geometric guess
        if hasattr(self, "targMidGlobPos"):
            p1 = self.targMidGlobPos[:3]
        else:
            p1 = self.guessMidpoint(p2)

        p1, err = self._solveTangent(p1, p2, tT)

        if err > 1e-8:                          # retry from the geometric seed
            p1, err = self._solveTangent(self.guessMidpoint(p2), p2, tT)
        p1, err = self._coupleSections(p1, p2, tT, err)

        # Midpoint orientation (section-1 end tangent), for logging
        k1, r1, s1, _, _, _ = self._solveSections(p1, p2)
        _, R1 = self.calcArcFK(k1, r1, s1)
        RxMid = math.atan2(R1[1][2], math.hypot(R1[0][2], R1[2][2]))
        RyMid = math.atan2(R1[0][2], R1[2][2])

        self.targMidGlobPos = np.array([p1[0], p1[1], p1[2], RxMid, RyMid, Rz/2])
        return self.targMidGlobPos

    @staticmethod
    def _calcTangentFromAngles(_Rx: float, _Ry: float) -> np.ndarray:
        '''Unit vector [x, y, z] in the direction the tool tip is pointing.'''
        return np.array([math.cos(_Rx)*math.sin(_Ry), math.sin(_Rx), math.cos(_Rx)*math.cos(_Ry)])

    def _solveSections(self, _p1: np.ndarray, _p2: np.ndarray) -> tuple[float, float, float, float, float, float]:
        '''Arc parameters (k1, r1, s1, k2, r2, s2) of both sections from the midpoint _p1 and tip point _p2.'''
        k1, r1, s1 = self.calcArcIK(_p1[0], _p1[1], _p1[2])
        _, R1 = self.calcArcFK(k1, r1, s1)
        wp = _p2 - _p1
        dp = R1.T @ wp
        k2, r2, s2 = self.calcArcIK(dp[0], dp[1], dp[2])
        return k1, r1, s1, k2, r2, s2

    def _calcTangentError(self, _p1: np.ndarray, _p2: np.ndarray, _targRot: np.ndarray) -> np.ndarray:
        '''Full vector difference between the tool axis and the target axis: its only zero is the
            correctly aligned direction (2 components would either degenerate at a pole or also
            vanish anti-parallel).'''
        k1, r1, s1, k2, r2, s2 = self._solveSections(_p1, _p2)
        _, R1 = self.calcArcFK(k1, r1, s1)
        _, R2 = self.calcArcFK(k2, r2, s2)
        toolAxis1 = R2[:,2]                     # tool axis in section 1's end frame
        toolAxis0 = R1 @ toolAxis1              # ... and in the base frame
        err = toolAxis0 - _targRot
        return err

    def guessMidpoint(self, _p2: np.ndarray) -> np.ndarray:
        '''Midpoint guess: halfway to the end of the bending sections (exact when straight).'''
        return _p2/2

    def _calcTangentJacobian(self, _p1: np.ndarray, _p2: np.ndarray, _tT: np.ndarray, _F: np.ndarray) -> np.ndarray:
        '''Finite-difference 3x3 Jacobian of the tangent error with respect to the midpoint.'''
        J = np.zeros((3,3))
        for j in range(3):
            h = 1e-5*max(1.0, abs(_p1[j]))
            pp = _p1.copy(); pp[j] += h
            Fp = self._calcTangentError(pp, _p2, _tT)
            J[:, j] = (Fp - _F) / h
        return J

    @staticmethod
    def _solveLinear3(_A: np.ndarray, _b: np.ndarray):
        '''Solves the 3x3 system _A x = _b by Gaussian elimination with partial pivoting (None if singular).'''
        M = np.hstack([_A, np.asarray(_b).reshape(3,1)])

        for c in range(3):
            piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
            if abs(M[piv][c]) < 1e-14:
                return None
            M[[c, piv]] = M[[piv, c]]
            for r in range(3):
                if r != c:
                    f = M[r][c]/M[c][c]
                    for k in range(c, 4):
                        M[r][k] -= f*M[c][k]
        return [M[r][3]/M[r][r] for r in range(3)]

    def _solveTangent(self, _p1, _p2, _tT):
        '''Levenberg-Marquardt on the tangent error over the 3 midpoint coordinates (the error lives
            on a sphere so its Jacobian is only rank 2 - the damping handles that and the leftover
            null direction). Returns (midpoint, error norm).'''
        F   = self._calcTangentError(_p1, _p2, _tT)
        fn  = math.sqrt(sum(f*f for f in F))    # error norm
        lam =  1e-6                             # damping

        for _ in range(60):
            if fn < 1e-10:
                break
            J = self._calcTangentJacobian(_p1, _p2, _tT, F)
            A = J.T @ J + lam*np.eye(3)
            b = -J.T @ F

            step = self._solveLinear3(A, b)
            if step is None:
                lam *= 10.0
                if lam > 1e3:
                    break
                continue
            n = math.sqrt(sum(c*c for c in step))
            if n > 40.0:
                step = [c*40.0/n for c in step]
            cand = [_p1[i] + step[i] for i in range(3)]
            Fc = self._calcTangentError(cand, _p2, _tT)
            fc = math.sqrt(sum(f*f for f in Fc))
            if fc < fn:
                _p1, F, fn = cand, Fc, fc
                lam = max(lam*0.2, 1e-9)
            else:
                lam *= 10.0
                if lam > 1e3:
                    break
        return _p1, fn

    def _calcCouplingError(self, _p1, _p2):
        '''Mechanical coupling error: the section-2 length change (from nominal sectL) must be half
            the section-1 length change.'''
        _, _, s1, _, _, s2 = self._solveSections(_p1, _p2)
        return (s2 - self.sectL) - 0.5*(s1 - self.sectL)

    def _coupleSections(self, _p1, _p2, _tT, _err):
        '''Walks along the null space of the tangent constraints towards satisfying the s1/s2
            coupling, re-polishing the tangent solve after each step. Returns (midpoint, tangent error).'''
        g = self._calcCouplingError(_p1, _p2)
        best = (abs(g), list(_p1), _err)       # best solution so far

        for _ in range(12):
            if abs(g) < 1e-6:
                break
            F = self._calcTangentError(_p1, _p2, _tT)
            J = self._calcTangentJacobian(_p1, _p2, _tT, F)
            # Null direction of the rank-2 Jacobian: largest cross product
            # of any two of its rows
            pairs = ((0, 1), (0, 2), (1, 2))
            cross = np.array([np.cross(J[a], J[b]) for a, b in pairs])   # shape (3,3): one candidate per row-pair
            norms = np.linalg.norm(cross, axis=1)
            bestIdx = np.argmax(norms)

            nn = norms[bestIdx]
            if nn < 1e-12:
                break
            n = cross[bestIdx] / nn

            gp = self._calcCouplingError(_p1 + n, _p2)
            dg = gp - g
            if abs(dg) < 1e-9:
                break
            alpha = max(-40.0, min(40.0, -g/dg))
            cand = _p1 + alpha*n
            cand, cerr = self._solveTangent(cand, _p2, _tT)
            cg = self._calcCouplingError(cand, _p2)

            if cerr < 1e-8 and abs(cg) < best[0]:
                best = (abs(cg), cand.copy(), _err)
                _p1, g = cand, cg
            else:
                break
        return best[1], best[2]

    @staticmethod
    def calcArcIK(_x: float, _y: float, _z: float) -> tuple[float, float, float]:
        '''Single constant-curvature section IK: tip position -> (k, r, s).'''
        normXY = math.sqrt(_x**2 + _y**2)
        if normXY <= 0.05:                                  # singularity: the section doesn't bend
            return 0.0, 0.0, _z

        k       = 2*normXY/(normXY**2 + _z**2)              # inverse of radius of curvature
        r       = math.atan2(_y, _x)                        # bend direction
        theta   = 2*math.atan2(normXY, _z)                  # bend angle
        s       = theta/k                                   # arc length of the section
        return k, r, s

    @staticmethod
    def calcArcFK(_k: float, _r: float, _s: float) -> tuple[np.ndarray, np.ndarray]:
        '''Single section FK: (k, r, s) -> tip position and rotation matrix.'''
        # R = Rz(r) * Ry(k*s) * Rz(-r)
        theta = _k*_s
        ct, st = math.cos(theta), math.sin(theta)
        cp, sp = math.cos(_r), math.sin(_r)
        if _k == 0:
            p = np.array([0.0, 0.0, _s])
        else:
            p = np.array([ (1/_k)*(1 - ct)*cp, (1/_k)*(1 - ct)*sp, (1/_k)*st])    # Cartesian Position [x, y, z]
        R = np.array([
            [cp*cp*ct + sp*sp, sp*cp*(ct - 1),   cp*st],
            [sp*cp*(ct - 1),   sp*sp*ct + cp*cp, sp*st],
            [-cp*st,           -sp*st,           ct],
        ])
        return p, R

    def _applyLimits(self, _k1: float, _s1: float, _k2: float, _s2: float) -> tuple:
        '''Clamps to the hardware limits (sMin <= s <= sMax, bend k*s <= thetaMax). A clamp
            correction to s1 propagates half to s2 so the commanded pair stays consistent with the
            mechanical coupling. Returns (k1, s1, k2, s2, list of limits hit).'''
        hits = []
        flags = [0, 0, 0, 0]
        tol = 1e-3                              # mm - flag only real clamps, not float round-off at sMax
        s1c = min(max(_s1, self.sMin), self.sMax)
        if s1c != _s1:
            hits.append(f"sect1 s {_s1:.1f} -> {s1c:.1f}")
            flags[0] = int(abs(s1c - _s1) > tol)
        s2c = _s2 + 0.5*(s1c - _s1)
        s2cl = min(max(s2c, self.sMin), self.sMax)
        if s2cl != s2c:
            hits.append(f"sect2 s {s2c:.1f} -> {s2cl:.1f}")
            flags[1] = int(abs(s2cl - s2c) > tol)
        s2c = s2cl
        if _k1*s1c > self.thetaMax:
            hits.append(f"sect1 bend {_k1*s1c:.3f} -> {self.thetaMax:.3f} rad")
            flags[2] = 1
            _k1 = self.thetaMax/s1c
        if _k2*s2c > self.thetaMax:
            hits.append(f"sect2 bend {_k2*s2c:.3f} -> {self.thetaMax:.3f} rad")
            flags[3] = 1
            _k2 = self.thetaMax/s2c
        self.limitFlags = flags
        return _k1, s1c, _k2, s2c, hits

    def calcEgoTargets(self, _targBendPos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Motor positions and velocities for the bend-end target (after calcMidpoint): arc
            parameters of both sections, clamped to the limits, then cable lengths, then motor angles.'''
        k1, r1, s1, k2, r2, s2 = self._solveSections(self.targMidGlobPos[:3], _targBendPos[:3])

        k1, s1, k2, s2, self.limitsHit = self._applyLimits(k1, s1, k2, s2)

        self.targConfigPos = [k1,r1,s1,k2,r2,s2]
        n1 = n2 = self.nodes
        rotOffset = self.rotationalOffset
        d = self.robotRadius

        n1calc = 2*n1*math.sin(s1*k1/(2*n1))    # cables run straight between the nodes: chord factor
        n2calc = 2*n2*math.sin(s2*k2/(2*n2))

        offsets1 = self.tendonAngles1
        offsets2 = self.tendonAngles2           # only 3 cables run through section 2

        l1 = np.full(6, s1) if k1 == 0 else n1calc*(1/k1 - d*np.sin(r1 + rotOffset + offsets1))
        l2 = np.full(3, s2) if k2 == 0 else n2calc*(1/k2 - d*np.sin(r2 + rotOffset + offsets2))
        la, lc, le = l1[0]+l2[0], l1[2]+l2[1], l1[4]+l2[2]

        targCablePos = np.array([la, l1[1], lc, l1[3], le, l1[5]])
        self.targCablePos = targCablePos

        targEgoPos = (self.initCableLen - targCablePos)/self.spoolRadius

        # Motor velocity (rad/s): finite difference of successive targets,
        # gives the STM32 PD loop a velocity reference when streaming
        now = time.monotonic()
        if hasattr(self, "_lastEgoPos"):
            dt = now - self._lastEgoTime
            if 0 < dt <= 0.5:                   # only when streaming targets
                targEgoVel = (targEgoPos - self._lastEgoPos)/dt
            else:
                targEgoVel = [0.0] * 6
        else:
            targEgoVel = [0.0] * 6
        self._lastEgoPos = targEgoPos
        self._lastEgoTime = now

        return targEgoPos, targEgoVel

    def calcForwardKinematics(self) -> np.ndarray:
        '''Tool pose [x, y, z, Rx, Ry, 0] recomputed from the last solved arc parameters, used to
            verify the inverse kinematics.'''
        k1, r1, s1, k2, r2, s2 = self.targConfigPos
        p1, R1 = self.calcArcFK(k1, r1, s1)
        p2, R2 = self.calcArcFK(k2, r2, s2)
        tip = p1 + R1 @ p2
        # Tool axis = third column of R1*R2
        t2 = R2[:,2]
        t = R1 @ t2
        # Straight rigid tool: tooltip sits toolLength along the tool axis,
        # keeping this consistent with the tooltip targets fed to solveFullIK
        tip = tip + self.toolLength * t
        Rx = math.atan2(t[1], math.hypot(t[0], t[2]))   # tilt towards +Y
        Ry = math.atan2(t[0], t[2])                     # tilt towards +X
        return np.array([tip[0], tip[1], tip[2], Rx, Ry, 0.0])
