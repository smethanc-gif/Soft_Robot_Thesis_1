## Goal - Write Inverse Kinematic Control For Cartesian Position to Motor Position
'''
Contains PCC Math Transforms from Cartesian to Actuator Space.\n
Additionally holds the configuration data for the physical continuum robot.\n
Uses an iterative "guess and check" method to determine midpoint between bent sections.\n
'''
import math
import time

PI = math.pi

class ContinuumRobot:
    def __init__(self):
        '''Initialise the 2 Section Robot for PCC Control\n
        Geometric Descriptions\n
        Control Offsets\n
        Physical Limits\n'''
# GEOMETRIC DESCRIPTION

        self.nodes              = 14
        self.sections           = 2
        self.robotRadius        = 15.25 # mm
        self.sectL              = 148.5 #mm
        self.initCableLen       = [self.sectL*2, self.sectL] * 3
        self.motorShaftDiameter = 10
        self.spoolRadius        = self.motorShaftDiameter/2    # mm of cable per rad of shaft
        self.toolLength         = 30 # mm
# CONTROL OFFSETS
        self.rotationalOffset   = 2.8623 # radians 0.297 0.349

# PHYSICAL LIMITS
        self.sMin               = 0.8*self.sectL          # section arc length range (mm)
        self.sMax               = self.sectL
        self.thetaMax           = PI/2                # max bend angle k*s per section
        self.limitsHit          = []                 # which limits clamped on the last solve

    def setTarget(self, _targGlobPos: list[float], _targGlobVel: list[float])-> bool:
        if len(_targGlobPos) != 6 or len(_targGlobVel) != 6:
            print("Error found in targetting\n")
            return False
        self.targGlobPos = _targGlobPos
        self.targGlobVel = _targGlobVel
        #print(_targGlobPos,"\n")
        return True

    def midpointCalc(self) -> list[float]:
        ''' Solve for the midpoint position that makes the tool tilt (Rx, Ry)
            come out at the target as well as the tool position. The tangent
            constraints only fix 2 of the 3 midpoint coordinates; the leftover
            freedom is used to satisfy the section coupling ds2 = ds1/2.'''
        x, y, z, Rx, Ry, Rz = self.targGlobPos
        p2 = [x, y, z]
        tT = self._tangentFromAngles(Rx, Ry)

# Checks if midpoint has already been calculated, and adjusts it 
# based on the error of the tip point (p2) to reach new midpoint
        if hasattr(self, "targMidGlobPos"):
            p1 = self.targMidGlobPos[:3]        
        else:
            p1 = self.simpleMidpointGuess(p2, tT)
        p1, err = self._tangentSolve(p1, p2, tT)

        if err > 1e-8:                          # retry from the geometric seed
            p1, err = self._tangentSolve(self.simpleMidpointGuess(p2, tT), p2, tT)
        p1, err = self._coupleSections(p1, p2, tT, err)

        # Midpoint orientation (section-1 end tangent), for logging
        k1, r1, s1, _, _, _ = self._solveSections(p1, p2)
        _, R1 = self.arcFK(k1, r1, s1)
        RxMid = math.atan2(R1[1][2], math.hypot(R1[0][2], R1[2][2]))
        RyMid = math.atan2(R1[0][2], R1[2][2])

        self.targMidGlobPos = [p1[0], p1[1], p1[2], RxMid, RyMid, Rz/2]
        return self.targMidGlobPos

    @staticmethod
    def _tangentFromAngles(Rx: float, Ry: float) -> list[float]:
        ''' Returns a unit vector  in the direction the tooltip is pointing\n
            In the form [x, y, z]'''
        return [math.cos(Rx)*math.sin(Ry), math.sin(Rx), math.cos(Rx)*math.cos(Ry)]

    def toolTransform(self, _toolGlobPos: list[float]) -> list[float]:
        '''Calc the endpoint for the bending sections'''
        x, y, z, Rx, Ry, Rz = _toolGlobPos
        t = self._tangentFromAngles(Rx, Ry)
        L = self.toolLength
        return [x - L*t[0], y - L*t[1], z - L*t[2], Rx, Ry, Rz]
    
    def _solveSections(self, p1: list[float], p2: list[float]) -> list[float]:
        ''' Curvature parameters of both sections, midpoint (p1) and tip point (p2)'''
        k1, r1, s1 = self.arcIK(p1[0], p1[1], p1[2])
        _, R1 = self.arcFK(k1, r1, s1)
        wx, wy, wz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        dx = R1[0][0]*wx + R1[1][0]*wy + R1[2][0]*wz
        dy = R1[0][1]*wx + R1[1][1]*wy + R1[2][1]*wz
        dz = R1[0][2]*wx + R1[1][2]*wy + R1[2][2]*wz
        k2, r2, s2 = self.arcIK(dx, dy, dz)
        return k1, r1, s1, k2, r2, s2

    def _tangentError(self, p1: list[float], p2: list[float], targRot: list[float])-> list[float]:
        ''' Full vector difference between the tool axis and the target axis:
        its only zero is the correctly aligned direction (2 components
        would either degenerate at a pole or also vanish anti-parallel)'''
        k1, r1, s1, k2, r2, s2 = self._solveSections(p1, p2)
        _, R1 = self.arcFK(k1, r1, s1)
        _, R2 = self.arcFK(k2, r2, s2)
        tz1_2 = [R2[0][2], R2[1][2], R2[2][2]]  # Z component of rotation vector. Transform from p1 to p2
        
        tx0_2 = R1[0][0] * tz1_2[0] + R1[0][1] * tz1_2[1] + R1[0][2] * tz1_2[2]     # X component of rotation vector. Transform from p0 to p2
        ty0_2 = R1[1][0] * tz1_2[0] + R1[1][1] * tz1_2[1] + R1[1][2] * tz1_2[2]     # Y component of rotation vector. Transform from p0 to p2
        tz0_2 = R1[2][0] * tz1_2[0] + R1[2][1] * tz1_2[1] + R1[2][2] * tz1_2[2]     # Z component of rotation vector. Transform from p0 to p2
        return [tx0_2 - targRot[0], ty0_2 - targRot[1], tz0_2 - targRot[2]]         # Returns the error between the target global transform and the calculated target transform
    
    def simpleMidpointGuess(self, p2: list[float], r_23) -> list[float]:
        '''Identifies MidPoint as halfway to end of bending point\n
        This works as in the initial position (straight up), the midpoint is easy to identify'''
        return p2[0]/2, p2[1]/2, p2[2]/2

    def _tangentJacobian(self, p1, p2, tT, F):
        ''' Finite-difference 3x3 Jacobian of the tangent error'''
        J = [[0.0]*3 for _ in range(3)]
        for j in range(3):
            h = 1e-5*max(1.0, abs(p1[j]))
            pp = list(p1); pp[j] += h
            Fp = self._tangentError(pp, p2, tT)
            for i in range(3):
                J[i][j] = (Fp[i] - F[i])/h
        return J

    @staticmethod
    def _solve3(A, b):
        '''3x3 linear solve, Gaussian elimination with partial pivoting'''
        M = [A[i][:] + [b[i]] for i in range(3)]
        for c in range(3):
            piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
            if abs(M[piv][c]) < 1e-14:
                return None
            M[c], M[piv] = M[piv], M[c]
            for r in range(3):
                if r != c:
                    f = M[r][c]/M[c][c]
                    for k in range(c, 4):
                        M[r][k] -= f*M[c][k]
        return [M[r][3]/M[r][r] for r in range(3)]

    def _tangentSolve(self, p1, p2, tT):
        '''Levenberg-Marquardt on the tangent error over the 3 midpoint
        coordinates (the error lives on a sphere so its Jacobian is only
        rank 2 - the damping handles that and the leftover null direction)'''
        p1  = list(p1)
        F   = self._tangentError(p1, p2, tT)
        fn  = math.sqrt(sum(f*f for f in F))
        lam = 1e-6

        for _ in range(60):
            if fn < 1e-10:
                break
            J = self._tangentJacobian(p1, p2, tT, F)
            A = [[sum(J[k][i]*J[k][j] for k in range(3)) + (lam if i == j else 0.0) ## Adds small offset lam to diagonal values
                  for j in range(3)] for i in range(3)] 
            b = [-sum(J[k][i]*F[k] for k in range(3)) for i in range(3)]
            step = self._solve3(A, b)
            if step is None:
                lam *= 10.0
                if lam > 1e3:
                    break
                continue
            n = math.sqrt(sum(c*c for c in step))
            if n > 40.0:
                step = [c*40.0/n for c in step]
            cand = [p1[i] + step[i] for i in range(3)]
            Fc = self._tangentError(cand, p2, tT)
            fc = math.sqrt(sum(f*f for f in Fc))
            if fc < fn:
                p1, F, fn = cand, Fc, fc
                lam = max(lam*0.2, 1e-9)
            else:
                lam *= 10.0
                if lam > 1e3:
                    break
        return p1, fn

    def _couplingError(self, p1, p2):
        ''' Mechanical coupling: the section-2 length change (from nominal
            sectL) must be half the section-1 length change'''
        _, _, s1, _, _, s2 = self._solveSections(p1, p2)
        return (s2 - self.sectL) - 0.5*(s1 - self.sectL)

    def _coupleSections(self, p1, p2, tT, err):
        ''' Walk along the null space of the tangent constraints towards
            satisfying the s1/s2 coupling, re-polishing the tangent solve
            after each step'''
        g = self._couplingError(p1, p2)
        best = (abs(g), list(p1), err)
        for _ in range(12):
            if abs(g) < 1e-6:
                break
            F = self._tangentError(p1, p2, tT)
            J = self._tangentJacobian(p1, p2, tT, F)
            # Null direction of the rank-2 Jacobian: largest cross product
            # of any two of its rows
            n = [0.0, 0.0, 0.0]
            nn = 0.0
            for (a, b) in ((0, 1), (0, 2), (1, 2)):
                c = [J[a][1]*J[b][2] - J[a][2]*J[b][1],
                     J[a][2]*J[b][0] - J[a][0]*J[b][2],
                     J[a][0]*J[b][1] - J[a][1]*J[b][0]]
                cn = math.sqrt(sum(v*v for v in c))
                if cn > nn:
                    n, nn = c, cn
            if nn < 1e-12:
                break
            n = [v/nn for v in n]
            gp = self._couplingError([p1[i] + n[i] for i in range(3)], p2)
            dg = gp - g
            if abs(dg) < 1e-9:
                break
            alpha = max(-40.0, min(40.0, -g/dg))
            cand = [p1[i] + alpha*n[i] for i in range(3)]
            cand, cerr = self._tangentSolve(cand, p2, tT)
            cg = self._couplingError(cand, p2)
            if cerr < 1e-8 and abs(cg) < best[0]:
                best = (abs(cg), list(cand), cerr)
                p1, g = cand, cg
            else:
                break
        return best[1], best[2]


    @staticmethod
    def arcIK(x: float, y: float, z: float)-> list[float]:
        '''Single constant-curvature section IK: tip position -> (k, r, s)'''
        normXY = math.sqrt(x**2 + y**2)
        ###          CHECK SINGULARITIES          ###
        if normXY <= 0.05:                             ## i.e. The section doesn't bend
            return 0.0, 0.0, z
        
        k       = 2*normXY/(normXY**2 + z**2)                # Inverse of radius of curvature
        r       = math.atan2(y, x)                           # Bend direction
        theta   = 2*math.atan2(normXY, z)                    # Bend angle
        s       = theta/k                                    # Elongation/ Length of curvature section
        return k, r, s

    @staticmethod
    def arcFK(k: float, r: float, s: float):
        ''' Single section FK: (k, r, s) -> tip position and rotation matrix'''
        # R = Rz(r) * Ry(k*s) * Rz(-r)
        theta = k*s
        ct, st = math.cos(theta), math.sin(theta)
        cp, sp = math.cos(r), math.sin(r)
        if k == 0:
            p = [0.0, 0.0, s]
        else:
            p = [ (1/k)*(1 - ct)*cp, (1/k)*(1 - ct)*sp, (1/k)*st]    # Cartesian Position [x, y, z]
        R = [                                               
            [cp*cp*ct + sp*sp, sp*cp*(ct - 1),   cp*st],    
            [sp*cp*(ct - 1),   sp*sp*ct + cp*cp, sp*st],    
            [-cp*st,           -sp*st,           ct],       
        ]
        return p, R     

    def _applyLimits(self, k1: float, s1: float, k2: float, s2: float) -> list[float]:
        ''' Clamp to the hardware limits (sMin <= s <= sMax, bend k*s <=
            thetaMax). A clamp correction to s1 propagates half to s2 so the
            commanded pair stays consistent with the mechanical coupling'''
        hits = []
        s1c = min(max(s1, self.sMin), self.sMax)
        if s1c != s1:
            hits.append(f"sect1 s {s1:.1f} -> {s1c:.1f}")
        s2c = s2 + 0.5*(s1c - s1)
        s2cl = min(max(s2c, self.sMin), self.sMax)
        if s2cl != s2c:
            hits.append(f"sect2 s {s2c:.1f} -> {s2cl:.1f}")
        s2c = s2cl
        if k1*s1c > self.thetaMax:
            hits.append(f"sect1 bend {k1*s1c:.3f} -> {self.thetaMax:.3f} rad")
            k1 = self.thetaMax/s1c
        if k2*s2c > self.thetaMax:
            hits.append(f"sect2 bend {k2*s2c:.3f} -> {self.thetaMax:.3f} rad")
            k2 = self.thetaMax/s2c
        return k1, s1c, k2, s2c, hits

    def PCC_Control (self) -> tuple[list[float], list[float]]:
        k1, r1, s1, k2, r2, s2 = self._solveSections(self.targMidGlobPos[:3], self.targGlobPos[:3])

        k1, s1, k2, s2, self.limitsHit = self._applyLimits(k1, s1, k2, s2)

        self.targCylPos = [k1,r1,s1,k2,r2,s2]
        #print(self.targCylPos, end="\n")
        n1 = n2 = self.nodes
        OFS = self.rotationalOffset
        d = self.robotRadius

        n1calc = 2*n1*math.sin(s1*k1/(2*n1))    # Calculations for future use
        n2calc = 2*n2*math.sin(s2*k2/(2*n2))
###          ACCOUNT FOR SINGULARITY          ###
        if k1 == 0:
            la1 = lb = lc1 = ld = le1 = lf = s1
        else:
            la1 = n1calc*(1/k1 - d*math.sin(r1 + OFS)) # Cable A (section 1 part)
            lb = n1calc*(1/k1 - d*math.sin(r1 + OFS + PI/3)) # Cable B
            lc1 = n1calc*(1/k1 - d*math.sin(r1 + OFS + 2*PI/3)) # Cable C (section 1 part)
            ld = n1calc*(1/k1 - d*math.sin(r1 + OFS + PI)) # Cable D
            le1 = n1calc*(1/k1 - d*math.sin(r1 + OFS - 2*PI/3)) # Cable E (section 1 part)
            lf = n1calc*(1/k1 - d*math.sin(r1 + OFS - PI/3)) # Cable F

        if k2 == 0:
            
            la2 = lc2 = le2 = (s2)
        else:
            la2 = n2calc*(1/k2 - d*math.sin(r2 + OFS))   # Cable A
            lc2 = n2calc*(1/k2 - d*math.sin(r2 + OFS + 2*PI/3))   # Cable C
            le2 = n2calc*(1/k2 - d*math.sin(r2 + OFS - 2*PI/3))   # Cable E
        la = la1+la2
        lc = lc1 + lc2
        le = le1 + le2

        _targCablePos = [
            la,
            lb,
            lc,
            ld,
            le,
            lf
        ]
        self.targCablePos = _targCablePos
        _targEgoPos = [                          # Motor Position (shaft radians)
            (self.initCableLen[i] - _targCablePos[i])/self.spoolRadius for i in range(6)
        ]

        # Motor velocity (rad/s): finite difference of successive targets,
        # gives the STM32 PD loop a velocity reference when streaming
        now = time.monotonic()
        if hasattr(self, "_lastEgoPos"):
            dt = now - self._lastEgoTime
            if 0 < dt <= 0.5:                   # only when streaming targets
                _targEgoVel = [(_targEgoPos[i] - self._lastEgoPos[i])/dt
                              for i in range(6)]
            else:
                _targEgoVel = [0.0] * 6
        else:
            _targEgoVel = [0.0] * 6
        self._lastEgoPos = _targEgoPos
        self._lastEgoTime = now

        return _targEgoPos, _targEgoVel
    
    def getCylPos(self):
        return self.targCylPos

    def getCablePos(self):
        return self.targCablePos

    def forwardKinematics(self) -> list[float]:
        ''' Recompute the tool pose from the last solved arc parameters,
            used to verify the inverse kinematics'''
        k1, r1, s1, k2, r2, s2 = self.targCylPos
        p1, R1 = self.arcFK(k1, r1, s1)
        p2, R2 = self.arcFK(k2, r2, s2)
        tip = [p1[i] + sum(R1[i][j]*p2[j] for j in range(3)) for i in range(3)]
        # Tool axis = third column of R1*R2
        t2 = [R2[0][2], R2[1][2], R2[2][2]]
        t = [sum(R1[i][j]*t2[j] for j in range(3)) for i in range(3)]
        # Straight rigid tool: tooltip sits toolLength along the tool axis,
        # keeping this consistent with the tooltip targets fed to FullIK
        tip = [tip[i] + self.toolLength*t[i] for i in range(3)]
        Rx = math.atan2(t[1], math.hypot(t[0], t[2]))   # tilt towards +Y
        Ry = math.atan2(t[0], t[2])                     # tilt towards +X
        return [tip[0], tip[1], tip[2], Rx, Ry, 0.0]
    
    def FullIK(self, _targGlobPos: list[float], _targGlobVel: list[float]) -> list[float]:
        ''' Runs toolTransform, setTarget, midpointCalc, and PCC_Control.\n
            returns the motor positions necessary to get to the desired Cartesian Coordinates'''
        targBendPos = self.toolTransform(_targGlobPos)
        self.setTarget(targBendPos, _targGlobVel)
        self.midpointCalc()
        return self.PCC_Control()

