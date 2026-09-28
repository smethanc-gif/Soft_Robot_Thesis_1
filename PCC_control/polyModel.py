## Goal - The fitted forward model shared by the PCC_Cable and Poly controllers and by results/polyFit.py
'''
Motor angles -> tip pose, with the constants fitted in results/polyFit.py
(see results/POLY_CONSTANTS.md for how they were derived):

  1. Each motor's angle is scaled by its tendon (cable) gain: the fraction of
     wound cable that actually shows up in the robot's shape.
  2. Cable lengths -> each section's PCC-equivalent arc length s and bend v
     (complex: direction + size), by inverting PCC_Control's cable equations.
  3. Poly curvature along each section, sigma in [0, 1]:
         kappa_j(sigma) = [ A_j + (sigma - 1/2)*B_j ] / s_j'
     A_j (overall bend) and B_j (slope) are built from v_1, v_2 with the
     fitted constants; s_j' = sectL + lam*(s_j - sectL).
  4. Section 1, then section 2, then the tool - composed as in
     constantCurve.forwardKinematics.
With the default constants (pccParams(), gains = 1) this is exactly PCC.
Everything is vectorised over samples: q is [sample][A..F].
'''
import os
import json
import numpy

try:
    from .constantCurve import ConstantCurvature
except ImportError:
    from constantCurve import ConstantCurvature

# ---------------- robot constants (from constantCurve.py) ----------------
_R = ConstantCurvature()
D = _R.robotRadius
SECT_L = _R.sectL
SPOOL = _R.spoolRadius
TOOL = _R.toolLength
CABLE0 = numpy.array(_R.initCableLen)
N_SUB = 8                               # sub-segments per section when integrating varying curvature

_B1 = _R.rotationalOffset + numpy.radians([0, 60, 120, 180, -120, -60])   # section-1 tendon angles, A..F
_B2 = _R.rotationalOffset + numpy.radians([0, 120, -120])                 # section-2 tendon angles, A, C, E
_W1 = numpy.c_[numpy.sin(_B1), numpy.cos(_B1)]      # l1_i = s1 - D*(W1_i . v1)
_W2 = numpy.c_[numpy.sin(_B2), numpy.cos(_B2)]
_INV1 = numpy.linalg.inv(numpy.c_[numpy.ones(3), -D*_W1[[1, 3, 5]]])     # [s1, v1x, v1y] from l_B, l_D, l_F
_INV2 = numpy.linalg.inv(numpy.c_[numpy.ones(3), -D*_W2])                  # [s2, v2x, v2y] from section-2 part of l_A, l_C, l_E

CONSTANTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poly_constants.json")


# ============================================================
# Motors <-> PCC-equivalent section shapes
# ============================================================

def sectionsFromMotors(q: numpy.ndarray):
    ''' Motor angles [sample][A..F] (rad) -> per-section arc length s (mm) and
        bend vector v (rad, as a complex number: x + iy), by inverting
        PCC_Control's cable model (linearised - within 0.03 mm / 0.02 deg of
        PCC's own arc parameters). Section 2's bend is in section 1's end
        frame, as in PCC.'''
    l = CABLE0 - SPOOL*numpy.atleast_2d(q)
    x1 = l[:, [1, 3, 5]] @ _INV1.T                                   # [s1, v1x, v1y]
    l1_ace = x1[:, :1] - D*(x1[:, 1:] @ _W1[[0, 2, 4]].T)             # section-1 part of A, C, E
    x2 = (l[:, [0, 2, 4]] - l1_ace) @ _INV2.T                        # [s2, v2x, v2y]
    return x1[:, 0], x1[:, 1] + 1j*x1[:, 2], x2[:, 0], x2[:, 1] + 1j*x2[:, 2]


def motorsFromSections(s1, v1, s2, v2) -> numpy.ndarray:
    ''' Exact inverse of sectionsFromMotors: section arc lengths and bends ->
        motor angles [sample][A..F] (rad). '''
    s1, v1, s2, v2 = (numpy.atleast_1d(x) for x in (s1, v1, s2, v2))
    b1 = numpy.c_[v1.real, v1.imag]
    b2 = numpy.c_[v2.real, v2.imag]
    l1 = s1[:, None] - D*(b1 @ _W1.T)                                # section-1 part of all six cables
    l2 = s2[:, None] - D*(b2 @ _W2.T)                                # section-2 part of A, C, E
    l = l1.copy()
    l[:, [0, 2, 4]] += l2
    return (CABLE0 - l)/SPOOL


# ============================================================
# Poly forward kinematics
# ============================================================

def _arcs(kx, ky, ds):
    ''' constantCurve.arcFK for many samples at once. '''
    k = numpy.hypot(kx, ky)
    r = numpy.arctan2(ky, kx)
    th = k*ds
    ct, st, cp, sp = numpy.cos(th), numpy.sin(th), numpy.cos(r), numpy.sin(r)
    small = k < 1e-12
    ks = numpy.where(small, 1.0, k)
    rho = numpy.where(small, 0.0, (1 - ct)/ks)
    z = numpy.where(small, ds, st/ks)
    p = numpy.stack([rho*cp, rho*sp, z], axis=-1)
    R = numpy.empty(k.shape + (3, 3))
    R[:, 0, 0] = cp*cp*ct + sp*sp; R[:, 0, 1] = sp*cp*(ct - 1); R[:, 0, 2] = cp*st
    R[:, 1, 0] = sp*cp*(ct - 1);   R[:, 1, 1] = sp*sp*ct + cp*cp; R[:, 1, 2] = sp*st
    R[:, 2, 0] = -cp*st;           R[:, 2, 1] = -sp*st;           R[:, 2, 2] = ct
    return p, R


def _sectionFK(k0: numpy.ndarray, k1: numpy.ndarray, s: numpy.ndarray):
    ''' One section with complex curvature k0 + k1*sigma (1/mm), split into
        N_SUB constant-curvature pieces. '''
    p = numpy.zeros((len(s), 3))
    R = numpy.broadcast_to(numpy.eye(3), (len(s), 3, 3))
    ds = s/N_SUB
    for i in range(N_SUB):
        kk = k0 + k1*(i + 0.5)/N_SUB
        pi, Ri = _arcs(kk.real, kk.imag, ds)
        p = p + numpy.einsum("mij,mj->mi", R, pi)
        R = R @ Ri
    return p, R


# Parameter layout: complex terms as (re, im) pairs, then the two length gains.
TERMS = ["c0_1", "h0_1", "c1_1", "h1_1", "c0_2", "h0_2", "c1_2", "h1_2"]
EXTRA_TERMS = ["m_1", "m_2", "z_1", "z_2"]        # bend-size (|v|^2) and compression (s - sectL) gain terms
ALL_TERMS = TERMS + EXTRA_TERMS
REAL_TERMS = ["lam_1", "lam_2"]


def pccParams(lam: float = 1.0) -> numpy.ndarray:
    ''' Full parameter vector that reproduces PCC (with length gain lam). '''
    p = numpy.zeros(2*len(ALL_TERMS) + len(REAL_TERMS))
    for name in ("c0_1", "c0_2"):
        p[2*ALL_TERMS.index(name)] = 1.0
    p[-2:] = lam
    return p


def unpack(params: numpy.ndarray) -> dict:
    out = {name: params[2*i] + 1j*params[2*i + 1] for i, name in enumerate(ALL_TERMS)}
    out.update({name: params[2*len(ALL_TERMS) + i] for i, name in enumerate(REAL_TERMS)})
    return out


def pack(constants: dict) -> numpy.ndarray:
    ''' Constants as saved in poly_constants.json -> full parameter vector. '''
    p = pccParams()
    for i, name in enumerate(ALL_TERMS):
        p[2*i], p[2*i + 1] = constants[name]
    p[-2], p[-1] = constants["lam_1"], constants["lam_2"]
    return p


def polyShape(params: numpy.ndarray, q: numpy.ndarray, gains=None) -> dict:
    ''' Both sections' Poly shape for motor angles q: arc lengths s1', s2',
        overall bends A1, A2 (rad, complex = direction + size), curvature
        slopes B1, B2, and the calibrated-PCC bends v1, v2 they grew from.
        gains = per-tendon cable-travel gains A..F (None = 1). '''
    P = unpack(params)
    s1, v1, s2, v2 = sectionsFromMotors(q if gains is None else q*gains)
    g1 = P["c0_1"] + P["m_1"]*abs(v1)**2 + P["z_1"]*(s1 - SECT_L)/SECT_L
    g2 = P["c0_2"] + P["m_2"]*abs(v2)**2 + P["z_2"]*(s2 - SECT_L)/SECT_L
    return {"A1": g1*v1 + P["h0_1"]*v2, "B1": P["c1_1"]*v1 + P["h1_1"]*v2,
            "A2": g2*v2 + P["h0_2"]*v1, "B2": P["c1_2"]*v2 + P["h1_2"]*v1,
            "s1": SECT_L + P["lam_1"]*(s1 - SECT_L), "s2": SECT_L + P["lam_2"]*(s2 - SECT_L),
            "v1": v1, "v2": v2}


def polyPose(params: numpy.ndarray, q: numpy.ndarray, gains=None):
    ''' Predicted tool tip [sample][xyz] and tool axis [sample][xyz]. '''
    S = polyShape(params, q, gains)
    A1, B1, A2, B2, s1p, s2p = S["A1"], S["B1"], S["A2"], S["B2"], S["s1"], S["s2"]
    # kappa(sigma) = (A + (sigma - 1/2)*B)/s' = k0 + k1*sigma
    p1, R1 = _sectionFK((A1 - B1/2)/s1p, B1/s1p, s1p)
    p2, R2 = _sectionFK((A2 - B2/2)/s2p, B2/s2p, s2p)
    axis = (R1 @ R2)[:, :, 2]
    return p1 + numpy.einsum("mij,mj->mi", R1, p2) + TOOL*axis, axis


def polyTip(params: numpy.ndarray, q: numpy.ndarray, gains=None) -> numpy.ndarray:
    ''' Predicted tool-tip positions [sample][xyz] for motor angles q
        (pccParams() and gains = None give PCC's own prediction). '''
    return polyPose(params, q, gains)[0]


# ============================================================
# Fitted constants
# ============================================================

def loadConstants(path: str = CONSTANTS_FILE) -> dict:
    ''' The constants saved by results/polyFit.py:
          gains  - tendon gains A..F
          pcc    - full parameter vector for cable-compensated PCC (length gain only)
          poly   - full parameter vector for the Poly model
          lag    - motor -> camera delay used in the fit (samples) '''
    with open(path) as f:
        C = json.load(f)
    return {"gains": numpy.array(C["tendon_gains_ABCDEF"]),
            "pcc": pccParams(C["pcc_calibrated"]["lam"]),
            "poly": pack(C["poly"]["constants"]),
            "lag": C["lag_samples"]}
