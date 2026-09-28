'''The fitted forward model shared by the PCC_Cable and Poly controllers and by
Results/calibration/polyFit.py.

Motor angles -> tip pose, with the constants fitted in Results/calibration/polyFit.py
(see Results/calibration/polyConstants.md for how they were derived):

  1. Each motor's angle is scaled by its tendon (cable) gain: the fraction of
     wound cable that actually shows up in the robot's shape.
  2. Cable lengths -> each section's PCC-equivalent arc length s and bend v
     (complex: direction + size), by inverting calcEgoTargets' cable equations.
  3. Poly curvature along each section, sigma in [0, 1]:
         kappa_j(sigma) = [ A_j + (sigma - 1/2)*B_j ] / s_j'
     A_j (overall bend) and B_j (slope) are built from v_1, v_2 with the
     fitted constants; s_j' = sectL + lam*(s_j - sectL).
  4. Section 1, then section 2, then the tool - composed as in
     ConstantCurvature.calcForwardKinematics.
With the default constants (getPCCParams(), gains = 1) this is exactly PCC.
Everything is vectorised over samples: q is [sample][A..F].
'''
import os
import json
import numpy as np

from Core.settings import loadRobotGeometry

# ---- Robot constants (Core/robotGeometry.json) ----
_GEOMETRY = loadRobotGeometry()
D = _GEOMETRY["robotRadius"]
SECT_L = _GEOMETRY["sectL"]
SPOOL = _GEOMETRY["spoolRadius"]
TOOL = _GEOMETRY["toolLength"]
CABLE0 = np.array([SECT_L*2, SECT_L]*3)
ROTATIONAL_OFFSET = _GEOMETRY["rotationalOffset"]
N_SUB = 8                               # sub-segments per section when integrating varying curvature

_B1 = ROTATIONAL_OFFSET + np.radians(_GEOMETRY["tendonAnglesDeg"]["section1"])   # section-1 tendon angles, A..F
_B2 = ROTATIONAL_OFFSET + np.radians(_GEOMETRY["tendonAnglesDeg"]["section2"])   # section-2 tendon angles, A, C, E
_W1 = np.c_[np.sin(_B1), np.cos(_B1)]      # l1_i = s1 - D*(W1_i . v1)
_W2 = np.c_[np.sin(_B2), np.cos(_B2)]
_INV1 = np.linalg.inv(np.c_[np.ones(3), -D*_W1[[1, 3, 5]]])     # [s1, v1x, v1y] from l_B, l_D, l_F
_INV2 = np.linalg.inv(np.c_[np.ones(3), -D*_W2])                  # [s2, v2x, v2y] from section-2 part of l_A, l_C, l_E

CONSTANTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyConstants.json")


# ---- Motors <-> PCC-equivalent section shapes ----

def calcSectionsFromMotors(_q: np.ndarray):
    '''Motor angles [sample][A..F] (rad) -> per-section arc length s (mm) and
        bend vector v (rad, as a complex number: x + iy), by inverting
        calcEgoTargets' cable model (linearised - within 0.03 mm / 0.02 deg of
        PCC's own arc parameters). Section 2's bend is in section 1's end
        frame, as in PCC.'''
    cableLen = CABLE0 - SPOOL*np.atleast_2d(_q)
    x1 = cableLen[:, [1, 3, 5]] @ _INV1.T                                   # [s1, v1x, v1y]
    l1ACE = x1[:, :1] - D*(x1[:, 1:] @ _W1[[0, 2, 4]].T)             # section-1 part of A, C, E
    x2 = (cableLen[:, [0, 2, 4]] - l1ACE) @ _INV2.T                        # [s2, v2x, v2y]
    return x1[:, 0], x1[:, 1] + 1j*x1[:, 2], x2[:, 0], x2[:, 1] + 1j*x2[:, 2]


def calcMotorsFromSections(_s1, _v1, _s2, _v2) -> np.ndarray:
    '''Exact inverse of calcSectionsFromMotors: section arc lengths and bends ->
        motor angles [sample][A..F] (rad).'''
    _s1, _v1, _s2, _v2 = (np.atleast_1d(x) for x in (_s1, _v1, _s2, _v2))
    b1 = np.c_[_v1.real, _v1.imag]
    b2 = np.c_[_v2.real, _v2.imag]
    l1 = _s1[:, None] - D*(b1 @ _W1.T)                                # section-1 part of all six cables
    l2 = _s2[:, None] - D*(b2 @ _W2.T)                                # section-2 part of A, C, E
    cableLen = l1.copy()
    cableLen[:, [0, 2, 4]] += l2
    return (CABLE0 - cableLen)/SPOOL


# ---- Poly forward kinematics ----

def _calcArcs(_kx, _ky, _ds):
    '''ConstantCurvature.calcArcFK for many samples at once, from curvature components (kx, ky) and length ds.'''
    k = np.hypot(_kx, _ky)
    r = np.arctan2(_ky, _kx)
    th = k*_ds
    ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(r), np.sin(r)
    small = k < 1e-12
    ks = np.where(small, 1.0, k)
    rho = np.where(small, 0.0, (1 - ct)/ks)
    z = np.where(small, _ds, st/ks)
    p = np.stack([rho*cp, rho*sp, z], axis=-1)
    R = np.empty(k.shape + (3, 3))
    R[:, 0, 0] = cp*cp*ct + sp*sp; R[:, 0, 1] = sp*cp*(ct - 1); R[:, 0, 2] = cp*st
    R[:, 1, 0] = sp*cp*(ct - 1);   R[:, 1, 1] = sp*sp*ct + cp*cp; R[:, 1, 2] = sp*st
    R[:, 2, 0] = -cp*st;           R[:, 2, 1] = -sp*st;           R[:, 2, 2] = ct
    return p, R


def _calcSectionFK(_k0: np.ndarray, _k1: np.ndarray, _s: np.ndarray):
    '''Tip position and rotation of one section with complex curvature k0 + k1*sigma (1/mm), split into
        N_SUB constant-curvature pieces.'''
    p = np.zeros((len(_s), 3))
    R = np.broadcast_to(np.eye(3), (len(_s), 3, 3))
    ds = _s/N_SUB
    for i in range(N_SUB):
        kk = _k0 + _k1*(i + 0.5)/N_SUB
        pi, Ri = _calcArcs(kk.real, kk.imag, ds)
        p = p + np.einsum("mij,mj->mi", R, pi)
        R = R @ Ri
    return p, R


# Parameter layout: complex terms as (re, im) pairs, then the two length gains.
TERMS = ["c0_1", "h0_1", "c1_1", "h1_1", "c0_2", "h0_2", "c1_2", "h1_2"]
EXTRA_TERMS = ["m_1", "m_2", "z_1", "z_2"]        # bend-size (|v|^2) and compression (s - sectL) gain terms
ALL_TERMS = TERMS + EXTRA_TERMS
REAL_TERMS = ["lam_1", "lam_2"]


def getPCCParams(_lam: float = 1.0) -> np.ndarray:
    '''Full parameter vector that reproduces PCC (with length gain _lam).'''
    p = np.zeros(2*len(ALL_TERMS) + len(REAL_TERMS))
    for name in ("c0_1", "c0_2"):
        p[2*ALL_TERMS.index(name)] = 1.0
    p[-2:] = _lam
    return p


def unpackParams(_params: np.ndarray) -> dict:
    '''Full parameter vector -> {term name: value} (complex terms as complex numbers, length gains real).'''
    out = {name: _params[2*i] + 1j*_params[2*i + 1] for i, name in enumerate(ALL_TERMS)}
    out.update({name: _params[2*len(ALL_TERMS) + i] for i, name in enumerate(REAL_TERMS)})
    return out


def packParams(_constants: dict) -> np.ndarray:
    '''Constants as saved in polyConstants.json -> full parameter vector.'''
    p = getPCCParams()
    for i, name in enumerate(ALL_TERMS):
        p[2*i], p[2*i + 1] = _constants[name]
    p[-2], p[-1] = _constants["lam_1"], _constants["lam_2"]
    return p


def calcPolyShape(_params: np.ndarray, _q: np.ndarray, _gains=None) -> dict:
    '''Both sections' Poly shape for motor angles _q: arc lengths s1', s2',
        overall bends A1, A2 (rad, complex = direction + size), curvature
        slopes B1, B2, and the calibrated-PCC bends v1, v2 they grew from.
        _gains = per-tendon cable-travel gains A..F (None = 1).'''
    terms = unpackParams(_params)
    s1, v1, s2, v2 = calcSectionsFromMotors(_q if _gains is None else _q*_gains)
    g1 = terms["c0_1"] + terms["m_1"]*abs(v1)**2 + terms["z_1"]*(s1 - SECT_L)/SECT_L
    g2 = terms["c0_2"] + terms["m_2"]*abs(v2)**2 + terms["z_2"]*(s2 - SECT_L)/SECT_L
    return {"A1": g1*v1 + terms["h0_1"]*v2, "B1": terms["c1_1"]*v1 + terms["h1_1"]*v2,
            "A2": g2*v2 + terms["h0_2"]*v1, "B2": terms["c1_2"]*v2 + terms["h1_2"]*v1,
            "s1": SECT_L + terms["lam_1"]*(s1 - SECT_L), "s2": SECT_L + terms["lam_2"]*(s2 - SECT_L),
            "v1": v1, "v2": v2}


def calcPolyPose(_params: np.ndarray, _q: np.ndarray, _gains=None):
    '''Predicted tool tip [sample][xyz] and tool axis [sample][xyz].'''
    shape = calcPolyShape(_params, _q, _gains)
    A1, B1, A2, B2, s1p, s2p = shape["A1"], shape["B1"], shape["A2"], shape["B2"], shape["s1"], shape["s2"]
    # kappa(sigma) = (A + (sigma - 1/2)*B)/s' = k0 + k1*sigma
    p1, R1 = _calcSectionFK((A1 - B1/2)/s1p, B1/s1p, s1p)
    p2, R2 = _calcSectionFK((A2 - B2/2)/s2p, B2/s2p, s2p)
    axis = (R1 @ R2)[:, :, 2]
    return p1 + np.einsum("mij,mj->mi", R1, p2) + TOOL*axis, axis


def calcPolyTip(_params: np.ndarray, _q: np.ndarray, _gains=None) -> np.ndarray:
    '''Predicted tool-tip positions [sample][xyz] for motor angles _q
        (getPCCParams() and _gains = None give PCC's own prediction).'''
    return calcPolyPose(_params, _q, _gains)[0]


# ---- Fitted constants ----

def loadConstants(_path: str = CONSTANTS_FILE) -> dict:
    '''The constants saved by Results/calibration/polyFit.py:
          gains  - tendon gains A..F
          pcc    - full parameter vector for cable-compensated PCC (length gain only)
          poly   - full parameter vector for the Poly model
          lag    - motor -> camera delay used in the fit (samples)'''
    with open(_path) as f:
        saved = json.load(f)
    return {"gains": np.array(saved["tendon_gains_ABCDEF"]),
            "pcc": getPCCParams(saved["pcc_calibrated"]["lam"]),
            "poly": packParams(saved["poly"]["constants"]),
            "lag": saved["lag_samples"]}
