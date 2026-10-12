'''Fits the polynomial-curvature (Poly) model's constants to logged test data.

The cable compensation - the tendon gains A..F ("tendon_gains_ABCDEF") in
Core/Poly/polyConstants.json - is calibrated by hand on the robot and is never
changed here: it is read from the file and held fixed, and only the Poly
curvature terms are fitted on top of it. There is no length gain: the section
arc lengths are used as PCC gives them.

Forward model, measured motor angles -> tip position:
  1. Cable lengths from the motor angles (same spool/offset constants as
     Core/PCC/constantCurvature.py), each tendon's travel scaled by its gain.
  2. Each section's PCC-equivalent bend vector v_j (rad, direction and
     size) and arc length s_j, from inverting calcEgoTargets' cable equations:
     B, D, F only run through section 1; A, C, E run through both.
  3. Poly curvature along each section, sigma in [0, 1]:
         kappa_j(sigma) = [ (c0_j*v_j + h0_j*v_k) + (sigma - 1/2)*(c1_j*v_j + h1_j*v_k) ] / s_j
     (k = the other section) - the same linear kx/ky form as
     Core/Poly/polyModel.py.
     Variants: gains real (scale only) or complex (scale + rotation of the
     bend vector); optional extra terms let the gain depend on bend size
     (|v|^2) and on compression (s_j - sectL).
  4. Tip = section 1 then section 2 then the tool, composed exactly as
     polyModel.calcPolyPose does (the model itself lives in Core/Poly/polyModel.py).
PCC is the special case c0 = 1, everything else 0 - so the fitted
constants show directly how the real robot differs from PCC.

The fitted constants are rounded to SIG_FIGS significant figures before they
are scored and saved, so every RMSE printed is that of the saved constants.

Run:  python Results/calibration/polyFit.py
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import time
import numpy as np
from scipy.optimize import least_squares
from Results.logReader import loadTest
from Results.analysis.analyse import calcHomeRef, getCameraGateMask
from Core.settings import loadControlSettings
# The model itself (motors -> section shapes -> Poly tip) lives with the
# controllers so the robot uses exactly what was fitted here.
import Core.Poly.polyModel as polyModel
from Core.Poly.polyModel import (SECT_L, N_SUB, ALL_TERMS, TERMS, CONSTANTS_FILE,
                                   calcSectionsFromMotors, getPCCParams, unpackParams, calcPolyShape, calcPolyTip)

TIME_STEP = loadControlSettings()["timeStep"]

# ---- Data sets: {group: [(moveMode folder, test number), ...]} ----
# 7.10 session (Testing/Thesis Testing Results.xlsx, sheet 7.10): one run of each circle setting
# trains, the repeats and the figure 8s validate.
TRAIN = {"upright circles": [("PCC", t) for t in (248, 249, 250, 251, 254, 255)]
                            + [("PCC_Cable", t) for t in (42, 44, 47, 48, 49)],
         "tilted circles": [("PCC", t) for t in (259, 260, 261, 268)] + [("PCC_Cable", t) for t in (55, 57)]}
VALID = {"upright circle repeats": [("PCC", t) for t in (253, 256, 267, 274)]
                                   + [("PCC_Cable", t) for t in (43, 45, 46, 50, 51, 52, 53, 59, 66, 67, 68)],
         "tilted circle repeat": [("PCC_Cable", 58)],
         "figure 8 (unseen path)": [("PCC", 278), ("PCC_Cable", 70), ("PCC_Cable", 71), ("PCC_Cable", 77)],
         "earlier session, z270": [("PCC_Cable", t) for t in (34, 35, 36, 37, 38, 39, 40)]}
STEP_RETURN_STRIDE = 3                  # step-returns are ~7000 samples each - thin them so they don't swamp the rest
POLY_VARIANT = "real+extra"             # Poly terms fitted (see VARIANTS)
SHAPE_WEIGHT = 1.0                      # mm of tip error per deg of section bend moved away from calibrated PCC
SIG_FIGS = 3                            # significant figures the fitted constants are rounded to

# Every variant uses polyModel's full parameter vector; a variant just picks
# which entries are fitted (the rest stay at their PCC values).
VARIANTS = {                                       # (terms fitted, rotations allowed)
    "real":          (TERMS, False),
    "real+extra":    (ALL_TERMS, False),
    "complex":       (TERMS, True),
    "complex+extra": (ALL_TERMS, True),
}


def _getFreeIndex(_variant: str) -> np.ndarray:
    '''Indices of the full parameter vector that a variant fits (the rest stay at their PCC values).'''
    terms, rotations = VARIANTS[_variant]
    idx = []
    for name in terms:
        i = 2*ALL_TERMS.index(name)
        idx += [i, i + 1] if rotations else [i]
    return np.array(idx)


def expandParams(_x: np.ndarray, _variant: str) -> np.ndarray:
    '''Fitted values for a variant -> full parameter vector.'''
    full = getPCCParams()
    full[_getFreeIndex(_variant)] = _x
    return full


def roundSigFigs(_x: np.ndarray, _sigFigs: int = SIG_FIGS) -> np.ndarray:
    '''Each value rounded to _sigFigs significant figures (zeros stay zero).'''
    return np.array([float(f"{v:.{_sigFigs}g}") for v in np.atleast_1d(_x)])


def loadCableCompensation() -> tuple[np.ndarray, np.ndarray]:
    '''The hand-calibrated cable compensation from polyConstants.json, held fixed during the fit:
        (tendon gains A..F, PCC parameter vector).'''
    constants = polyModel.loadConstants()
    return constants["gains"], constants["pcc"]


# ---- Data ----

def loadFitData(_test: int, _lag: int, _ctrl: str = "PCC", _stride: int = 1):
    '''Measured motor angles and the camera tip `lag` samples later
        (home-referenced), task samples only, camera jumps removed.'''
    d = loadTest(_test, _ctrl)
    curh, _ = calcHomeRef(d)
    ok = getCameraGateMask(d) & (d["t"] >= 0)
    n = len(ok) - _lag
    i = np.flatnonzero(ok[:n] & ok[_lag:])[::_stride]
    return d["cm"][i], curh[i + _lag]


def stackTests(_tests: dict, _lag: int):
    '''Fit data of all tests in {group: [(ctrl, test), ...]} stacked into (motor angles, tips, test label per sample).'''
    qs, tips, labels = [], [], []
    for group, tests in _tests.items():
        for ctrl, t in tests:
            q, tip = loadFitData(t, _lag, ctrl, _stride=STEP_RETURN_STRIDE if "step-return" in group else 1)
            qs.append(q); tips.append(tip); labels += [f"{ctrl} {t}"]*len(q)
    return np.vstack(qs), np.vstack(tips), np.array(labels)


def calcRmse(_pred: np.ndarray, _meas: np.ndarray) -> float:
    '''RMS tip distance (mm) between predicted and measured [sample][xyz].'''
    return float(np.sqrt(np.mean(np.sum((_pred - _meas)**2, axis=1))))


# ---- Fit / evaluate ----

def findBestLag(_tests: dict, _gains: np.ndarray, _pccParams: np.ndarray, _lags=range(0, 9)):
    '''Motor-to-camera delay (samples) that best lines the cable-compensated PCC prediction up with the camera.'''
    scores = {}
    for L in _lags:
        q, tip, _ = stackTests(_tests, L)
        scores[L] = calcRmse(calcPolyTip(_pccParams, q, _gains), tip)
    return min(scores, key=scores.get), scores


def _solveLeastSquares(_residual, _x0):
    '''Levenberg-Marquardt, stopping once the RMSE improves by < 1e-6 per step -
        far below the camera's noise, and much faster than the default settings.'''
    return least_squares(_residual, _x0, x_scale="jac", method="lm",
                         ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=400*len(_x0))


def fitPolyTerms(_tests: dict, _lag: int, _gains: np.ndarray, _variant: str = POLY_VARIANT, _shapeWeight: float = 0.0, _verbose: bool = True) -> np.ndarray:
    '''Fits a variant's Poly constants on top of the fixed tendon gains; returns the full parameter
        vector with the fitted constants rounded to SIG_FIGS significant figures.
        _shapeWeight > 0 holds each section's overall bend near the calibrated-PCC bend:
        _shapeWeight mm of tip error "costs" the same as 1 deg of bend change. The camera only
        sees the tip, so without this the fit is free to swap bend between the sections.'''
    q, tip, _ = stackTests(_tests, _lag)
    x0 = getPCCParams()[_getFreeIndex(_variant)]
    w = _shapeWeight/np.radians(1.0)/np.sqrt(3)     # per sample: 4 bend components vs 3 tip components

    def calcResidual(_x):
        '''Tip errors (plus the shape-hold penalty when _shapeWeight > 0) for fitted values _x.'''
        full = expandParams(_x, _variant)
        r = (calcPolyTip(full, q, _gains) - tip).ravel()
        if _shapeWeight > 0:
            shape = calcPolyShape(full, q, _gains)
            d1, d2 = shape["A1"] - shape["v1"], shape["A2"] - shape["v2"]
            r = np.r_[r, w*d1.real, w*d1.imag, w*d2.real, w*d2.imag]
        return r
    t0 = time.time()
    res = _solveLeastSquares(calcResidual, x0)
    full = expandParams(res.x, _variant)
    rounded = expandParams(roundSigFigs(res.x), _variant)
    if _verbose:
        print(f"  {_variant}{' + shape held' if _shapeWeight > 0 else ''}: {len(x0)} constants, {len(q)} samples,"
              f" {time.time() - t0:.0f} s ({res.nfev} evaluations): training RMSE {calcRmse(calcPolyTip(full, q, _gains), tip):.2f} mm"
              f" ({calcRmse(calcPolyTip(rounded, q, _gains), tip):.2f} mm rounded to {SIG_FIGS} s.f.)")
    return rounded


def evaluateModel(_params: np.ndarray, _tests: dict, _lag: int, _gains=None) -> dict:
    '''Tip prediction RMSE (mm) of one model on every test: {(ctrl, test): (group, RMSE)}.'''
    out = {}
    for group, tests in _tests.items():
        for ctrl, t in tests:
            q, tip = loadFitData(t, _lag, ctrl)
            out[(ctrl, t)] = (group, calcRmse(calcPolyTip(_params, q, _gains), tip))
    return out


def printConstants(_params: np.ndarray):
    '''Prints a full parameter vector's constants (complex ones as size and angle).'''
    terms = unpackParams(_params)
    print("Fitted constants (PCC: c0 = 1 at 0 deg, everything else 0):")
    for name, v in terms.items():
        if abs(v) > 0:
            print(f"   {name:6s} = {abs(v):.{SIG_FIGS}g} at {np.degrees(np.angle(v)):+6.1f} deg")


def checkInversion(_test: int = 196):
    '''PCC's own commands, pushed back through calcSectionsFromMotors + calcPolyTip,
        must land on PCC's own calculated tip. main.py logs the calculated
        tip before that step's IK runs, so it's compared with the next row.'''
    d = loadTest(_test)
    r = np.flatnonzero(d["t"] >= 0)[:-1]
    pred = calcPolyTip(getPCCParams(), d["dm"][r])
    err = np.linalg.norm(pred - d["cal"][r + 1, :3], axis=1)
    print(f"Inversion check on Test {_test}: PCC commands -> tip vs PCC's calculated tip: max {err.max():.2f} mm, mean {err.mean():.2f} mm")


def saveConstants(_poly: np.ndarray, _lag: int, _summary: dict):
    '''Saves the Poly constants (and the data sets and scores that produced them) to polyConstants.json.
        The hand-calibrated cable compensation ("tendon_gains_ABCDEF") is left exactly as it is.'''
    with open(CONSTANTS_FILE) as f:
        saved = json.load(f)
    cable = list(saved["tendon_gains_ABCDEF"])
    saved.update({"model": "fitted to measured motors -> camera tip (Results/calibration/polyFit.py)",
                  "lag_samples": _lag, "n_sub": N_SUB,
                  "train_tests": {g: [list(x) for x in tests] for g, tests in TRAIN.items()},
                  "valid_tests": {g: [list(x) for x in tests] for g, tests in VALID.items()},
                  "poly": {"variant": POLY_VARIANT, "shape_weight_mm_per_deg": SHAPE_WEIGHT, "sig_figs": SIG_FIGS,
                           "constants": {k: [v.real, v.imag] for k, v in unpackParams(_poly).items()}},
                  "mean_tip_rmse_mm": _summary})
    assert saved["tendon_gains_ABCDEF"] == cable
    with open(CONSTANTS_FILE, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nSaved the Poly constants to {CONSTANTS_FILE} (cable compensation unchanged)")


def main():
    '''Fits the Poly terms on TRAIN with the hand-calibrated cable compensation held fixed, scores
        every model on TRAIN and VALID, runs the shape check, and saves the Poly constants.'''
    checkInversion()

    gains, pccCal = loadCableCompensation()
    print("Cable compensation (held fixed): tendon gains A..F " + "  ".join(f"{m} {g:.3g}" for m, g in zip("ABCDEF", gains)))

    lag, scores = findBestLag(TRAIN, gains, pccCal)
    print(f"Motor -> camera delay: {lag} samples ({lag*TIME_STEP:.2f} s); cable-compensated RMSE by lag: "
          + ", ".join(f"{L}:{s:.1f}" for L, s in scores.items()))

    print(f"\n===== Fitting the Poly terms on {sum(len(v) for v in TRAIN.values())} test(s) =====")
    models = {"PCC": (getPCCParams(), None),
              "PCC + cable compensation": (pccCal, gains),
              "Poly, free shape": (fitPolyTerms(TRAIN, lag, gains), gains),
              "Poly, shape held": (fitPolyTerms(TRAIN, lag, gains, _shapeWeight=SHAPE_WEIGHT), gains)}

    groups = {**{f"train: {g}": tests for g, tests in TRAIN.items()}, **VALID}
    perModel = {name: evaluateModel(p, groups, lag, g) for name, (p, g) in models.items()}
    names = list(models)
    print("\nTip prediction RMSE (mm):")
    print("   group                   | test           | " + " | ".join(f"{n:>24s}" for n in names))
    for key, (group, _) in perModel[names[0]].items():
        print(f"   {group:23s} | {key[0]:9s} {key[1]:4d} | " + " | ".join(f"{perModel[n][key][1]:24.1f}" for n in names))

    summary = {name: {g: float(np.mean([v[1] for v in res.values() if v[0] == g])) for g in groups} for name, res in perModel.items()}
    print("\n=== Summary: mean tip RMSE (mm) ===")
    for name, s in summary.items():
        print(f"   {name:24s} " + " | ".join(f"{g} {v:5.1f}" for g, v in s.items()))

    print("\nShape check - overall bend of each section (deg, + = along section 1's bend) for the same motor angles:")
    for ctrl, t in [tests[0] for tests in TRAIN.values()]:
        q, _ = loadFitData(t, lag, ctrl)
        q = q[len(q)//3:len(q)//3 + 1]
        u = calcSectionsFromMotors(q*gains)[1]
        u = u/abs(u)
        row = []
        for name, (p, g) in models.items():
            shape = calcPolyShape(p, q, g)
            row.append(f"{name} {np.degrees((shape['A1']/u).real)[0]:+4.0f}/{np.degrees((shape['A2']/u).real)[0]:+4.0f}")
        print(f"   {ctrl} {t}: " + " | ".join(row))

    poly = models["Poly, shape held"][0]
    printConstants(poly)
    saveConstants(poly, lag, summary)


if __name__ == "__main__":
    main()
