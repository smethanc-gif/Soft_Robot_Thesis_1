'''Fits the polynomial-curvature (Poly) model's constants to logged test data.

Fitted in two stages so the constants stay physical:
  Stage 1 - calibrated PCC: a gain on each tendon's cable travel (A..F) and
            one shared length gain.
  Stage 2 - Poly curvature terms on top, with the tendon gains held fixed
            (fitting both at once lets them trade off into nonsense values).

Forward model, measured motor angles -> tip position:
  1. Cable lengths from the motor angles (same spool/offset constants as
     Core/PCC/constantCurvature.py), each tendon's travel scaled by its gain.
  2. Each section's PCC-equivalent bend vector v_j (rad, direction and
     size) and arc length s_j, from inverting calcEgoTargets' cable equations:
     B, D, F only run through section 1; A, C, E run through both.
  3. Poly curvature along each section, sigma in [0, 1]:
         kappa_j(sigma) = [ (c0_j*v_j + h0_j*v_k) + (sigma - 1/2)*(c1_j*v_j + h1_j*v_k) ] / s_j'
     (k = the other section) - the same linear kx/ky form as
     Core/Poly/polyModel.py. s_j' = sectL + lam_j*(s_j - sectL).
     Variants: gains real (scale only) or complex (scale + rotation of the
     bend vector); optional extra terms let the gain depend on bend size
     (|v|^2) and on compression (s_j - sectL).
  4. Tip = section 1 then section 2 then the tool, composed exactly as
     polyModel.calcPolyPose does (the model itself lives in Core/Poly/polyModel.py).
PCC is the special case c0 = 1, everything else 0, lam = 1 - so the fitted
constants show directly how the real robot differs from PCC.

Run from the repo root:  python Results/calibration/polyFit.py
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
from Core.Poly.polyModel import (SECT_L, N_SUB, ALL_TERMS, REAL_TERMS, TERMS, CONSTANTS_FILE,
                                   calcSectionsFromMotors, getPCCParams, unpackParams, calcPolyShape, calcPolyTip)

TIME_STEP = loadControlSettings()["timeStep"]

# ---- Data sets ----
TRAIN = {"step-return": [191, 203, 204],
         "circle": [192, 193, 195, 196, 197, 199],
         "figure 8": [201, 202, 205, 206, 209],
         "helix": [211, 212, 214, 215, 216]}
VALID = {"circle": [194, 198], "figure 8": [208], "helix": [213, 217]}
STEP_RETURN_STRIDE = 3                  # step-returns are ~7000 samples each - thin them so they don't swamp the rest
POLY_VARIANT = "real+extra"             # stage-2 terms kept (see VARIANTS)
SHAPE_WEIGHT = 1.0                      # mm of tip error per deg of section bend moved away from calibrated PCC

# Every variant uses polyModel's full parameter vector; a variant just picks
# which entries are fitted (the rest stay at their PCC values).
VARIANTS = {                                       # (terms fitted, rotations allowed, one shared length gain)
    "real":          (TERMS, False, True),
    "real+extra":    (ALL_TERMS, False, True),
    "complex":       (TERMS, True, False),
    "complex+extra": (ALL_TERMS, True, False),
}


def _getFreeIndex(_variant: str) -> np.ndarray:
    '''Indices of the full parameter vector that a variant fits (the rest stay at their PCC values).'''
    terms, rotations, shared = VARIANTS[_variant]
    idx = []
    for name in terms:
        i = 2*ALL_TERMS.index(name)
        idx += [i, i + 1] if rotations else [i]
    idx.append(2*len(ALL_TERMS))                   # lam_1
    if not shared:
        idx.append(2*len(ALL_TERMS) + 1)           # lam_2
    return np.array(idx)


def expandParams(_x: np.ndarray, _variant: str) -> np.ndarray:
    '''Fitted values for a variant -> full parameter vector.'''
    full = getPCCParams()
    full[_getFreeIndex(_variant)] = _x
    if VARIANTS[_variant][2]:
        full[-1] = full[-2]                        # shared length gain
    return full


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
    '''Fit data of all tests in {group: [tests]} stacked into (motor angles, tips, test label per sample).'''
    qs, tips, labels = [], [], []
    for group, ts in _tests.items():
        for t in ts:
            q, tip = loadFitData(t, _lag, _stride=STEP_RETURN_STRIDE if group == "step-return" else 1)
            qs.append(q); tips.append(tip); labels += [t]*len(q)
    return np.vstack(qs), np.vstack(tips), np.array(labels)


def calcRmse(_pred: np.ndarray, _meas: np.ndarray) -> float:
    '''RMS tip distance (mm) between predicted and measured [sample][xyz].'''
    return float(np.sqrt(np.mean(np.sum((_pred - _meas)**2, axis=1))))


# ---- Fit / evaluate ----

def findBestLag(_tests: dict, _lags=range(0, 9)):
    '''Motor-to-camera delay (samples) that best lines PCC's prediction up with the camera.'''
    scores = {}
    for L in _lags:
        q, tip, _ = stackTests(_tests, L)
        scores[L] = calcRmse(calcPolyTip(getPCCParams(), q), tip)
    return min(scores, key=scores.get), scores


def _solveLeastSquares(_residual, _x0):
    '''Levenberg-Marquardt, stopping once the RMSE improves by < 1e-6 per step -
        far below the camera's noise, and much faster than the default settings.'''
    return least_squares(_residual, _x0, x_scale="jac", method="lm",
                         ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=400*len(_x0))


def fitTendonGains(_tests: dict, _lag: int, _verbose: bool = True):
    '''Stage 1: PCC with a gain on each tendon's cable travel (A..F) and one
        shared length gain. Returns (gains, full PCC parameter vector with that
        length gain).'''
    q, tip, _ = stackTests(_tests, _lag)
    il = 2*len(ALL_TERMS)

    def makeParams(_lam):
        '''PCC parameter vector with both length gains set to _lam.'''
        p = getPCCParams(); p[il] = p[il + 1] = _lam
        return p
    t0 = time.time()
    res = _solveLeastSquares(lambda x: (calcPolyTip(makeParams(x[6]), q, x[:6]) - tip).ravel(), np.ones(7))
    gains, full = res.x[:6], makeParams(res.x[6])
    if _verbose:
        print(f"  tendon gains: 7 constants, {len(q)} samples, {time.time() - t0:.0f} s:"
              f" training RMSE {calcRmse(calcPolyTip(getPCCParams(), q), tip):.1f} (PCC) -> {calcRmse(calcPolyTip(full, q, gains), tip):.1f} mm")
    return gains, full


def fitPolyTerms(_tests: dict, _lag: int, _variant: str = "real", _gains=None, _shapeWeight: float = 0.0, _verbose: bool = True) -> np.ndarray:
    '''Fit a variant's constants (on top of fixed tendon gains, if given);
        returns the full parameter vector.
        shapeWeight > 0 holds each section's overall bend near the calibrated-
        PCC bend: shapeWeight mm of tip error "costs" the same as 1 deg of
        bend change. The camera only sees the tip, so without this the fit is
        free to swap bend between the sections (it straightened section 2).'''
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
    if _verbose:
        print(f"  {_variant}{' + tendon gains' if _gains is not None else ''}{' + shape held' if _shapeWeight > 0 else ''}:"
              f" {len(x0)} constants, {len(q)} samples,"
              f" {time.time() - t0:.0f} s ({res.nfev} evaluations): training RMSE"
              f" {calcRmse(calcPolyTip(getPCCParams(), q), tip):.1f} (PCC) -> {calcRmse(calcPolyTip(full, q, _gains), tip):.1f} mm")
    return full


def evaluateModel(_params: np.ndarray, _tests: dict, _lag: int, _gains=None) -> dict:
    '''Per-test tip prediction RMSE (mm): plain PCC vs the given model.'''
    out = {}
    for group, ts in _tests.items():
        for t in ts:
            q, tip = loadFitData(t, _lag)
            out[t] = (group, calcRmse(calcPolyTip(getPCCParams(), q), tip), calcRmse(calcPolyTip(_params, q, _gains), tip))
    return out


def printScores(_title: str, _scores: dict):
    '''Prints evaluateModel's scores as a table with the mean; returns (mean PCC, mean model) RMSE.'''
    print(f"\n{_title}")
    print("   test  | group        | PCC RMSE | Poly RMSE | change")
    for t, (g, pcc, poly) in _scores.items():
        print(f"   {t:5d} | {g:12s} | {pcc:8.1f} | {poly:9.1f} | {100*(poly/pcc - 1):+5.0f}%")
    pcc = np.mean([v[1] for v in _scores.values()]); poly = np.mean([v[2] for v in _scores.values()])
    print(f"   mean  |              | {pcc:8.1f} | {poly:9.1f} | {100*(poly/pcc - 1):+5.0f}%")
    return pcc, poly


def printConstants(_params: np.ndarray):
    '''Prints a full parameter vector's constants (complex ones as size and angle).'''
    terms = unpackParams(_params)
    print("Fitted constants (PCC: c0 = 1 at 0 deg, everything else 0, lam = 1):")
    for name, v in terms.items():
        if isinstance(v, complex):
            if abs(v) > 0:
                print(f"   {name:6s} = {abs(v):6.3f} at {np.degrees(np.angle(v)):+6.1f} deg")
        else:
            print(f"   {name:6s} = {v:6.3f}")


def checkInversion(_test: int = 196):
    '''PCC's own commands, pushed back through calcSectionsFromMotors + calcPolyTip,
        must land on PCC's own calculated tip. main.py logs the calculated
        tip before that step's IK runs, so it's compared with the next row.'''
    d = loadTest(_test)
    r = np.flatnonzero(d["t"] >= 0)[:-1]
    pred = calcPolyTip(getPCCParams(), d["dm"][r])
    err = np.linalg.norm(pred - d["cal"][r + 1, :3], axis=1)
    print(f"Inversion check on Test {_test}: PCC commands -> tip vs PCC's calculated tip: max {err.max():.2f} mm, mean {err.mean():.2f} mm")


def saveConstants(_gains, _pccCal, _poly, _lag: int, _summary: dict):
    '''Tendon gains + length gain (calibrated PCC) and the Poly constants
        fitted on top of them, with the scores that chose them.'''
    out = {"model": "fitted to measured motors -> camera tip (Results/calibration/polyFit.py)",
           "lag_samples": _lag, "n_sub": N_SUB, "train_tests": TRAIN, "valid_tests": VALID,
           "tendon_gains_ABCDEF": [float(g) for g in _gains],
           "pcc_calibrated": {"lam": float(_pccCal[2*len(ALL_TERMS)])},
           "poly": {"variant": POLY_VARIANT, "shape_weight_mm_per_deg": SHAPE_WEIGHT,
                    "constants": {k: ([v.real, v.imag] if isinstance(v, complex) else v) for k, v in unpackParams(_poly).items()}},
           "mean_tip_rmse_mm": _summary}
    with open(CONSTANTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved constants to {CONSTANTS_FILE}")


def main():
    '''Both fits (validation runs held out, and all figure 8s held out), the comparison tables,
        the shape check, and saves the constants used by the controllers.'''
    checkInversion()

    lag, scores = findBestLag(TRAIN)
    print(f"Motor -> camera delay: {lag} samples ({lag*TIME_STEP:.2f} s); PCC RMSE by lag: "
          + ", ".join(f"{L}:{s:.1f}" for L, s in scores.items()))

    no8 = {g: ts for g, ts in TRAIN.items() if g != "figure 8"}
    all8 = {"figure 8": TRAIN["figure 8"] + VALID["figure 8"]}
    summary, perTest, final = {}, {}, None
    for label, train, test in (("validation (mf 4)", TRAIN, VALID), ("unseen figure 8s", no8, all8)):
        print(f"\n===== Fitting for the {label} check =====")
        # Stage 1: tendon gains with PCC; stage 2: Poly terms on top, gains fixed
        gains, pccCal = fitTendonGains(train, lag)
        models = {"PCC": (getPCCParams(), None), "PCC + tendon gains": (pccCal, gains),
                  "Poly + gains, free shape": (fitPolyTerms(train, lag, POLY_VARIANT, gains), gains),
                  "Poly + gains, shape held": (fitPolyTerms(train, lag, POLY_VARIANT, gains, SHAPE_WEIGHT), gains)}
        for name, (p, g) in models.items():
            sc = evaluateModel(p, test, lag, g)
            summary.setdefault(name, {})[label] = float(np.mean([v[2] for v in sc.values()]))
            perTest.setdefault(label, {})[name] = sc
        if final is None:
            final = (gains, pccCal, models)

    for label, byModel in perTest.items():
        names = list(byModel)
        print(f"\nTip RMSE (mm), {label}:")
        print("   test  | group     | " + " | ".join(f"{n:>22s}" for n in names))
        for t in byModel[names[0]]:
            print(f"   {t:5d} | {byModel[names[0]][t][0]:9s} | " + " | ".join(f"{byModel[n][t][2]:22.1f}" for n in names))

    print("\n=== Summary: mean tip RMSE (mm) ===")
    for name, s in summary.items():
        print(f"   {name:24s} " + " | ".join(f"{k} {v:5.1f}" for k, v in s.items()))

    gains, pccCal, models = final
    print("\nShape check - overall bend of each section (deg, + = along section 1's bend) for the same motor angles:")
    for t in (196, 206, 212):
        q, _ = loadFitData(t, lag)
        q = q[len(q)//3:len(q)//3 + 1]
        u = calcSectionsFromMotors(q)[1]
        u = u/abs(u)
        row = []
        for name, (p, g) in models.items():
            shape = calcPolyShape(p, q, g)
            row.append(f"{name} {np.degrees((shape['A1']/u).real)[0]:+4.0f}/{np.degrees((shape['A2']/u).real)[0]:+4.0f}")
        print(f"   Test {t}: " + " | ".join(row))

    print("\nTendon gains (cable travel vs model), A..F: " + "  ".join(f"{m} {g:.3f}" for m, g in zip("ABCDEF", gains))
          + f" | length gain {pccCal[2*len(ALL_TERMS)]:.3f}")
    poly = models["Poly + gains, shape held"][0]
    printConstants(poly)
    saveConstants(gains, pccCal, poly, lag, summary)


if __name__ == "__main__":
    main()
