'''Fits the polynomial-curvature (Poly) model's constants to logged test data.

Fitted in two stages so the constants stay physical:
  Stage 1 - calibrated PCC: a gain on each tendon's cable travel (A..F) and
            one shared length gain.
  Stage 2 - Poly curvature terms on top, with the tendon gains held fixed
            (fitting both at once lets them trade off into nonsense values).

Forward model, measured motor angles -> tip position:
  1. Cable lengths from the motor angles (same spool/offset constants as
     constantCurve.py), each tendon's travel scaled by its gain.
  2. Each section's PCC-equivalent bend vector v_j (rad, direction and
     size) and arc length s_j, from inverting PCC_Control's cable equations:
     B, D, F only run through section 1; A, C, E run through both.
  3. Poly curvature along each section, sigma in [0, 1]:
         kappa_j(sigma) = [ (c0_j*v_j + h0_j*v_k) + (sigma - 1/2)*(c1_j*v_j + h1_j*v_k) ] / s_j'
     (k = the other section) - the same linear kx/ky form as
     poly.PolynomialCurvature. s_j' = sectL + lam_j*(s_j - sectL).
     Variants: gains real (scale only) or complex (scale + rotation of the
     bend vector); optional extra terms let the gain depend on bend size
     (|v|^2) and on compression (s_j - sectL).
  4. Tip = section 1 then section 2 then the tool, composed exactly as
     poly.PolynomialCurvature.forwardKinematics does.
PCC is the special case c0 = 1, everything else 0, lam = 1 - so the fitted
constants show directly how the real robot differs from PCC.

Run from the repo root:  python results/polyFit.py
'''
import os
import json
import time
import numpy
from scipy.optimize import least_squares

try:
    from . import analyse as A
except ImportError:
    import analyse as A
# The model itself (motors -> section shapes -> Poly tip) lives with the
# controllers so the robot uses exactly what was fitted here.
import PCC_control.polyModel as polyModel
from PCC_control.polyModel import (SECT_L, N_SUB, ALL_TERMS, REAL_TERMS, TERMS, CONSTANTS_FILE,
                                   sectionsFromMotors, pccParams, unpack, polyShape, polyTip)

# ---------------- data sets ----------------
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


def _freeIndex(variant: str) -> numpy.ndarray:
    terms, rotations, shared = VARIANTS[variant]
    idx = []
    for name in terms:
        i = 2*ALL_TERMS.index(name)
        idx += [i, i + 1] if rotations else [i]
    idx.append(2*len(ALL_TERMS))                   # lam_1
    if not shared:
        idx.append(2*len(ALL_TERMS) + 1)           # lam_2
    return numpy.array(idx)


def expand(x: numpy.ndarray, variant: str) -> numpy.ndarray:
    ''' Fitted values for a variant -> full parameter vector. '''
    full = pccParams()
    full[_freeIndex(variant)] = x
    if VARIANTS[variant][2]:
        full[-1] = full[-2]                        # shared length gain
    return full


# ============================================================
# Data
# ============================================================

def testData(test: int, lag: int, ctrl: str = "PCC", stride: int = 1):
    ''' Measured motor angles and the camera tip `lag` samples later
        (home-referenced), task samples only, camera jumps removed. '''
    d = A.loadTest(test, ctrl)
    curh, _ = A.homeRef(d)
    ok = A.cameraGate(d) & (d["t"] >= 0)
    n = len(ok) - lag
    i = numpy.flatnonzero(ok[:n] & ok[lag:])[::stride]
    return d["cm"][i], curh[i + lag]


def stack(tests: dict, lag: int):
    qs, tips, labels = [], [], []
    for group, ts in tests.items():
        for t in ts:
            q, tip = testData(t, lag, stride=STEP_RETURN_STRIDE if group == "step-return" else 1)
            qs.append(q); tips.append(tip); labels += [t]*len(q)
    return numpy.vstack(qs), numpy.vstack(tips), numpy.array(labels)


def rmse(pred: numpy.ndarray, meas: numpy.ndarray) -> float:
    return float(numpy.sqrt(numpy.mean(numpy.sum((pred - meas)**2, axis=1))))


# ============================================================
# Fit / evaluate
# ============================================================

def bestLag(tests: dict, lags=range(0, 9)):
    ''' Motor-to-camera delay (samples) that best lines PCC's prediction up with the camera. '''
    scores = {}
    for L in lags:
        q, tip, _ = stack(tests, L)
        scores[L] = rmse(polyTip(pccParams(), q), tip)
    return min(scores, key=scores.get), scores


def _lsq(residual, x0):
    # Levenberg-Marquardt, stopping once the RMSE improves by < 1e-6 per step -
    # far below the camera's noise, and much faster than the default settings
    return least_squares(residual, x0, x_scale="jac", method="lm",
                         ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=400*len(x0))


def fitTendonGains(tests: dict, lag: int, verbose: bool = True):
    ''' Stage 1: PCC with a gain on each tendon's cable travel (A..F) and one
        shared length gain. Returns (gains, full PCC parameter vector with that
        length gain). '''
    q, tip, _ = stack(tests, lag)
    il = 2*len(ALL_TERMS)

    def params(lam):
        p = pccParams(); p[il] = p[il + 1] = lam
        return p
    t0 = time.time()
    res = _lsq(lambda x: (polyTip(params(x[6]), q, x[:6]) - tip).ravel(), numpy.ones(7))
    gains, full = res.x[:6], params(res.x[6])
    if verbose:
        print(f"  tendon gains: 7 constants, {len(q)} samples, {time.time() - t0:.0f} s:"
              f" training RMSE {rmse(polyTip(pccParams(), q), tip):.1f} (PCC) -> {rmse(polyTip(full, q, gains), tip):.1f} mm")
    return gains, full


def fit(tests: dict, lag: int, variant: str = "real", gains=None, shapeWeight: float = 0.0,
        verbose: bool = True) -> numpy.ndarray:
    ''' Fit a variant's constants (on top of fixed tendon gains, if given);
        returns the full parameter vector.
        shapeWeight > 0 holds each section's overall bend near the calibrated-
        PCC bend: shapeWeight mm of tip error "costs" the same as 1 deg of
        bend change. The camera only sees the tip, so without this the fit is
        free to swap bend between the sections (it straightened section 2). '''
    q, tip, _ = stack(tests, lag)
    x0 = pccParams()[_freeIndex(variant)]
    w = shapeWeight/numpy.radians(1.0)/numpy.sqrt(3)     # per sample: 4 bend components vs 3 tip components

    def residual(x):
        full = expand(x, variant)
        r = (polyTip(full, q, gains) - tip).ravel()
        if shapeWeight > 0:
            S = polyShape(full, q, gains)
            d1, d2 = S["A1"] - S["v1"], S["A2"] - S["v2"]
            r = numpy.r_[r, w*d1.real, w*d1.imag, w*d2.real, w*d2.imag]
        return r
    t0 = time.time()
    res = _lsq(residual, x0)
    full = expand(res.x, variant)
    if verbose:
        print(f"  {variant}{' + tendon gains' if gains is not None else ''}{' + shape held' if shapeWeight > 0 else ''}:"
              f" {len(x0)} constants, {len(q)} samples,"
              f" {time.time() - t0:.0f} s ({res.nfev} evaluations): training RMSE"
              f" {rmse(polyTip(pccParams(), q), tip):.1f} (PCC) -> {rmse(polyTip(full, q, gains), tip):.1f} mm")
    return full


def evaluate(params: numpy.ndarray, tests: dict, lag: int, gains=None) -> dict:
    ''' Per-test tip prediction RMSE (mm): plain PCC vs the given model. '''
    out = {}
    for group, ts in tests.items():
        for t in ts:
            q, tip = testData(t, lag)
            out[t] = (group, rmse(polyTip(pccParams(), q), tip), rmse(polyTip(params, q, gains), tip))
    return out


def report(title: str, scores: dict):
    print(f"\n{title}")
    print("   test  | group        | PCC RMSE | Poly RMSE | change")
    for t, (g, pcc, poly) in scores.items():
        print(f"   {t:5d} | {g:12s} | {pcc:8.1f} | {poly:9.1f} | {100*(poly/pcc - 1):+5.0f}%")
    pcc = numpy.mean([v[1] for v in scores.values()]); poly = numpy.mean([v[2] for v in scores.values()])
    print(f"   mean  |              | {pcc:8.1f} | {poly:9.1f} | {100*(poly/pcc - 1):+5.0f}%")
    return pcc, poly


def describe(params: numpy.ndarray):
    P = unpack(params)
    print("Fitted constants (PCC: c0 = 1 at 0 deg, everything else 0, lam = 1):")
    for name, v in P.items():
        if isinstance(v, complex):
            if abs(v) > 0:
                print(f"   {name:6s} = {abs(v):6.3f} at {numpy.degrees(numpy.angle(v)):+6.1f} deg")
        else:
            print(f"   {name:6s} = {v:6.3f}")


def checkInversion(test: int = 196):
    ''' PCC's own commands, pushed back through sectionsFromMotors + polyTip,
        must land on PCC's own calculated tip. main.py logs the calculated
        tip before that step's IK runs, so it's compared with the next row.'''
    d = A.loadTest(test)
    r = numpy.flatnonzero(d["t"] >= 0)[:-1]
    pred = polyTip(pccParams(), d["dm"][r])
    err = numpy.linalg.norm(pred - d["cal"][r + 1, :3], axis=1)
    print(f"Inversion check on Test {test}: PCC commands -> tip vs PCC's calculated tip: max {err.max():.2f} mm, mean {err.mean():.2f} mm")


def saveConstants(gains, pccCal, poly, lag: int, summary: dict):
    ''' Tendon gains + length gain (calibrated PCC) and the Poly constants
        fitted on top of them, with the scores that chose them. '''
    out = {"model": "fitted to measured motors -> camera tip (results/polyFit.py)",
           "lag_samples": lag, "n_sub": N_SUB, "train_tests": TRAIN, "valid_tests": VALID,
           "tendon_gains_ABCDEF": [float(g) for g in gains],
           "pcc_calibrated": {"lam": float(pccCal[2*len(ALL_TERMS)])},
           "poly": {"variant": POLY_VARIANT, "shape_weight_mm_per_deg": SHAPE_WEIGHT,
                    "constants": {k: ([v.real, v.imag] if isinstance(v, complex) else v) for k, v in unpack(poly).items()}},
           "mean_tip_rmse_mm": summary}
    with open(CONSTANTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved constants to {CONSTANTS_FILE}")


def main():
    checkInversion()

    lag, scores = bestLag(TRAIN)
    print(f"Motor -> camera delay: {lag} samples ({lag*0.04:.2f} s); PCC RMSE by lag: "
          + ", ".join(f"{L}:{s:.1f}" for L, s in scores.items()))

    no8 = {g: ts for g, ts in TRAIN.items() if g != "figure 8"}
    all8 = {"figure 8": TRAIN["figure 8"] + VALID["figure 8"]}
    summary, perTest, final = {}, {}, None
    for label, train, test in (("validation (mf 4)", TRAIN, VALID), ("unseen figure 8s", no8, all8)):
        print(f"\n===== Fitting for the {label} check =====")
        # Stage 1: tendon gains with PCC; stage 2: Poly terms on top, gains fixed
        gains, pccCal = fitTendonGains(train, lag)
        models = {"PCC": (pccParams(), None), "PCC + tendon gains": (pccCal, gains),
                  "Poly + gains, free shape": (fit(train, lag, POLY_VARIANT, gains), gains),
                  "Poly + gains, shape held": (fit(train, lag, POLY_VARIANT, gains, SHAPE_WEIGHT), gains)}
        for name, (p, g) in models.items():
            sc = evaluate(p, test, lag, g)
            summary.setdefault(name, {})[label] = float(numpy.mean([v[2] for v in sc.values()]))
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
        q, _ = testData(t, lag)
        q = q[len(q)//3:len(q)//3 + 1]
        u = sectionsFromMotors(q)[1]
        u = u/abs(u)
        row = []
        for name, (p, g) in models.items():
            S = polyShape(p, q, g)
            row.append(f"{name} {numpy.degrees((S['A1']/u).real)[0]:+4.0f}/{numpy.degrees((S['A2']/u).real)[0]:+4.0f}")
        print(f"   Test {t}: " + " | ".join(row))

    print("\nTendon gains (cable travel vs model), A..F: " + "  ".join(f"{m} {g:.3f}" for m, g in zip("ABCDEF", gains))
          + f" | length gain {pccCal[2*len(ALL_TERMS)]:.3f}")
    poly = models["Poly + gains, shape held"][0]
    describe(poly)
    saveConstants(gains, pccCal, poly, lag, summary)


if __name__ == "__main__":
    main()
