'''Figures for the oral presentation and thesis (PCC results), drawn from the logs and from
pccResults.json (written by pccResults.py, which this script runs first if the file is missing).

Run from anywhere:
    python Results/analysis/presentationFigures.py
PNGs are written to Results/analysis/output/figures/.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Results.logReader import loadTest, parseLog, getLogPath
from Results.analysis import analyse as A
from Results.analysis import pccResults as P

warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # quiet when Arial is missing (falls back)
FIG_DIR = P.OUT_DIR / "figures"

# ---- Style ----
BLUE, BLUE_DARK, BLUE_LIGHT = "#2A78D6", "#104281", "#9EC5F2"
INK, INK2, MUTED, GRID, AXIS, TINT = "#0B0B0B", "#52514E", "#898781", "#E1E0D9", "#C3C2B7", "#F2F2EF"
DESIRED = "#B9B8B2"
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz
MAIN_TYPES = ["S-shape circle", "Figure 8", "Helix", "C-shape circle", "Step-return (C)"]


def setStyle():
    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 13,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.labelsize": 14,
        "axes.titlesize": 14, "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "legend.frameon": False, "legend.fontsize": 12.5, "legend.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 250})


def save(_fig, _name: str) -> pathlib.Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{_name}.png"
    _fig.savefig(path)
    plt.close(_fig)
    print(f"  {path.name}")
    return path


def loadResults() -> dict:
    path = P.OUT_DIR / "pccResults.json"
    if not path.exists():
        P.main()
    with open(path) as f:
        return json.load(f)


def cleanRuns(_res: dict, _type: str) -> list[dict]:
    return [r for r in _res["runs"] if r["class"] == "clean" and r["type"] == _type and "rmse" in r]


def homeRefPaths(_test: int, _ctrl: str = "PCC"):
    '''t, desired xyz and home-referenced camera xyz (NaN where camera-gated) for t >= 0.'''
    d, run, ok, dt, curh, home = A._prepareTest(_test, _ctrl)
    cur = np.where(ok[:, None], curh, np.nan)
    return d["t"][run], d["des"][run, :3], cur, dt


# ---- Figures ----

def plotRunsByType(_res: dict):
    '''Every clean run's RMSE, one row per trajectory type, with the median marked.'''
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.93, bottom=0.17)
    rng = np.random.default_rng(1)
    for i, t in enumerate(MAIN_TYPES):
        rm = np.array([r["rmse"] for r in cleanRuns(_res, t)])
        y = len(MAIN_TYPES) - 1 - i
        ax.scatter(rm, y + rng.uniform(-0.17, 0.17, rm.size), s=22, color=BLUE, alpha=0.55, lw=0)
        med = np.median(rm)
        ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, lw=2.4)
        ax.text(med, y + 0.38, f"{med:.0f}", ha="center", va="bottom", fontsize=11.5, color=INK)
    ax.set_yticks(range(len(MAIN_TYPES)))
    ax.set_yticklabels([f"{t}  (n={len(cleanRuns(_res, t))})" for t in MAIN_TYPES[::-1]], fontsize=12)
    ax.set_ylim(-0.6, len(MAIN_TYPES) - 0.2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Tip RMSE per run (mm)")
    ax.grid(axis="y", visible=False)
    return save(fig, "runs_by_type")


def plotErrorBudget(_res: dict):
    '''Median RMSE as measured, with the time lag removed, and with lag and reach removed.'''
    types = ["S-shape circle", "Figure 8", "Helix", "C-shape circle"]
    steps = [("z_datum", "As measured", BLUE_DARK), ("lag", "Time lag removed", BLUE), ("gain", "Lag and reach removed", BLUE_LIGHT)]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.12)
    w = 0.26
    for j, (k, lab, col) in enumerate(steps):
        vals = [_res["by_type"][t]["budget_median"][k] for t in types]
        x = np.arange(len(types)) + (j - 1)*w
        ax.bar(x, vals, w*0.92, color=col, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\nreach {_res['by_type'][t]['reach_median']:.2f}" for t in types], fontsize=11.5)
    ax.set_ylabel("Median RMSE (mm)")
    ax.set_ylim(0, 65)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.25), fontsize=11.5, columnspacing=1.0, handlelength=1.2)
    return save(fig, "error_budget")


def pickTypicalRun(_res: dict, _type: str = "S-shape circle") -> int:
    '''The clean run of a type whose RMSE is closest to that type's median (full camera coverage).'''
    runs = [r for r in cleanRuns(_res, _type) if r["coverage"] > 0.99]
    med = _res["by_type"][_type]["rmse_median"]
    return min(runs, key=lambda r: abs(r["rmse"] - med))["test"]


def plotExampleRun(_test: int, _rmse: float):
    '''Desired vs measured tip path of one run, top and side views.'''
    t, des, cur, dt = homeRefPaths(_test)
    zDes = float(np.mean(des[:, 2]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.16)
    for a, (i, j), lab in ((a1, (0, 1), "Top view (X–Y)"), (a2, (0, 2), "Side view (X–Z)")):
        a.plot(des[:, i], des[:, j], color=DESIRED, lw=6, label="Desired")
        a.plot(cur[:, i], cur[:, j], color=BLUE, lw=1.2, label="Measured")
        a.set_title(lab, pad=8)
        a.set_xlabel("X (mm)")
    a1.set_aspect("equal"); a1.set_xlim(-130, 130); a1.set_ylim(-130, 130); a1.set_ylabel("Y (mm)")
    a1.set_xticks([-100, 0, 100]); a1.set_yticks([-100, 0, 100])
    a2.set_xlim(-130, 130); a2.set_ylim(zDes - 40, zDes + 60); a2.set_ylabel("Z (mm)"); a2.set_xticks([-100, 0, 100])
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.0))
    fig.text(0.98, 0.93, f"Test {_test}: RMSE {_rmse:.0f} mm", ha="right", fontsize=11.5, color=INK2)
    r = np.hypot(cur[:, 0], cur[:, 1])
    print(f"  example Test_{_test}: desired r {np.mean(np.hypot(des[:, 0], des[:, 1])):.0f} mm, measured r {np.nanmean(r):.0f} mm, "
          f"z offset {np.nanmean(cur[:, 2]) - zDes:+.0f} mm")
    return save(fig, "example_run")


def plotRepeatability(_res: dict):
    '''Error traces of the largest group of identical runs, every lap overlaid.'''
    g = _res["repeats"][0]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.17)
    period = None
    for test in g["tests"]:
        t, des, cur, dt = homeRefPaths(test)
        if period is None:
            period = P.loadMeta(test, "PCC")["taskTime"]/P.loadMeta(test, "PCC")["movementFreq"]
        e = np.linalg.norm(des - cur, axis=1)
        for k in range(int(round(t[-1]/period))):
            m = (t >= k*period) & (t < (k + 1)*period)
            ax.plot(t[m] - k*period, e[m], color=BLUE, lw=0.9, alpha=0.3)
    ax.set_xlim(0, period); ax.set_ylim(0, 100)
    ax.set_xlabel("Time within lap (s)"); ax.set_ylabel("Tip error (mm)")
    ax.set_title(f"{g['settings'].split(',')[0].strip('(').strip(chr(39))}: {g['n']} identical runs, every lap overlaid", fontsize=13)
    ax.text(period*0.99, 95, f"RMSE {g['rmse_min']:.0f}–{g['rmse_max']:.0f} mm (median {g['rmse_median']:.0f})",
            ha="right", va="top", fontsize=11.5, color=INK2)
    return save(fig, "repeatability")


def speedGroups(_res: dict) -> dict:
    '''Paths run at two or more speeds (same settings apart from movementFreq): median RMSE, lag and
        speed per speed setting.'''
    groups = {}
    for r in _res["runs"]:
        if r["class"] != "clean" or "rmse" not in r or not isinstance(r["movementFreq"], (int, float)):
            continue
        k = P.settingsKey(r)
        groups.setdefault(k[:1] + k[2:], {}).setdefault(r["movementFreq"], []).append(r)
    out = {}
    for k, byMf in groups.items():
        if len(byMf) >= 2:
            name = f"{k[0]}, r{k[1]} z{k[2]}" + (f"→{k[3]}" if k[3] != k[2] else "")
            out[name] = [(np.median([r["speed"] for r in v]), np.median([r["rmse"] for r in v]),
                          np.median([r["lag_s"] for r in v]), len(v)) for mf, v in sorted(byMf.items())]
    return out


def plotSpeed(_res: dict):
    '''RMSE and best-fit time lag against mean tip speed for paths run at several speeds.'''
    groups = speedGroups(_res)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.16)
    shades = ["#2A78D6", "#104281", "#5598E7", "#1C5CAB", "#7FB0EE", "#0B2F5E"]
    marks = ["o", "s", "^", "D", "v", "P"]
    for (name, pts), col, mk in zip(groups.items(), shades, marks):
        sp, rm, lg, n = np.array(pts).T
        a1.plot(sp, rm, marker=mk, color=col, lw=1.5, ms=7, mec="white", mew=1.2, label=name)
        a2.plot(sp, lg, marker=mk, color=col, lw=1.5, ms=7, mec="white", mew=1.2)
        print(f"  speed {name}: speed {np.round(sp)}, RMSE {np.round(rm, 1)}, lag {np.round(lg, 2)}, runs {n.astype(int)}")
    a1.set_xlabel("Mean tip speed (mm/s)"); a1.set_ylabel("Median RMSE (mm)"); a1.set_ylim(0, 80)
    a2.set_xlabel("Mean tip speed (mm/s)"); a2.set_ylabel("Best-fit time lag (s)"); a2.set_ylim(0, 0.6)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.01), fontsize=10.5, columnspacing=1.0)
    return save(fig, "speed")


def plotLoopTiming(_res: dict):
    '''Distribution of Pi control-step durations over every run that ran to the end.'''
    steps = []
    for r in _res["runs"]:
        if r["class"] in ("stopped early", "motor trip"):
            continue
        t, _ = parseLog(getLogPath(r["test"], "PCC", "desiredGlob"))
        t = np.array(t)
        steps.append(np.diff(t[t >= 0])*1000)
    a = np.concatenate(steps)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.17)
    bins = np.arange(35, 70.5, 0.5)
    h, e = np.histogram(np.clip(a, 35, 70), bins=bins)
    ax.stairs(100*h/a.size, e, color=BLUE, lw=1.8, fill=False)
    ax.stairs(100*h/a.size, e, color=BLUE, alpha=0.12, fill=True)
    ax.axvline(40, color=INK, lw=1.2)
    ax.text(41.5, ax.get_ylim()[1]*0.9, "40 ms target (25 Hz)", fontsize=12, color=INK, va="top")
    ax.set_xlim(35, 70)
    ax.text(69, ax.get_ylim()[1]*0.6, f"{100*np.mean(a > 1000*P.OVERRUN_S):.0f}% of steps late\n(steps over 70 ms counted at 70)", ha="right", fontsize=11.5, color=INK2)
    ax.set_xlabel("Pi control-step duration (ms)"); ax.set_ylabel("Share of steps (%)")
    print(f"  loop: {a.size} steps, median {np.median(a):.1f} ms, p95 {np.percentile(a, 95):.1f} ms, "
          f"{100*np.mean(a > 1000*P.OVERRUN_S):.0f}% late, max {a.max():.0f} ms")
    return save(fig, "loop_timing")


def plotActuatorTracking(_test: int = 253):
    '''Commanded vs measured angle of motor B, and the tracking error of all six motors.'''
    td, dm = parseLog(getLogPath(_test, "PCC", "desiredEgo"))
    tc, cm = parseLog(getLogPath(_test, "PCC", "currentEgo"))
    n = min(len(td), len(tc))
    t = np.array(td[:n]); dm = np.array(dm).T[:n]; cm = np.array(cm).T[:n]
    run = t >= -2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.1), sharex=True, gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.14})
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.17)
    a1.plot(t[run], dm[run, 1], color=MUTED, lw=2.0, ls=(0, (4, 2)), label="Commanded (from model)")
    a1.plot(t[run], cm[run, 1], color=INK, lw=1.6, label="Measured (encoder)")
    a1.set_ylabel("Motor B (rad)", fontsize=12.5)
    a1.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.32), fontsize=12)
    err = dm - cm
    for i in range(6):
        a2.plot(t[run], err[run, i], color=INK2, lw=0.9, alpha=0.55)
    mx = np.abs(err[t >= 0]).max()
    a2.axhspan(-mx, mx, color=TINT, zorder=0, lw=0)
    a2.set_ylabel("Error (rad)", fontsize=12.5); a2.set_xlabel("Time (s)"); a2.set_ylim(-0.75, 1.1)
    a2.text(29.5, 0.8, f"all 6 motors within ±{mx:.2f} rad", ha="right", va="center", fontsize=11.5, color=INK2)
    a1.set_xlim(-2, 30)
    print(f"  actuator Test_{_test}: max motor tracking error {mx:.2f} rad")
    return save(fig, "actuator_tracking")


def plotCurvatureModes():
    '''Concept: curvature along a section for constant (PCC), affine and quadratic polynomial
        curvature with the same total bend, and the shapes they produce.'''
    s = np.linspace(0, 1, 200)
    L, theta = 145.0, np.radians(60)
    profiles = [("Constant (PCC): κ = θ₀", np.full_like(s, 1.0), BLUE),
                ("Affine: κ = θ₀ + θ₁σ", 0.4 + 1.2*s, "#1BAF7A"),
                ("Quadratic: κ = θ₀ + θ₁σ + θ₂σ²", 0.2 + 0.4*s + 2.4*s**2, "#EB6834")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.24)
    for lab, k, col in profiles:
        k = k/_trapz(k, s)*theta/L*1000          # same total bend for each profile (1/m)
        a1.plot(s, k, color=col, lw=2.4, label=lab)
        ang = np.concatenate([[0], np.cumsum((k[1:] + k[:-1])/2*np.diff(s)*L/1000)])
        x = np.concatenate([[0], np.cumsum(np.sin(ang[1:])*np.diff(s)*L)])
        z = np.concatenate([[0], np.cumsum(np.cos(ang[1:])*np.diff(s)*L)])
        a2.plot(x, z, color=col, lw=2.4)
        a2.plot(x[-1], z[-1], "o", color=col, ms=6)
    a1.set_xlabel("Position along section, σ"); a1.set_ylabel("Curvature κ (1/m)")
    a1.set_xticks([0, 0.5, 1]); a1.set_xticklabels(["0\nbase", "½", "1\nend"]); a1.set_ylim(0, 20)
    a2.set_aspect("equal"); a2.set_xlim(-10, 90); a2.set_ylim(0, 150)
    a2.set_xlabel("x (mm)"); a2.set_ylabel("z (mm)")
    a2.text(88, 8, "same total bend,\ndifferent tip", ha="right", fontsize=11.5, color=INK2)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0), fontsize=11, columnspacing=1.0, handlelength=1.4)
    return save(fig, "curvature_modes")


# Published PCC tip errors used for the comparison (as % of robot length, my calculation from each paper).
#   Zhai 2025 (Cyborg Bionic Syst. 6:0339): PCC baseline Table 2, RMSE x 1.6, y 2.3 mm -> 2.8 mm over 270 mm; camera closed loop, T = 40 s
#   Gong 2025 (Sens. Actuators A 391:116669): mean tip offset 1.7 % of body length, 8 static poses, open loop
#   Peng 2024 (IEEE RA-L 9(4)): "Conv" PCC estimate from tendon lengths, RMSE ~60 mm over 450 mm (Sec. IV-E)
#   Qi 2024 (IEEE/ASME TMECH, doi 10.1109/TMECH.2024.3402609): CC baseline, open loop, quasi-static,
#       mean 3.52 cm (two-section line) to 5.92 cm (two-section circle) over 2 x 9.30 cm backbone
LITERATURE = [
    {"study": "Zhai et al. 2025", "setup": "planar · 3 sections · ≈7 mm/s · camera loop", "lo": 100*2.8/270, "hi": 100*2.8/270, "label": "≈1%  (2.8 mm RMSE)", "kind": "camera"},
    {"study": "Gong et al. 2025", "setup": "planar · 2 sections · static · no tip feedback", "lo": 1.7, "hi": 1.7, "label": "1.7%  (mean)", "kind": "none"},
    {"study": "Peng et al. 2024", "setup": "3D · 3 sections · estimate from tendon lengths", "lo": 100*60/450, "hi": 100*60/450, "label": "≈13%  (≈60 mm RMSE)", "kind": "none"},
    {"study": "Qi et al. 2024", "setup": "3D · 2 soft sections · slow · open loop", "lo": 100*35.2/186, "hi": 100*59.2/186, "label": "19–32%  (mean 35–59 mm)", "kind": "none"},
]


def plotLiteratureComparison(_res: dict):
    '''PCC tip error as % of robot length: published results against this work's range of
        per-path median RMSEs.'''
    meds = [_res["by_type"][t]["rmse_median"] for t in MAIN_TYPES]
    L = _res["robot_length_mm"]
    mine = {"study": "This work", "setup": "3D · 2 soft sections · ≤130 mm/s · encoders only",
            "lo": 100*min(meds)/L, "hi": 100*max(meds)/L, "label": f"{100*min(meds)/L:.0f}–{100*max(meds)/L:.0f}%  ({min(meds):.0f}–{max(meds):.0f} mm RMSE)", "kind": "this"}
    rows = sorted(LITERATURE + [mine], key=lambda r: r["lo"])[::-1]
    col = {"camera": INK2, "none": MUTED, "this": BLUE}
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    fig.subplots_adjust(left=0.41, right=0.98, top=0.86, bottom=0.15)
    ax.axvspan(mine["lo"], mine["hi"], color="#E8F1FB", zorder=0, lw=0)
    for y, r in enumerate(rows):
        c = col[r["kind"]]
        if r["hi"] - r["lo"] > 0.3:
            ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=9, solid_capstyle="round", zorder=3)
        else:
            ax.plot([r["lo"]], [y], "o", color=c, ms=11, zorder=3)
        right = r["hi"] < 25                 # long bars near the right edge are labelled on their left
        ax.text(r["hi"] + 1.0 if right else r["lo"] - 1.0, y, r["label"], va="center", ha="left" if right else "right", fontsize=10.5,
                color=INK if r["kind"] == "this" else INK2, fontweight="bold" if r["kind"] == "this" else "normal")
        tr = ax.get_yaxis_transform()
        ax.text(-0.03, y + 0.13, r["study"], transform=tr, ha="right", va="center", fontsize=12.5,
                color=BLUE if r["kind"] == "this" else INK, fontweight="bold")
        ax.text(-0.03, y - 0.2, r["setup"], transform=tr, ha="right", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Tip error, % of robot length")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, marker="o", lw=0, ms=9), Line2D([], [], color=MUTED, marker="o", lw=0, ms=9)]
    ax.legend(h, ["camera in the control loop", "no tip feedback"], loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=10.5, ncol=3, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.2)
    return save(fig, "literature_comparison")


def main():
    setStyle()
    res = loadResults()
    print(f"Figures -> {FIG_DIR}")
    plotRunsByType(res)
    plotErrorBudget(res)
    ex = pickTypicalRun(res)
    plotExampleRun(ex, next(r["rmse"] for r in res["runs"] if r["test"] == ex))
    plotRepeatability(res)
    plotSpeed(res)
    plotLoopTiming(res)
    plotActuatorTracking()
    plotCurvatureModes()
    plotLiteratureComparison(res)


if __name__ == "__main__":
    main()
