'''Figures for the thesis results chapter, drawn at the width of one column of the thesis
(3.39 in, two-column A4 layout) so the text stays readable when printed.

Reads pccResults.json (pccResults.py) and methodComparison.json (methodComparison.py), running
either script first if its file is missing. The PCC figures follow the oral presentation's
(presentationFigures.py) layout and colours; the three methods keep one colour and marker each
throughout: PCC blue circles, cable-compensated PCC orange squares, PAC aqua triangles.

Run from anywhere:
    python Results/analysis/thesisFigures.py
PNGs are written to Results/analysis/output/thesisFigures/.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Results.logReader import loadTest, loadMeta
from Results.analysis import pccResults as P
from Results.analysis import methodComparison as M
from Results.analysis import presentationFigures as PF

warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
FIG_DIR = P.OUT_DIR / "thesisFigures"
COL_W = 3.39                                   # in, one column of the thesis
INK, INK2, MUTED, GRID, AXIS, TINT, DESIRED = PF.INK, PF.INK2, PF.MUTED, PF.GRID, PF.AXIS, PF.TINT, PF.DESIRED
COLOURS = {"PCC": "#2A78D6", "PCC_Cable": "#EB6834", "Poly": "#1BAF7A"}
MARKERS = {"PCC": "o", "PCC_Cable": "s", "Poly": "^"}
LABELS = {"PCC": "PCC", "PCC_Cable": "Cable-comp. PCC", "Poly": "PAC"}
MAIN_TYPES = ["S-shape circle", "Figure 8", "Helix", "C-shape circle", "Step-return (C)"]


def setStyle():
    '''Presentation style, scaled down to a printed column.'''
    PF.setStyle()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.2, "axes.linewidth": 0.6,
                         "grid.linewidth": 0.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "savefig.dpi": 300})


def save(_fig, _name: str) -> pathlib.Path:
    '''Writes one figure to FIG_DIR as a PNG.'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{_name}.png"
    _fig.savefig(path)
    plt.close(_fig)
    print(f"  {path.name}")
    return path


def loadJson(_name: str, _make) -> dict:
    '''A results file from P.OUT_DIR, made with _make() first if it is missing.'''
    path = P.OUT_DIR / _name
    if not path.exists():
        _make()
    with open(path) as f:
        return json.load(f)


# ---- PCC baseline ----

def plotPccRunsByType(_res: dict):
    '''RMSE of every clean PCC run, by path, with the median of each path marked.'''
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.2)
    rng = np.random.default_rng(1)
    for i, typ in enumerate(MAIN_TYPES):
        rm = np.array([r["rmse"] for r in PF.cleanRuns(_res, typ)])
        ax.scatter(i + rng.uniform(-0.22, 0.22, rm.size), rm, s=7, color=COLOURS["PCC"], alpha=0.55, lw=0, zorder=3)
        med = np.median(rm)
        ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=1.4, zorder=4)
        ax.text(i + 0.36, med, f"{med:.0f}", va="center", fontsize=7, color=INK)
        ax.text(i, 3, f"n = {rm.size}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(MAIN_TYPES)))
    ax.set_xticklabels(["S-shape\ncircle", "Figure 8", "Helix", "C-shape\ncircle", "Step-return\n(C)"])
    ax.set_xlim(-0.6, len(MAIN_TYPES) - 0.3)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Run RMSE (mm)")
    ax.grid(axis="x", visible=False)
    return save(fig, "pcc_runs_by_type")


def plotPccExampleRun(_test: int, _rmse: float):
    '''Desired and measured tip path of one typical PCC run, top and side views.'''
    t, des, cur, dt = PF.homeRefPaths(_test)
    zDes = float(np.mean(des[:, 2]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.13, right=0.98, top=0.8, bottom=0.2)
    for a, (i, j), lab in ((a1, (0, 1), "Top view (X–Y)"), (a2, (0, 2), "Side view (X–Z)")):
        a.plot(des[:, i], des[:, j], color=DESIRED, lw=3.5, label="Desired")
        a.plot(cur[:, i], cur[:, j], color=COLOURS["PCC"], lw=0.7, label="Measured")
        a.set_title(lab, pad=3)
        a.set_xlabel("X (mm)")
    a1.set_aspect("equal"); a1.set_xlim(-130, 130); a1.set_ylim(-130, 130); a1.set_ylabel("Y (mm)")
    a1.set_xticks([-100, 0, 100]); a1.set_yticks([-100, 0, 100])
    a2.set_xlim(-130, 130); a2.set_ylim(zDes - 40, zDes + 60); a2.set_ylabel("Z (mm)"); a2.set_xticks([-100, 0, 100])
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, bbox_to_anchor=(0.1, 1.0))
    fig.text(0.98, 0.93, f"Test {_test}, RMSE {_rmse:.0f} mm", ha="right", fontsize=7, color=INK2)
    return save(fig, "pcc_example_run")


def plotPccRepeatability(_res: dict):
    '''Error traces of the largest group of identical PCC runs, every lap overlaid in one colour (the
        presentation's format), with the test days the runs came from.'''
    g = _res["repeats"][0]
    days = M.loadTestDays()
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)
    period = None
    for test in g["tests"]:
        t, des, cur, dt = PF.homeRefPaths(test)
        if period is None:
            meta = loadMeta(test, "PCC")
            period = meta["taskTime"]/meta["movementFreq"]
        e = np.linalg.norm(des - cur, axis=1)
        for k in range(int(round(t[-1]/period))):
            m = (t >= k*period) & (t < (k + 1)*period)
            ax.plot(t[m] - k*period, e[m], color=COLOURS["PCC"], lw=0.6, alpha=0.3)
    nDays = len({days.get(f"PCC/{x}") for x in g["tests"]})
    ax.set_xlim(0, period); ax.set_ylim(0, 100)
    ax.set_xlabel("Time within lap (s)"); ax.set_ylabel("Tip error (mm)")
    ax.set_title(f"S-shape circle, r 100 mm, z 300 mm: {g['n']} runs, {nDays} test days", pad=3)
    ax.text(period*0.99, 95, f"Run RMSE {g['rmse_min']:.0f}–{g['rmse_max']:.0f} mm (median {g['rmse_median']:.0f})",
            ha="right", va="top", fontsize=7, color=INK2)
    return save(fig, "pcc_repeatability")


def relabelSpeedGroup(_name: str) -> str:
    '''Shorter legend name for a speedGroups key: figure 8 by its half-width a, helix from top to bottom.'''
    name = _name.replace("S-shape circle", "S-circle").replace("Figure 8, r55", "Figure 8, a100")
    m = __import__("re").match(r"(Helix, r\d+) z(\d+)→(\d+)", name)
    return f"{m.group(1)} z{m.group(3)}→{m.group(2)}" if m else name


def plotPccSpeed(_res: dict):
    '''Median RMSE against mean tip speed, for PCC paths run at several speeds.'''
    groups = PF.speedGroups(_res)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.18)
    shades = ["#2A78D6", "#104281", "#5598E7", "#1C5CAB", "#7FB0EE", "#0B2F5E"]
    marks = ["o", "s", "^", "D", "v", "P"]
    for (name, pts), col, mk in zip(groups.items(), shades, marks):
        sp, rm, lg, n = np.array(pts).T
        ax.plot(sp, rm, marker=mk, color=col, lw=0.9, ms=4, mec="white", mew=0.6, label=relabelSpeedGroup(name))
    ax.set_xlabel("Mean tip speed (mm/s)"); ax.set_ylabel("Median RMSE (mm)"); ax.set_ylim(0, 85)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.01), fontsize=6.3, columnspacing=0.8, handlelength=1.6)
    return save(fig, "pcc_speed")


# Published PCC tip errors (as % of robot length, calculated from each paper's reported values):
#   Zhai 2025: PCC baseline RMSE x 1.6, y 2.3 mm (Table 2, Traj. 1) -> 2.8 mm over a 270 mm robot; planar, camera loop, T = 40 s
#   Gong 2025: mean tip offset 1.7 % of body length over 8 static poses, open loop, planar
#   Peng 2024: conventional estimate from tendon lengths, RMSE about 60 mm over the 450 mm torso (Sec. IV-E)
LITERATURE = [
    # Zhai et al. 2025, Tables 2-3: per-axis RMSE combined as sqrt(x^2 + y^2); path perimeter ≈267 mm, so a 40 s period is ≈7 mm/s
    # and a 5 s period ≈53 mm/s. The 5 s result includes their online Jacobian correction.
    {"study": "(Yujia Zhai, 2025)", "setup": "planar, 40 s period (≈7 mm/s)", "lo": 100*2.8/270, "hi": 100*2.8/270, "label": "≈1% (2.8 mm RMSE)", "kind": "camera"},
    {"study": "(Yujia Zhai, 2025)", "setup": "planar, 5 s period (≈53 mm/s)", "lo": 100*19.3/270, "hi": 100*19.3/270, "label": "≈7% (19 mm RMSE)", "kind": "camera"},
    {"study": "(Gong, et al., 2025)", "setup": "planar, static, open loop", "lo": 1.7, "hi": 1.7, "label": "1.7% (mean)", "kind": "none"},
    # Peng et al. 2024: 3D tracking, tip estimate from tendon lengths (≈60 mm) and from fused IMUs (≈20 mm), 450 mm robot
    {"study": "(Peng, et al., 2024)", "setup": "3D, tip from tendon lengths", "lo": 100*60/450, "hi": 100*60/450, "label": "≈13% (≈60 mm RMSE)", "kind": "none"},
    # Qi et al. 2024: two-section cable-driven silicone arm, 206 mm, open loop; mean error at static points on a circle and a line
    {"study": "(Qi, et al., 2024)", "setup": "3D, soft, PCC baseline", "lo": 100*35.2/206, "hi": 100*59.2/206, "label": "17–29% (mean)", "kind": "none"},
    {"study": "(Qi, et al., 2024)", "setup": "with cable-deformation model", "lo": 100*17.0/206, "hi": 100*37.6/206, "label": "8–18% (mean)", "kind": "none"},
    {"study": "(Peng, et al., 2024)", "setup": "3D, tip from fused IMUs", "lo": 100*20/450, "hi": 100*20/450, "label": "≈4% (≈20 mm RMSE)", "kind": "imu"},
]


def plotLiteratureComparison(_res: dict):
    '''Tip error as % of robot length: published results against the range of this work's PCC
        per-path median RMSEs. Marker shape gives what, if anything, measured the tip in the loop.'''
    meds = [_res["by_type"][t]["rmse_median"] for t in MAIN_TYPES]
    L = _res["robot_length_mm"]
    mine = {"study": "This work (PCC)", "setup": "3D, soft, encoders only", "lo": 100*min(meds)/L, "hi": 100*max(meds)/L,
            "label": f"{100*min(meds)/L:.0f}–{100*max(meds)/L:.0f}% ({min(meds):.0f}–{max(meds):.0f} mm RMSE)", "kind": "this"}
    rows = sorted(LITERATURE + [mine], key=lambda r: r["lo"])[::-1]
    col = {"camera": INK2, "imu": INK2, "none": MUTED, "this": COLOURS["PCC"]}
    mk = {"camera": "o", "imu": "D", "none": "o", "this": "o"}
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    fig.subplots_adjust(left=0.33, right=0.985, top=0.91, bottom=0.12)
    ax.axvspan(mine["lo"], mine["hi"], color="#E8F1FB", zorder=0, lw=0)
    for y, r in enumerate(rows):
        c = col[r["kind"]]
        if r["hi"] - r["lo"] > 0.3:
            ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=5, solid_capstyle="round", zorder=3)
        else:
            ax.plot([r["lo"]], [y], mk[r["kind"]], color=c, ms=5 if mk[r["kind"]] == "o" else 4.2, zorder=3)
        ax.text(r["hi"] + 0.8, y, r["label"], va="center", ha="left", fontsize=6.3, color=INK if r["kind"] == "this" else INK2,
                fontweight="bold" if r["kind"] == "this" else "normal")
        tr = ax.get_yaxis_transform()
        ax.text(-0.03, y + 0.15, r["study"], transform=tr, ha="right", va="center", fontsize=7,
                color=COLOURS["PCC"] if r["kind"] == "this" else INK, fontweight="bold")
        ax.text(-0.03, y - 0.2, r["setup"], transform=tr, ha="right", va="center", fontsize=5.6, color=INK2)
    ax.set_yticks([]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 46); ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Tip error, % of robot length")
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    h = [Line2D([], [], color=INK2, marker="o", lw=0, ms=4), Line2D([], [], color=INK2, marker="D", lw=0, ms=3.5),
         Line2D([], [], color=MUTED, marker="o", lw=0, ms=4)]
    ax.legend(h, ["camera in the loop", "on-board IMUs", "no tip sensing"], loc="lower left", bbox_to_anchor=(-0.46, 1.0), fontsize=6.3,
              ncol=3, columnspacing=0.7, handletextpad=0.15, borderaxespad=0.1)
    return save(fig, "literature_comparison")


# ---- Method comparison ----

def methodLegend(_ax, _loc="upper center", _anchor=(0.5, 1.02), _extra: list = None):
    '''Legend of the three methods (colour + marker), plus any _extra handles.'''
    h = [Line2D([], [], color=COLOURS[c], marker=MARKERS[c], lw=0, ms=4) for c in M.METHODS]
    l = [LABELS[c] for c in M.METHODS]
    for hh, ll in (_extra or []):
        h.append(hh); l.append(ll)
    _ax.legend(h, l, loc=_loc, bbox_to_anchor=_anchor, ncol=len(h), fontsize=6.3, columnspacing=0.8, handletextpad=0.2, borderaxespad=0.1)


def plotMatchedAccuracy(_cmp: dict):
    '''RMSE of every clean run of each method on the same paths and speeds. Hollow markers are runs on a
        path the method was tuned or fitted on; the count of motor trips on that setting is at the right.'''
    keep = ["S-shape circle, r 100, z 250", "S-shape circle, r 100, z 270", "S-shape circle, r 100, z 280",
            "S-shape circle, r 100, z 300", "C-shape circle, r 150, z 250, 45°", "Figure 8, a 100, z 270", "Helix, r 90, z 300→250"]
    names = {"S-shape circle, r 100, z 250": "S-circle z 250", "S-shape circle, r 100, z 270": "S-circle z 270",
             "S-shape circle, r 100, z 280": "S-circle z 280", "S-shape circle, r 100, z 300": "S-circle z 300",
             "C-shape circle, r 150, z 250, 45°": "C-circle 45°", "Figure 8, a 100, z 270": "Figure 8 z 270",
             "Helix, r 90, z 300→250": "Helix 300→250"}
    entries = {e["label"]: e for e in _cmp["matched"]}
    runs = {(r["ctrl"], r["test"]): r for r in _cmp["runs"]}
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    fig.subplots_adjust(left=0.24, right=0.86, top=0.92, bottom=0.12)
    offs = {"PCC": -0.25, "PCC_Cable": 0.0, "Poly": 0.25}
    for y, lab in enumerate(keep):
        e = entries[lab]
        trips = []
        for ctrl in M.METHODS:
            m = e["methods"][ctrl]
            for test, day, rm in m["runs"]:
                seen = runs[(ctrl, test)]["seen"] in ("tuned on", "fitted on")
                ax.plot(min(rm, 99), y + offs[ctrl], marker=MARKERS[ctrl], ms=3.6, lw=0, mec=COLOURS[ctrl],
                        mfc="white" if seen else COLOURS[ctrl], mew=0.8, alpha=0.9, zorder=3)
            if m.get("n"):
                ax.plot([m["median"]]*2, [y + offs[ctrl] - 0.1, y + offs[ctrl] + 0.1], color=INK, lw=1.1, zorder=4)
            if m["n_trips"]:
                trips.append(f"{LABELS[ctrl].split()[0] if ctrl != 'PCC_Cable' else 'Cable'} {m['n_trips']}")
        if trips:
            ax.text(1.01, y, "trips: " + ", ".join(trips), transform=ax.get_yaxis_transform(), va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([names[k] for k in keep])
    ax.set_ylim(len(keep) - 0.5, -0.6)
    ax.set_xlim(0, 100); ax.set_xlabel("Run RMSE (mm)")
    ax.grid(axis="y", visible=False)
    methodLegend(ax, _anchor=(0.42, 1.09), _extra=[(Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, ms=4), "tuned/fitted path")])
    return save(fig, "method_matched_accuracy")


def plotMatchedErrorTraces(_cmp: dict):
    '''Tip error against time for one representative run of each method (the run closest to that
        method's median) on three matched paths.'''
    groups = [("S-shape circle, r 100, z 250", "S-shape circle, r 100 mm, z 250 mm"),
              ("Figure 8, a 100, z 270", "Figure 8, a 100 mm, z 270 mm"),
              ("Helix, r 90, z 300→250", "Helix, r 90 mm, z 300 → 250 mm")]
    entries = {e["label"]: e for e in _cmp["matched"]}
    fig, axs = plt.subplots(3, 1, figsize=(COL_W, 4.3), sharex=False, gridspec_kw={"hspace": 0.55})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.08)
    chosen = {}
    for ax, (lab, title) in zip(axs, groups):
        e = entries[lab]
        for ctrl in M.METHODS:
            m = e["methods"][ctrl]
            if not m.get("n"):
                continue
            pacDay = e["methods"]["Poly"]["runs"][0][1] if e["methods"]["Poly"].get("n") else None
            pool = [x for x in m["runs"] if x[1] == pacDay] or m["runs"]        # same test day as PAC where possible
            med = float(np.median([x[2] for x in pool]))
            test, day, rm = min(pool, key=lambda x: abs(x[2] - med))
            chosen.setdefault(lab, []).append((ctrl, test))
            d, run, ok, dt, curh, _ = M.P.A._prepareTest(test, ctrl)
            t = d["t"][run]
            err = np.linalg.norm(d["des"][run, :3] - curh, axis=1)
            ax.plot(t, np.where(ok, err, np.nan), color=COLOURS[ctrl], lw=0.8, label=f"{LABELS[ctrl]} {test} ({rm:.0f} mm)")
        ax.set_title(title, pad=2)
        ax.set_xlim(0, 30); ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Tip error (mm)")
        ax.legend(loc="upper right", ncol=3, fontsize=5.6, columnspacing=0.6, handlelength=1.2, borderaxespad=0.1)
    axs[-1].set_xlabel("Task time (s)")
    print(f"  traces: {chosen}")
    return save(fig, "method_error_traces")


def shade(_hex: str, _k: float) -> tuple:
    '''_hex darkened towards black by the fraction _k.'''
    c = np.array(matplotlib.colors.to_rgb(_hex))
    return tuple(c*(1 - _k))


def lapErrors(_test: int, _ctrl: str, _grid: np.ndarray) -> tuple:
    '''Tip error of each lap of one run, every lap on the same time-within-lap grid (NaN where the
        camera was gated), and the lap period.'''
    t, des, cur, dt = PF.homeRefPaths(_test, _ctrl)
    meta = loadMeta(_test, _ctrl)
    period = meta["taskTime"]/meta["movementFreq"]
    e = np.linalg.norm(des - cur, axis=1)
    laps = []
    for k in range(int(round(t[-1]/period))):
        m = (t >= k*period) & (t < (k + 1)*period)
        laps.append((t[m] - k*period, e[m]))
    return laps, period


def plotMethodRepeatability(_cmp: dict, _panels: tuple = (("S-shape circle, r 100, z 270", "PCC_Cable"),
                                                          ("S-shape circle, r 100, z 300", "PCC_Cable"),
                                                          ("S-shape circle, r 100, z 300", "Poly"))):
    '''Figure 17's format for the proposed methods: every lap of every clean run of the method's most
        repeated settings overlaid, with the number of runs and test days, and PCC's median lap on the
        same setting for reference. Runs from a method's second test day are drawn darker.'''
    matched = {m["label"]: m["methods"] for m in _cmp["matched"]}
    fig, axs = plt.subplots(len(_panels), 1, figsize=(COL_W, 1.25*len(_panels) + 0.55), sharex=True, gridspec_kw={"hspace": 0.42})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.1)
    grid = np.linspace(0, 7.5, 151)
    for i, (ax, (label, ctrl)) in enumerate(zip(axs, _panels)):
        runs = matched[label][ctrl]["runs"]
        period = 7.5
        dayOrder = sorted({x[1] for x in runs})
        for test, day, rmse in runs:
            laps, period = lapErrors(test, ctrl, grid)
            c = COLOURS[ctrl] if dayOrder.index(day) == 0 else shade(COLOURS[ctrl], 0.45)   # later test day darker
            for tl, el in laps:
                ax.plot(tl, el, color=c, lw=0.6, alpha=0.3 if dayOrder.index(day) == 0 else 0.45)
        # PCC's median lap on the same setting, for scale
        ref = []
        for test, day, rmse in matched[label]["PCC"]["runs"]:
            laps, _ = lapErrors(test, "PCC", grid)
            ref += [np.interp(grid, tl, el, left=np.nan, right=np.nan) for tl, el in laps if tl.size > 10]
        ax.plot(grid, np.nanmedian(np.array(ref), axis=0), color=COLOURS["PCC"], lw=1.0, ls=(0, (3, 1.5)))
        rm = [x[2] for x in runs]
        nDays = len({x[1] for x in runs})
        z = label.split("z ")[1]
        ax.set_title(f"({'abc'[i]}) {LABELS[ctrl]}, z {z} mm: {len(runs)} runs, {nDays} test day{'s' if nDays > 1 else ''}", pad=3, loc="left")
        ax.text(period*0.99, 96, f"Run RMSE {min(rm):.0f}–{max(rm):.0f} mm (median {np.median(rm):.0f})", ha="right", va="top", fontsize=6.5, color=INK2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8})
        ax.set_xlim(0, period); ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100])
    axs[len(_panels)//2].set_ylabel("Tip error (mm)")
    axs[-1].set_xlabel("Time within lap (s)")
    h = [Line2D([], [], color=COLOURS["PCC_Cable"], lw=1.2), Line2D([], [], color=COLOURS["Poly"], lw=1.2),
         Line2D([], [], color=COLOURS["PCC"], lw=1.0, ls=(0, (3, 1.5)))]
    fig.legend(h, ["Cable-comp. PCC lap", "PAC lap", "PCC, median lap"], loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.0),
               columnspacing=0.9, handlelength=1.6, borderaxespad=0.1)
    print("  repeatability runs: " + "; ".join(f"{c} {l}: {[x[0] for x in matched[l][c]['runs']]}" for l, c in _panels))
    return save(fig, "method_repeatability")


def plotStepTimes(_cmp: dict):
    '''Distribution of Pi control-step durations of each method, over the runs of COMPARE_DAYS
        (the two test days on which all three methods were run).'''
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.2)
    bins = np.arange(30, 121, 2)
    for ctrl in M.METHODS:
        runs = [r for r in M.selectRuns(_cmp["runs"], ctrl) if r["class"] not in ("stopped early", "motor trip") and r["day"] in M.COMPARE_DAYS]
        a = np.concatenate([M.loadStepTimes(r["test"], ctrl) for r in runs])
        h, e = np.histogram(np.clip(a, 30, 120), bins=bins)
        ax.stairs(100*h/a.size, e, color=COLOURS[ctrl], lw=1.1)
        ax.stairs(100*h/a.size, e, color=COLOURS[ctrl], alpha=0.12, fill=True)
    ax.axvline(40, color=INK, lw=0.8)
    ax.text(43, 0.95, "40 ms target (25 Hz)", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK, va="top")
    ax.set_xlim(30, 120); ax.set_xlabel("Pi control-step duration (ms)"); ax.set_ylabel("Share of steps (%)")
    methodLegend(ax, _anchor=(0.5, 1.2))
    return save(fig, "method_step_times")


def plotMotorTracking(_tests: dict = None, _window: tuple = (10, 16)):
    '''Motor tracking error (command minus measured, all six motors) of each method on the same helix,
        run the same day, over a 6 s window: what the vibration under PAC looks like at the Pi's sample rate.'''
    _tests = _tests or {"PCC": 293, "PCC_Cable": 85, "Poly": 37}
    fig, axs = plt.subplots(3, 1, figsize=(COL_W, 3.1), sharex=True, gridspec_kw={"hspace": 0.35})
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.12)
    for ax, ctrl in zip(axs, M.METHODS):
        d = loadTest(_tests[ctrl], ctrl)
        m = (d["t"] >= _window[0]) & (d["t"] <= _window[1])
        rate = 1/np.median(np.diff(d["t"][d["t"] >= 0]))
        for j in range(6):
            ax.plot(d["t"][m], d["dm"][m, j] - d["cm"][m, j], color=COLOURS[ctrl], lw=0.6, alpha=0.75, marker=".", ms=1.6)
        ax.axhline(0, color=AXIS, lw=0.5)
        ax.set_ylim(-0.8, 0.8)
        ax.text(0.01, 0.92, f"{LABELS[ctrl]}, Test {_tests[ctrl]} ({rate:.0f} Hz median)", transform=ax.transAxes, fontsize=6.5, color=INK, va="top")
    axs[1].set_ylabel("Motor tracking error (rad)")
    axs[-1].set_xlabel("Task time (s)")
    axs[-1].set_xlim(*_window)
    return save(fig, "method_motor_tracking")


def plotReliability(_cmp: dict):
    '''(a) Share of runs that ended in a motor runaway, with 95 % Wilson intervals; (b) the largest motor
        tracking error of every run before any runaway, by method, runaways filled and the PD current
        limit (15 A at Kp = 11 A/rad, 1.36 rad) marked.'''
    rel = _cmp["reliability"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.3), gridspec_kw={"height_ratios": [1, 1.7], "hspace": 0.55})
    fig.subplots_adjust(left=0.27, right=0.97, top=0.9, bottom=0.12)
    for y, ctrl in enumerate(M.METHODS):
        s = rel[ctrl]["evaluation"]
        lo, hi = s["ci95"]
        a1.plot([100*lo, 100*hi], [y, y], color=COLOURS[ctrl], lw=2.2, alpha=0.35, solid_capstyle="round")
        a1.plot(100*s["runaway_rate"], y, marker=MARKERS[ctrl], color=COLOURS[ctrl], ms=5, lw=0)
        a1.text(100*hi + 1.5, y, f"{s['n_runaway']} of {s['n']} runs", va="center", fontsize=6.5, color=INK2)
    a1.set_yticks(range(3)); a1.set_yticklabels([LABELS[c] for c in M.METHODS]); a1.set_ylim(2.6, -0.6)
    a1.set_xlim(0, 60); a1.set_xlabel("Runs ending in a motor runaway (%)")
    a1.grid(axis="y", visible=False)
    a1.set_title("(a) Runaway rate, 95% interval", pad=3)
    rng = np.random.default_rng(2)
    runs = [r for r in M.selectRuns(_cmp["runs"], None, "evaluation") if r["class"] in ("clean", "motor trip", "camera: no data", "camera: offset")]
    for y, ctrl in enumerate(M.METHODS):
        v = [r for r in runs if r["ctrl"] == ctrl]
        x = np.array([r["track_max"] for r in v])
        ra = np.array([r["trip_kind"] == "runaway" for r in v])
        jit = y + rng.uniform(-0.25, 0.25, x.size)
        a2.scatter(np.minimum(x[~ra], 3.9), jit[~ra], s=6, facecolors="white", edgecolors=COLOURS[ctrl], lw=0.6, marker=MARKERS[ctrl], zorder=3)
        a2.scatter(np.minimum(x[ra], 3.9), jit[ra], s=12, color=COLOURS[ctrl], marker=MARKERS[ctrl], lw=0, zorder=4)
    a2.axvline(M.SAT_RAD, color=INK, lw=0.8, ls=(0, (3, 2)))
    a2.text(M.SAT_RAD + 0.06, -0.5, "PD output at the 15 A current limit", fontsize=6, color=INK, va="center")
    a2.set_yticks(range(3)); a2.set_yticklabels([LABELS[c] for c in M.METHODS]); a2.set_ylim(2.5, -0.75)
    a2.set_xlim(0, 4); a2.set_xlabel("Largest motor tracking error before any runaway (rad)")
    a2.grid(axis="y", visible=False)
    a2.set_title("(b) Each run; filled = ended in a runaway", pad=3)
    return save(fig, "method_reliability")


def plotRunaway(_test: int = 246, _ctrl: str = "PCC", _span: float = 3.0):
    '''Commanded and measured angle of the motor that ran away in one run, around the onset.'''
    d = loadTest(_test, _ctrl)
    row = M.calcMotorLoop(_test, _ctrl)
    j = "ABCDEF".index(row["onset_motor"])
    t0 = row["onset_t"]
    m = (d["t"] >= t0 - _span) & (d["t"] <= t0 + 0.5)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.21)
    ax.plot(d["t"][m], d["dm"][m, j], color=MUTED, lw=1.4, ls=(0, (4, 2)), label="Commanded")
    ax.plot(d["t"][m], d["cm"][m, j], color=INK, lw=1.0, marker=".", ms=2, label="Measured (encoder)")
    ax.axhline(-20, color=AXIS, lw=0.6); ax.text(t0 - 0.9, -19, "−20 rad trip", fontsize=6, color=INK2, va="bottom", ha="right")
    ax.set_ylim(-32, 8); ax.set_xlim(t0 - _span, t0 + 0.5)
    ax.set_xlabel("Task time (s)"); ax.set_ylabel(f"Motor {row['onset_motor']} (rad)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1)
    ax.text(0.02, 0.06, f"{LABELS[_ctrl]}, Test {_test}", transform=ax.transAxes, ha="left", fontsize=6.5, color=INK2)
    return save(fig, "runaway_example")


def main():
    setStyle()
    res = loadJson("pccResults.json", P.main)
    cmp = loadJson("methodComparison.json", M.main)
    print("Thesis figures:")
    plotPccRunsByType(res)
    typical = 288
    plotPccExampleRun(typical, next(r["rmse"] for r in res["runs"] if r["test"] == typical))
    plotPccRepeatability(res)
    plotPccSpeed(res)
    plotLiteratureComparison(res)
    plotMatchedAccuracy(cmp)
    plotMatchedErrorTraces(cmp)
    plotMethodRepeatability(cmp)
    plotStepTimes(cmp)
    plotMotorTracking()
    plotReliability(cmp)
    plotRunaway()


if __name__ == "__main__":
    main()
