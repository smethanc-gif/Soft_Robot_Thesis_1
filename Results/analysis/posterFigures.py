'''Figures for the two A3 event posters (the robot and PCC; PCC's accuracy), drawn at the size they are
printed so their text stays readable from a metre away. PCC only; the other methods are not plotted.

Reads pccResults.json (pccResults.py), running it first if missing; uses the logged runs through
presentationFigures.homeRefPaths, like thesisFigures.py. Equations are typeset with matplotlib's mathtext in Computer Modern.

Run from anywhere:
    python Results/analysis/posterFigures.py
PNGs (300 dpi, transparent where they sit on the poster's white) go to Results/analysis/output/posterFigures/.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from Results.logReader import loadMeta
from Results.analysis import pccResults as P
from Results.analysis import presentationFigures as PF

warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
OUT = P.OUT_DIR / "posterFigures"
PX = 1/96                                    # one CSS px of the A3 poster artboard, in inches
INK, INK2, MUTED, RULE = "#12161C", "#4A5260", "#7A828F", "#D5D9E0"
BLUE, BLUE_DARK, DESIRED, ORANGE = "#2A78D6", "#1F63B8", "#B9BEC7", "#EB6834"
MAIN_TYPES = ["S-shape circle", "Figure 8", "Helix", "C-shape circle", "Step-return (C)"]
LENGTH = 320.0                               # mm, 2 x 145 mm sections + 30 mm tool


def setStyle():
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "text.color": INK,
        "axes.edgecolor": RULE, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.labelsize": 11,
        "axes.titlesize": 11.5, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E9EE", "grid.linewidth": 0.8,
        "xtick.color": RULE, "ytick.color": RULE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.frameon": False, "legend.fontsize": 10.5,
        "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "savefig.dpi": 300, "mathtext.fontset": "cm",
        "figure.facecolor": "white", "axes.facecolor": "white"})


def save(_fig, _name: str, _transparent: bool = False) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{_name}.png"
    _fig.savefig(path, transparent=_transparent)
    plt.close(_fig)
    print(f"  {path.name}")
    return path


def loadResults() -> dict:
    path = P.OUT_DIR / "pccResults.json"
    if not path.exists():
        P.main()
    with open(path) as f:
        return json.load(f)


# ---- Poster 2: results ----

def plotTypicalRun(_test: int = 288, _w: float = 485, _h: float = 262):
    '''Desired and measured tip path of one run close to the S-shaped circle median: top and side views.'''
    t, des, cur, dt = PF.homeRefPaths(_test)
    zDes = float(np.mean(des[:, 2]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(_w*PX, _h*PX), gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(left=0.135, right=0.985, top=0.77, bottom=0.2)
    for a, (i, j), lab in ((a1, (0, 1), "Top view"), (a2, (0, 2), "Side view")):
        a.plot(des[:, i], des[:, j], color=DESIRED, lw=4.5, label="Commanded path")
        a.plot(cur[:, i], cur[:, j], color=BLUE, lw=1.0, label="Measured tip")
        a.set_title(lab, pad=4)
        a.set_xlabel("x (mm)")
    a1.set_aspect("equal"); a1.set_xlim(-125, 125); a1.set_ylim(-125, 125); a1.set_ylabel("y (mm)")
    a1.set_xticks([-100, 0, 100]); a1.set_yticks([-100, 0, 100])
    a2.set_xlim(-125, 125); a2.set_ylim(zDes - 40, zDes + 50); a2.set_ylabel("z (mm)"); a2.set_xticks([-100, 0, 100])
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, bbox_to_anchor=(0.02, 1.0), handlelength=1.6, columnspacing=1.2)
    return save(fig, "poster_typical_run")


def plotByPath(_res: dict, _w: float = 485, _h: float = 262):
    '''RMSE of every clean PCC run by path, medians marked, with the % of robot length on the right.'''
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.2)
    rng = np.random.default_rng(1)
    for i, typ in enumerate(MAIN_TYPES):
        rm = np.array([r["rmse"] for r in PF.cleanRuns(_res, typ)])
        ax.scatter(i + rng.uniform(-0.22, 0.22, rm.size), rm, s=13, color=BLUE, alpha=0.5, lw=0, zorder=3)
        med = np.median(rm)
        ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=2.2, zorder=4, solid_capstyle="butt")
        ax.text(i + 0.36, med, f"{med:.0f}", va="center", fontsize=10.5, color=INK, fontweight="semibold")
        ax.text(i, 4, f"n = {rm.size}", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(MAIN_TYPES)))
    ax.set_xticklabels(["S-shaped\ncircle", "Figure 8", "Helix", "C-shaped\ncircle", "Step-\nreturn"])
    ax.set_xlim(-0.6, len(MAIN_TYPES) - 0.25); ax.set_ylim(0, 110)
    ax.set_ylabel("Run RMSE (mm)")
    ax.grid(axis="x", visible=False)
    sec = ax.secondary_yaxis("right", functions=(lambda v: 100*v/LENGTH, lambda p: p*LENGTH/100))
    sec.set_ylabel("% of robot length"); sec.set_yticks([0, 10, 20, 30])
    sec.spines["right"].set_visible(True); sec.spines["right"].set_color(RULE)
    return save(fig, "poster_by_path")


def plotRepeatability(_res: dict, _w: float = 485, _h: float = 262):
    '''Every lap of the eight identical PCC runs of one S-shaped circle, overlaid.'''
    g = _res["repeats"][0]
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.2)
    period = None
    for test in g["tests"]:
        t, des, cur, dt = PF.homeRefPaths(test)
        if period is None:
            meta = loadMeta(test, "PCC")
            period = meta["taskTime"]/meta["movementFreq"]
        e = np.linalg.norm(des - cur, axis=1)
        for k in range(int(round(t[-1]/period))):
            m = (t >= k*period) & (t < (k + 1)*period)
            ax.plot(t[m] - k*period, e[m], color=BLUE, lw=0.8, alpha=0.3)
    ax.set_xlim(0, period); ax.set_ylim(0, 100)
    ax.set_xlabel("Time within one lap (s)"); ax.set_ylabel("Tip error (mm)")
    ax.text(period*0.985, 96, f"{g['n']} runs × 4 laps · run RMSE {g['rmse_min']:.0f}–{g['rmse_max']:.0f} mm", ha="right", va="top",
            fontsize=10, color=INK2, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    return save(fig, "poster_repeatability")


LITERATURE = [  # (citation, conditions, low %, high %, sensing) - same values as thesisFigures.LITERATURE
    ("(Zhai et al., 2025)", "planar · camera · 7 mm/s", 100*2.8/270, 100*2.8/270, "camera"),
    ("(Gong et al., 2025)", "planar · static · open loop", 1.7, 1.7, "none"),
    ("(Peng et al., 2024)", "3D · tip from IMUs", 100*20/450, 100*20/450, "imu"),
    ("(Zhai et al., 2025)", "planar · camera · 53 mm/s", 100*19.3/270, 100*19.3/270, "camera"),
    ("(Qi et al., 2024)", "3D soft · cable model", 100*17.0/206, 100*37.6/206, "none"),
    ("(Peng et al., 2024)", "3D · tip from tendon lengths", 100*60/450, 100*60/450, "none"),
    ("(Qi et al., 2024)", "3D soft · PCC baseline", 100*35.2/206, 100*59.2/206, "none"),
]


def plotLiterature(_res: dict, _w: float = 485, _h: float = 300):
    '''Tip error as % of robot length: published results against this work's per-path medians.'''
    meds = [_res["by_type"][t]["rmse_median"] for t in MAIN_TYPES]
    mine = ("This work (PCC)", "3D soft · encoders only", 100*min(meds)/LENGTH, 100*max(meds)/LENGTH, "this")
    rows = sorted(LITERATURE + [mine], key=lambda r: r[2])[::-1]
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.4, right=0.97, top=0.88, bottom=0.17)
    ax.axvspan(mine[2], mine[3], color="#EAF2FC", zorder=0, lw=0)
    col = {"camera": INK2, "imu": INK2, "none": MUTED, "this": BLUE}
    for y, (cite, cond, lo, hi, kind) in enumerate(rows):
        c = col[kind]
        if hi - lo > 0.3:
            ax.plot([lo, hi], [y, y], color=c, lw=6.5, solid_capstyle="round", zorder=3)
        else:
            ax.plot([lo], [y], "D" if kind == "imu" else "o", color=c, ms=6.5 if kind != "imu" else 5.5, zorder=3)
        tr = ax.get_yaxis_transform()
        ax.text(-0.04, y + 0.19, cite, transform=tr, ha="right", va="center", fontsize=9.8,
                color=BLUE_DARK if kind == "this" else INK, fontweight="semibold")
        ax.text(-0.04, y - 0.24, cond, transform=tr, ha="right", va="center", fontsize=8.6, color=INK2)
    ax.set_yticks([]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 30); ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Tip error, % of robot length")
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    h = [Line2D([], [], color=INK2, marker="o", lw=0, ms=6), Line2D([], [], color=INK2, marker="D", lw=0, ms=5),
         Line2D([], [], color=MUTED, marker="o", lw=0, ms=6)]
    ax.legend(h, ["camera in the loop", "IMUs", "no tip sensing"], loc="lower left", bbox_to_anchor=(-0.62, 1.01), fontsize=9.5,
              ncol=3, columnspacing=0.9, handletextpad=0.25, borderaxespad=0.1)
    return save(fig, "poster_literature")


# ---- Poster 1: the method ----

def plotArcGeometry(_w: float = 470, _h: float = 210):
    '''One section as a circular arc: side view in its bending plane (radius 1/kappa, bend angle
        theta = kappa*s, arc length s) and the top view giving the bending direction phi.'''
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(_w*PX, _h*PX), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.12})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    for a in (a1, a2):
        a.set_aspect("equal"); a.axis("off")
    # side view: base at origin, centre of curvature at (R, 0), arc of angle theta
    R, th = 1.0, np.deg2rad(70)
    phi = np.linspace(0, th, 80)
    x, y = R - R*np.cos(phi), R*np.sin(phi)
    a1.plot([-0.25, 1.35], [0, 0], color=RULE, lw=1.2)
    a1.plot([0, 0], [0, 1.05], color=RULE, lw=1.0, ls=(0, (3, 3)))
    a1.plot(x, y, color=BLUE, lw=5, solid_capstyle="round", zorder=3)
    a1.plot([R, x[-1]], [0, y[-1]], color=INK2, lw=1.2)
    a1.plot([R, 0], [0, 0], color=INK2, lw=1.2)
    a1.add_patch(Arc((R, 0), 0.42, 0.42, theta1=180 - np.rad2deg(th), theta2=180, color=INK, lw=1.2))
    a1.text(R - 0.3, 0.1, r"$\theta=\kappa s$", fontsize=13, ha="right", va="bottom", color=INK)
    a1.text(0.55, -0.1, r"$r = 1/\kappa$", fontsize=13, ha="center", va="top", color=INK)
    a1.text(0.06, 0.55, r"$s$", fontsize=14, ha="left", color=BLUE_DARK)
    a1.plot([x[-1]], [y[-1]], "o", color=INK, ms=5, zorder=4)
    a1.text(x[-1] + 0.08, y[-1] - 0.02, "section end", fontsize=9.5, color=INK2, va="center", ha="left")
    a1.text(0, -0.1, "base", fontsize=9.5, color=INK2, ha="center", va="top")
    a1.text(-0.25, 1.08, "Side view (bending plane)", fontsize=10.5, color=INK, fontweight="semibold", va="bottom")
    a1.set_xlim(-0.3, 1.45); a1.set_ylim(-0.32, 1.25)
    # top view: bending direction phi from x
    a2.add_patch(Circle((0, 0), 0.75, fill=False, color=RULE, lw=1.2))
    a2.annotate("", xy=(1.0, 0), xytext=(-1.0, 0), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.0})
    a2.annotate("", xy=(0, 1.0), xytext=(0, -1.0), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.0})
    a2.text(1.02, -0.05, "$x$", fontsize=12, va="top"); a2.text(0.05, 1.02, "$y$", fontsize=12, ha="left")
    ang = np.deg2rad(40)
    a2.add_patch(FancyArrowPatch((0, 0), (0.75*np.cos(ang), 0.75*np.sin(ang)), arrowstyle="-|>", mutation_scale=14, color=BLUE, lw=3))
    a2.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=40, color=INK, lw=1.2))
    a2.text(0.38, 0.12, r"$\varphi$", fontsize=14, color=INK)
    a2.text(-1.0, 1.08, "Top view", fontsize=10.5, color=INK, fontweight="semibold", va="bottom")
    a2.set_xlim(-1.05, 1.2); a2.set_ylim(-1.05, 1.25)
    return save(fig, "poster_arc_geometry", _transparent=True)


def _arc2d(_start, _heading, _k, _s, _n=60):
    '''Points of a planar constant-curvature arc from _start, initial heading _heading (rad), curvature
        _k (1/mm, positive bends left) and length _s (mm); returns the points and the end heading.'''
    u = np.linspace(0, _s, _n)
    if abs(_k) < 1e-9:
        pts = np.c_[_start[0] + u*np.cos(_heading), _start[1] + u*np.sin(_heading)]
    else:
        h = _heading + _k*u
        pts = np.c_[_start[0] + (np.sin(h) - np.sin(_heading))/_k, _start[1] - (np.cos(h) - np.cos(_heading))/_k]
    return pts, _heading + _k*_s


def plotShapeSolve(_w: float = 500, _h: float = 280):
    '''Side view of the two-section shape solve: the target tip pose, the 30 mm tool taken off along the
        target direction, the joint placed by the solver, and one arc per section.'''
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_aspect("equal"); ax.axis("off")
    p1, h1 = _arc2d((0, 0), np.pi/2, -np.deg2rad(42)/145, 145)          # section 1 bends right
    p2, h2 = _arc2d(p1[-1], h1, np.deg2rad(72)/128, 128)                 # section 2 bends back left (compressed)
    tip = p2[-1] + 30*np.array([np.cos(h2), np.sin(h2)])
    ax.plot([-70, 90], [0, 0], color=INK2, lw=1.5)
    for xx in np.arange(-66, 90, 10):
        ax.plot([xx, xx - 7], [0, -7], color=RULE, lw=1)
    ax.plot(p1[:, 0], p1[:, 1], color=BLUE, lw=9, solid_capstyle="butt", zorder=3)
    ax.plot(p2[:, 0], p2[:, 1], color="#7FB0EA", lw=9, solid_capstyle="butt", zorder=3)
    ax.plot([p2[-1, 0], tip[0]], [p2[-1, 1], tip[1]], color=MUTED, lw=5, solid_capstyle="butt", zorder=3)
    d = np.array([np.cos(h2), np.sin(h2)])
    ax.annotate("", xy=tip + 38*d, xytext=tip + 4*d, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3, "ls": (0, (3, 2))})
    ax.add_patch(Circle(tip, 6.5, color="#C6E03B", zorder=5, ec=INK, lw=0.8))
    ax.add_patch(Circle(tip, 13, fill=False, ec=INK, lw=1.2, zorder=5))
    ax.add_patch(Circle(p1[-1], 6.5, color="white", ec=INK, lw=2, zorder=6))
    ax.add_patch(Circle(p2[-1], 3.5, color=INK, zorder=6))
    lab = {"fontsize": 10.5, "color": INK}
    sub = {"fontsize": 9.5, "color": INK2}
    ax.text(tip[0] + 20, tip[1] + 2, "Target", ha="left", va="bottom", fontweight="semibold", **lab)
    ax.text(tip[0] + 20, tip[1] - 1, "tip position + tool direction", ha="left", va="top", **sub)
    mid2 = p2[len(p2)//2]
    ax.text(mid2[0] + 22, mid2[1] + 8, "Section 2", ha="left", va="center", fontweight="semibold", **lab)
    ax.text(mid2[0] + 22, mid2[1] - 8, r"$\kappa_2,\ \varphi_2,\ s_2$", ha="left", va="center", fontsize=11.5, color=INK)
    mid1 = p1[len(p1)//2]
    ax.text(mid1[0] + 22, mid1[1] + 8, "Section 1", ha="left", va="center", fontweight="semibold", **lab)
    ax.text(mid1[0] + 22, mid1[1] - 8, r"$\kappa_1,\ \varphi_1,\ s_1$", ha="left", va="center", fontsize=11.5, color=INK)
    ax.text(p1[-1, 0] - 24, p1[-1, 1] + 3, "Joint", ha="right", va="bottom", fontweight="semibold", **lab)
    ax.text(p1[-1, 0] - 24, p1[-1, 1] - 2, "placed by the solver", ha="right", va="top", **sub)
    tm = (p2[-1] + tip)/2
    ax.text(tm[0] - 20, tm[1] - 2, "30 mm tool", ha="right", va="center", **sub)
    ax.text(-62, 6, "base", ha="left", va="bottom", **sub)
    ax.set_xlim(-190, 175); ax.set_ylim(-14, 330)
    return save(fig, "poster_shape_solve", _transparent=True)


def plotTendonLayout(_w: float = 240, _h: float = 240):
    '''Cross-section of the robot: six tendons on a 15.25 mm radius; A, C and E run to the tip, B, D and F
        end at the joint. Angles as in the oral presentation's tendon_layout figure.'''
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), 19.5, color="#F1F3F6", ec=MUTED, lw=1.2, zorder=1))
    ax.add_patch(Circle((0, 0), 15.25, fill=False, ec=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=2))
    ang = {"A": 286, "B": 226, "C": 166, "D": 106, "E": 46, "F": 346}
    for name, a in ang.items():
        t = np.deg2rad(a)
        x, y = 15.25*np.cos(t), 15.25*np.sin(t)
        tip = name in "ACE"
        ax.add_patch(Circle((x, y), 2.1, color=BLUE if tip else "white", ec=BLUE, lw=1.8, zorder=4))
        ax.text(23.5*np.cos(t), 23.5*np.sin(t), name, ha="center", va="center", fontsize=12, fontweight="semibold", color=INK)
    tA = np.deg2rad(ang["A"])
    ax.plot([0, 15.25*np.cos(tA)], [0, 15.25*np.sin(tA)], color=INK2, lw=1.1, zorder=3)
    ax.text(3.4, -6.5, r"$d$", fontsize=12.5, color=INK)
    ax.add_patch(Circle((0, 0), 0.7, color=INK2, zorder=3))
    ax.set_xlim(-27, 27); ax.set_ylim(-27, 27)
    return save(fig, "poster_tendon_layout", _transparent=True)


def plotSpool(_w: float = 190, _h: float = 110):
    '''One motor spool winding its tendon: angle q, radius 5 mm.'''
    fig, ax = plt.subplots(figsize=(_w*PX, _h*PX))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), 5, color="#F1F3F6", ec=INK2, lw=1.4, zorder=2))
    ax.add_patch(Circle((0, 0), 0.6, color=INK2, zorder=3))
    ax.plot([0, 13], [5, 5], color=BLUE, lw=2.4, zorder=3)
    ax.annotate("", xy=(13.8, 5), xytext=(8, 5), arrowprops={"arrowstyle": "<|-", "color": BLUE, "lw": 0})
    ax.plot([0, 0], [0, 5], color=INK2, lw=1, zorder=3)
    ax.text(0.6, 2.4, r"$r$", fontsize=11.5, color=INK, va="center")
    ax.add_patch(Arc((0, 0), 14.5, 14.5, theta1=200, theta2=320, color=INK, lw=1.2))
    ax.annotate("", xy=(5.6, -4.6), xytext=(5.3, -5.0), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2})
    ax.text(-8.6, -6.6, r"$q_i$", fontsize=12.5, color=INK, ha="right")
    ax.text(14.8, 5, "tendon", fontsize=9.5, color=INK2, va="center")
    ax.set_xlim(-12.5, 27); ax.set_ylim(-10, 8.5)
    return save(fig, "poster_spool", _transparent=True)


EQUATIONS = {
    # shape solve, in the form used on the oral presentation's PCC slide (Jones & Walker), with kappa for curvature
    "eq_shape1": r"$\kappa=\dfrac{2\sqrt{x^2+y^2}}{x^2+y^2+z^2}\qquad \varphi=\operatorname{atan}\left(\dfrac{y}{x}\right)$",
    "eq_shape2": r"$s=\dfrac{2\,\operatorname{atan}\left(\dfrac{\sqrt{x^2+y^2}}{z}\right)}{\kappa}$",
    "eq_cable": r"$l_i = 2n\,\sin\left(\dfrac{\kappa s}{2n}\right)\left(\dfrac{1}{\kappa}-d\,\sin(\varphi+\varphi_0+\beta_i)\right)$",
    "eq_motor": r"$q_i=\dfrac{L_{i,0}-l_i}{r_\mathrm{spool}}$",
}


def renderEquations(_pt: float = 15.0):
    '''Each equation as a transparent PNG, tightly cropped, typeset in Computer Modern at _pt (the poster's size).'''
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})   # matplotlib's own TeX layout, Computer Modern
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(8, 1))
        fig.text(0, 0.5, tex, fontsize=_pt, color=INK, va="center")
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  {name}.png")


def main():
    setStyle()
    res = loadResults()
    print("Poster figures:")
    plotTypicalRun()
    plotByPath(res)
    plotRepeatability(res)
    plotLiterature(res)
    plotArcGeometry()
    plotShapeSolve()
    plotTendonLayout()
    plotSpool()
    renderEquations()


if __name__ == "__main__":
    main()
