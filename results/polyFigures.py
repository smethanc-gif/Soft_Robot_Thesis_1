'''Figures for results/POLY_CONSTANTS.md. Run from the repo root after polyFit.py:
    python results/polyFigures.py
'''
import os
import sys
import json
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polyFit as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette, light mode (slots 1-3 validate for colour-blind separation)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

C = json.load(open(P.CONSTANTS_FILE))
S = C["mean_tip_rmse_mm"]


def modelComparison():
    models = [("PCC", GREY), ("PCC + tendon gains", BLUE),
              ("Poly + gains, free shape", ORANGE), ("Poly + gains, shape held", AQUA)]
    checks = list(next(iter(S.values())).keys())
    labels = {"validation (mf 4)": "Validation runs\n(movementfreq 4)", "unseen figure 8s": "Figure 8s\n(never seen in training)"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.19
    for i, (name, col) in enumerate(models):
        x = numpy.arange(len(checks)) + (i - 1.5)*(w + 0.01)
        vals = [S[name][c] for c in checks]
        bars = ax.bar(x, vals, w, color=col, label=name, edgecolor=SURFACE, linewidth=1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([labels.get(c, c) for c in checks])
    ax.set_ylabel("Tip prediction RMSE (mm)")
    ax.set_title("How well each model predicts the camera tip from the motor angles", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poly_model_comparison.png"), dpi=150)
    plt.close(fig)


def tendonGains():
    g = numpy.array(C["tendon_gains_ABCDEF"])
    names = "ABCDEF"
    longT = [0, 2, 4]
    # direction each tendon pulls the tip (deg), from constantCurve.rotationalOffset + PCC_Control's offsets
    pull = (90 - numpy.degrees(P._R.rotationalOffset) - numpy.array([0, 60, 120, 180, -120, -60])) % 360

    fig = plt.figure(figsize=(10, 4.3))
    ax = fig.add_subplot(1, 2, 1)
    cols = [BLUE if i in longT else ORANGE for i in range(6)]
    ax.bar(range(6), g, 0.6, color=cols, edgecolor=SURFACE)
    for i, v in enumerate(g):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax.text(5.45, 1.0, "PCC assumes 1.00", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(6)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Cable travel delivered / modelled")
    ax.set_title("Fitted tendon gains", loc="left", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="runs through both sections (A, C, E)"),
                       Patch(color=ORANGE, label="section 1 only (B, D, F)")],
              frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(axis="x", visible=False)

    # Top view: test 191 step-return reach by direction vs where each tendon pulls
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    reach = {0: 0.73, 45: 0.85, 90: 0.69, 135: 0.77, 180: 0.87, 225: 0.71, 270: 0.85, 315: 0.80}
    th = numpy.radians(list(reach) + [0])
    ax2.plot(th, list(reach.values()) + [reach[0]], color=INK, linewidth=2, marker="o", markersize=5,
             label="Test 191 reach by bend direction")
    for i, d in enumerate(pull):
        col = BLUE if i in longT else ORANGE
        ax2.plot([numpy.radians(d)]*2, [0.55, 0.95], color=col, linewidth=2)
        ax2.text(numpy.radians(d), 1.0, f"{names[i]} {g[i]:.2f}", ha="center", va="center", fontsize=9, color=INK)
    ax2.set_ylim(0.55, 1.02)
    ax2.set_yticks([0.6, 0.7, 0.8, 0.9]); ax2.set_yticklabels(["0.6", "0.7", "0.8", "0.9"], fontsize=7, color=INK2)
    ax2.set_title("Reach by direction (top view) vs tendon pull directions", fontsize=11, pad=14)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poly_tendon_gains.png"), dpi=150)
    plt.close(fig)


def unseenFigure8(test=206):
    ''' Fit without any figure 8s, then predict one - so this is an honest unseen path. '''
    lag = C["lag_samples"]
    no8 = {k: v for k, v in P.TRAIN.items() if k != "figure 8"}
    gains, _ = P.fitTendonGains(no8, lag, verbose=False)
    poly = P.fit(no8, lag, P.POLY_VARIANT, gains, P.SHAPE_WEIGHT, verbose=False)
    q, tip = P.testData(test, lag)
    k = slice(0, len(q)//2)                           # first half of the run, for clarity
    pcc, pol = P.polyTip(P.pccParams(), q[k]), P.polyTip(poly, q[k], gains)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(tip[k, 0], tip[k, 1], color=INK, linewidth=2, label="Measured (camera)")
    ax.plot(pcc[:, 0], pcc[:, 1], color=GREY, linewidth=2, linestyle="--",
            label=f"PCC prediction (RMSE {P.rmse(pcc, tip[k]):.1f} mm)")
    ax.plot(pol[:, 0], pol[:, 1], color=AQUA, linewidth=2,
            label=f"Poly prediction (RMSE {P.rmse(pol, tip[k]):.1f} mm)")
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(f"Test {test}, a figure 8 the model never saw: tip predicted from the motor angles",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poly_unseen_figure8.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    modelComparison()
    tendonGains()
    unseenFigure8()
    print("saved to", OUT)
