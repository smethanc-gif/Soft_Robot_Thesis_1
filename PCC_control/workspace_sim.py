"""
workspaceSim.py - Maps the robot's reachable tool tip workspace.

Sweeps the full actuation space of the two-section PCC robot defined in
PCC_Control.py, subject to its physical limits and coupling:
  sMin <= s1 <= sMax                (section arc length)
  s2 = sectL + (s1 - sectL)/2       (mechanical coupling)
  0 <= k*s <= thetaMax              (bend angle per section)
  -pi <= r <= pi                    (bend direction per section)
and forward-kinematics every combination to a tooltip position, including
the straight rigid tool of length toolLength extending along the tool axis
from the end of the bending sections.

Produces one figure with 4 graphs:
  - 3D scatter of the reachable tip positions
  - top down projection onto the X-Y plane
  - X-Z cross-section through the origin (bend planes locked to X-Z)
  - Y-Z cross-section through the origin (bend planes locked to Y-Z)

TILT_METRIC controls what points are coloured by (positions are binned
into VOXEL-sized cells): "range" is the angular spread of tool tilts
reaching each cell, "min"/"max" are the least/most-tilted tool axis
reaching each cell (degrees from vertical), and None colours by height
instead. A position being reachable does not mean every tool orientation
is reachable there - this shows how much orientation freedom the robot
has in each region. The estimate improves with denser sweep resolution.
The console output also prints the overall min/max tilt achievable
across the whole workspace and each cross-section, independent of the
plot colouring.

Install dependency:
  pip install matplotlib
"""

import math
from math import sin as S
from math import cos as C
from math import pi as PI

import matplotlib.pyplot as plt
from PCC_control import ContinuumRobot

# Colour points by tool-tilt info instead of height. One of:
#   "range" - angular spread of achievable tool axes at each position
#   "min"   - the least-tilted (closest to vertical) axis at each position
#   "max"   - the most-tilted axis at each position
#   None    - colour by height (Z) instead
TILT_METRIC = "range"
VOXEL = 15.0        # mm cell size for the tilt binning

# Sweep resolution (points per parameter)
N_S = 6             # arc lengths in the full 3D sweep
N_THETA = 8         # bend angles per section in the full 3D sweep
N_R = 14            # bend directions per section in the full 3D sweep
N_S_PLANE = 25      # arc lengths in the cross-section sweeps
N_THETA_PLANE = 25  # bend angles per section in the cross-section sweeps




def frange(a, b, n):
    if n == 1:
        return [a]
    return [a + (b - a)*i/(n - 1) for i in range(n)]


def tiltMagnitude(t):
    """Tool tilt from vertical (+Z axis), degrees, in [0, 180]."""
    return math.degrees(math.acos(max(-1.0, min(1.0, t[2]))))


def tipPose(robot, s1, th1, r1, th2, r2):
    """Forward kinematics: arc parameters -> tooltip position and tool axis."""
    s2 = robot.sectL + 0.5*(s1 - robot.sectL)   # coupling: ds2 = ds1/2
    k1 = th1/s1
    k2 = th2/s2
    p1, R1 = robot.arcFK(k1, r1, s1)
    p2, R2 = robot.arcFK(k2, r2, s2)
    p = [p1[i] + R1[i][0]*p2[0] + R1[i][1]*p2[1] + R1[i][2]*p2[2]
         for i in range(3)]
    t2 = [R2[0][2], R2[1][2], R2[2][2]]         # tool axis = (R1*R2) z column
    t = [R1[i][0]*t2[0] + R1[i][1]*t2[1] + R1[i][2]*t2[2] for i in range(3)]
    # Straight rigid tool: tooltip sits toolLength along the tool axis past
    # the end of the bending sections (mirrors ContinuumRobot.toolTransform)
    p = [p[i] + robot.toolLength*t[i] for i in range(3)]
    return p, t


def sweepWorkspace(robot):
    """Full 3D sweep over s1, theta1, r1, theta2, r2."""
    pts, tans = [], []
    sVals = frange(robot.sMin, robot.sMax, N_S)
    thVals = frange(0.0, robot.thetaMax, N_THETA)
    rVals = frange(-PI, PI, N_R + 1)[:-1]       # -pi duplicates +pi
    for s1 in sVals:
        for th1 in thVals:
            for r1 in (rVals if th1 > 0 else [0.0]):    # r is moot when straight
                for th2 in thVals:
                    for r2 in (rVals if th2 > 0 else [0.0]):
                        p, t = tipPose(robot, s1, th1, r1, th2, r2)
                        pts.append(p)
                        tans.append(t)
    return pts, tans


def sweepPlane(robot, planeAngle):
    """
    Dense sweep with both bend planes locked to one plane through the
    origin (planeAngle 0 -> X-Z plane, pi/2 -> Y-Z plane), so every point
    lies exactly in the cut. Returns (in-plane coordinate, z, tilt) with
    tilt the signed in-plane tool angle from vertical (radians).
    """
    pts = []
    sVals = frange(robot.sMin, robot.sMax, N_S_PLANE)
    thVals = frange(0.0, robot.thetaMax, N_THETA_PLANE)
    ca, sa = C(planeAngle), S(planeAngle)
    for s1 in sVals:
        for th1 in thVals:
            for r1 in ([planeAngle, planeAngle + PI] if th1 > 0 else [planeAngle]):
                for th2 in thVals:
                    for r2 in ([planeAngle, planeAngle + PI] if th2 > 0 else [planeAngle]):
                        p, t = tipPose(robot, s1, th1, r1, th2, r2)
                        pts.append((p[0]*ca + p[1]*sa,          # signed in-plane coord
                                    p[2],
                                    math.atan2(t[0]*ca + t[1]*sa, t[2])))
    return pts


def tiltCells3D(pts, tans, metric):
    """
    Per-point tool-tilt value (degrees): bin positions into VOXEL cells,
    then per cell report either the angular spread of tool axes reaching
    it ("range": twice the largest angle from the cell's mean axis), or
    the least/most-tilted axis reaching it ("min"/"max": tiltMagnitude,
    degrees from vertical).
    """
    cells = {}
    keys = []
    for p, t in zip(pts, tans):
        key = (math.floor(p[0]/VOXEL), math.floor(p[1]/VOXEL), math.floor(p[2]/VOXEL))
        keys.append(key)
        cells.setdefault(key, []).append(t)

    result = {}
    if metric == "range":
        for key, ts in cells.items():
            m = [sum(t[i] for t in ts) for i in range(3)]
            n = math.sqrt(sum(v*v for v in m))
            if n < 1e-9:                        # axes cancel out: full spread
                result[key] = 180.0
                continue
            m = [v/n for v in m]
            worst = max(math.acos(max(-1.0, min(1.0, sum(t[i]*m[i] for i in range(3)))))
                        for t in ts)
            result[key] = min(math.degrees(2*worst), 180.0)
    else:
        for key, ts in cells.items():
            mags = [tiltMagnitude(t) for t in ts]
            result[key] = min(mags) if metric == "min" else max(mags)
    return [result[k] for k in keys]


def circularRange(angles):
    """Spread of a set of angles on the circle: 2*pi minus the largest gap
    between consecutive sorted angles (immune to the +/-pi wraparound)."""
    if len(angles) == 1:
        return 0.0
    a = sorted(angles)
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    gaps.append(2*PI - (a[-1] - a[0]))
    return 2*PI - max(gaps)


def circularMinMax(angles):
    """(min, max) of the contiguous arc a set of signed angles spans,
    robust to the +/-pi wraparound: rotate the circle so it starts right
    after the largest gap (mirrors circularRange's gap logic). "max" may
    come back above +pi when the arc straddles the wrap point (e.g. a
    cluster around 178 deg and -179 deg returns (178, 181), not (-179, 179))."""
    if len(angles) == 1:
        return angles[0], angles[0]
    a = sorted(angles)
    n = len(a)
    gaps = [(a[i + 1] - a[i], i) for i in range(n - 1)]
    gaps.append((2*PI - (a[-1] - a[0]), n - 1))     # wraparound gap, a[-1] -> a[0]
    _, idx = max(gaps, key=lambda g: g[0])
    if idx == n - 1:                                # largest gap is the wraparound: no wrap crossing
        return a[0], a[-1]
    return a[idx + 1], a[idx] + 2*PI


def tiltCells2D(pts, metric):
    """Same idea for a planar slice: (u, z, tilt) -> per-point value (deg),
    "range", "min", or "max" of the tilts reaching each cell."""
    cells = {}
    keys = []
    for u, z, a in pts:
        key = (math.floor(u/VOXEL), math.floor(z/VOXEL))
        keys.append(key)
        cells.setdefault(key, []).append(a)
    if metric == "range":
        result = {k: math.degrees(circularRange(v)) for k, v in cells.items()}
    else:
        result = {}
        for k, angles in cells.items():
            lo, hi = circularMinMax(angles)
            result[k] = math.degrees(lo if metric == "min" else hi)
    return [result[k] for k in keys]


def sortByColor(vals, colors):
    """Draw high-colour points last so they stay visible in dense areas."""
    order = sorted(range(len(colors)), key=lambda i: colors[i])
    return [vals[i] for i in order], [colors[i] for i in order]


def main():
    robot = ContinuumRobot()

    print("Sweeping full workspace...")
    cloud, tangents = sweepWorkspace(robot)
    print(f"  {len(cloud)} tip positions")
    print("Sweeping X-Z and Y-Z cross-sections...")
    sliceXZ = sweepPlane(robot, 0.0)
    sliceYZ = sweepPlane(robot, PI/2)
    print(f"  {len(sliceXZ)} points per section")

    xs = [p[0] for p in cloud]
    ys = [p[1] for p in cloud]
    zs = [p[2] for p in cloud]
    print(f"x range [{min(xs):.1f}, {max(xs):.1f}] mm, "
          f"y range [{min(ys):.1f}, {max(ys):.1f}] mm, "
          f"z range [{min(zs):.1f}, {max(zs):.1f}] mm")

    # Overall min/max achievable tool tilt, independent of the plot
    # colouring below - this is the direct numeric answer, everywhere the
    # tool tip can reach, pooled together.
    tiltMags = [tiltMagnitude(t) for t in tangents]
    print(f"tool tilt (from vertical) over full workspace: "
          f"min {min(tiltMags):.1f} deg, max {max(tiltMags):.1f} deg")
    xzLo, xzHi = circularMinMax([a for _, _, a in sliceXZ])
    yzLo, yzHi = circularMinMax([a for _, _, a in sliceYZ])
    print(f"X-Z slice in-plane tilt: [{math.degrees(xzLo):.1f}, {math.degrees(xzHi):.1f}] deg")
    print(f"Y-Z slice in-plane tilt: [{math.degrees(yzLo):.1f}, {math.degrees(yzHi):.1f}] deg")

    if TILT_METRIC:
        print(f"Binning tool tilt ({TILT_METRIC})...")
        cloudColor = tiltCells3D(cloud, tangents, TILT_METRIC)
        cloud, cloudColor = sortByColor(cloud, cloudColor)
        xs = [p[0] for p in cloud]
        ys = [p[1] for p in cloud]
        zs = [p[2] for p in cloud]
        xzColor = tiltCells2D(sliceXZ, TILT_METRIC)
        yzColor = tiltCells2D(sliceYZ, TILT_METRIC)
        sliceXZ, xzColor = sortByColor(sliceXZ, xzColor)
        sliceYZ, yzColor = sortByColor(sliceYZ, yzColor)
        labelWord = {"range": "range", "min": "min", "max": "max"}[TILT_METRIC]
        cmap, label = "plasma", f"Tool tilt {labelWord} (deg)"
    else:
        cloudColor = zs
        xzColor = yzColor = None
        cmap, label = "viridis", "Z (mm)"

    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax3d.scatter(xs, ys, zs, c=cloudColor, cmap=cmap, s=1, alpha=0.15)
    if TILT_METRIC:
        fig.colorbar(sc, ax=ax3d, label=label, shrink=0.7)
    ax3d.set_title("Reachable Tip Positions (3D)")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_box_aspect((1, 1, 1))

    axXY = fig.add_subplot(2, 2, 2)
    sc = axXY.scatter(xs, ys, c=cloudColor, cmap=cmap, s=1, alpha=0.3)
    fig.colorbar(sc, ax=axXY, label=label)
    axXY.set_title("Top Down (X-Y projection)")
    axXY.set_xlabel("X (mm)")
    axXY.set_ylabel("Y (mm)")
    axXY.set_aspect("equal")
    axXY.grid(True)

    axXZ = fig.add_subplot(2, 2, 3)
    sc = axXZ.scatter([p[0] for p in sliceXZ], [p[1] for p in sliceXZ],
                      c=xzColor, cmap=cmap if xzColor else None,
                      s=1, alpha=0.3, color=None if xzColor else "tab:blue")
    if xzColor:
        fig.colorbar(sc, ax=axXZ, label=label)
    axXZ.set_title("X-Z cross-section (Y = 0 plane)")
    axXZ.set_xlabel("X (mm)")
    axXZ.set_ylabel("Z (mm)")
    axXZ.set_aspect("equal")
    axXZ.grid(True)

    axYZ = fig.add_subplot(2, 2, 4)
    sc = axYZ.scatter([p[0] for p in sliceYZ], [p[1] for p in sliceYZ],
                      c=yzColor, cmap=cmap if yzColor else None,
                      s=1, alpha=0.3, color=None if yzColor else "tab:red")
    if yzColor:
        fig.colorbar(sc, ax=axYZ, label=label)
    axYZ.set_title("Y-Z cross-section (X = 0 plane)")
    axYZ.set_xlabel("Y (mm)")
    axYZ.set_ylabel("Z (mm)")
    axYZ.set_aspect("equal")
    axYZ.grid(True)

    fig.suptitle("Tool Tip Workspace"
                 + (f" - coloured by tool tilt {TILT_METRIC}" if TILT_METRIC else ""))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
