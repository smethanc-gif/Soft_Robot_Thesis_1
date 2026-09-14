"""
test_cable_lengths.py - Validates the cable length calculation in
PCC_control.PCC_Control against an independent geometric model.

Independent model: the backbone is a constant-curvature arc carrying
`nodes` disks per section. Each cable is anchored at radius robotRadius at
a fixed azimuth phi on every disk; the cable length is the sum of the
straight chords between consecutive anchor points. Disk poses come from
arcFK evaluated at partial arc lengths (arcFK itself is validated
separately in test_arc_kinematics.py).

The code's formula  l = 2n*sin(k*s/2n) * (1/k - d*sin(r + OFS + off))
implies cable azimuths  phi = pi/2 - OFS - off  (off = 0, pi/3, ... per
cable). This test checks that the code's cable lengths (recovered exactly
from its motor output, cable = init - motor*spoolRadius, which inverts
the code's own final conversion) match the chord-sum model.

Run:  python test_cable_lengths.py
"""
import math

from PCC_control import ContinuumRobot

PI = math.pi
FAILURES = []

# cable name -> (formula angular offset, spans both sections?)
CABLES = [("A", 0.0, True), ("B", PI / 3, False), ("C", 2 * PI / 3, True),
          ("D", PI, False), ("E", -2 * PI / 3, True), ("F", -PI / 3, False)]


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def sectionCableChordSum(robot, k, r, s, phi, nSeg):
    """Chord-sum cable length for one section, cable at disk azimuth phi."""
    d = robot.robotRadius
    ax, ay = d * math.cos(phi), d * math.sin(phi)
    total, prev = 0.0, None
    for j in range(nSeg + 1):
        p, R = robot.arcFK(k, r, s * j / nSeg)
        a = [p[i] + R[i][0] * ax + R[i][1] * ay for i in range(3)]
        if prev is not None:
            total += math.dist(a, prev)
        prev = a
    return total


def cableLengthsNumeric(robot, k1, r1, s1, k2, r2, s2):
    """Independent full-robot cable lengths in code order [A..F]."""
    n = robot.nodes
    out = []
    for _, off, spansBoth in CABLES:
        phi = PI / 2 - robot.rotationalOffset - off
        l = sectionCableChordSum(robot, k1, r1, s1, phi, n)
        if spansBoth:
            l += sectionCableChordSum(robot, k2, r2, s2, phi, n)
        out.append(l)
    return out


def cableLengthsFromCode(robot, k1, r1, s1, k2, r2, s2):
    """Drive the real PCC_Control code path at exact arc parameters and
    recover its internal cable lengths from the motor output."""
    p1, R1 = robot.arcFK(k1, r1, s1)
    p2loc, _ = robot.arcFK(k2, r2, s2)
    p2 = [p1[i] + sum(R1[i][j] * p2loc[j] for j in range(3)) for i in range(3)]
    robot.targMidGlobPos = [p1[0], p1[1], p1[2], 0, 0, 0]
    robot.targGlobPos = [p2[0], p2[1], p2[2], 0, 0, 0]
    motors, _ = robot.PCC_Control()
    if robot.limitsHit:
        raise ValueError(f"test config clamped by limits: {robot.limitsHit}")
    return [robot.initCableLen[i] - motors[i] * robot.spoolRadius
            for i in range(6)]


def test_cable_formula():
    print("cable length formula vs chord-sum geometric model")
    robot = ContinuumRobot()
    worst, worstCfg = 0.0, None
    cases = 0
    for th1 in [0.0, 0.2, 0.7, 1.2, PI / 2 - 0.01]:
        for th2 in [0.0, 0.3, 0.9, PI / 2 - 0.01]:
            for r1 in [i * PI / 3 + 0.1 for i in range(-2, 3)]:
                for r2 in [-1.7, 0.4, 2.6]:
                    for s1 in [robot.sMin + 0.5, 0.5 * (robot.sMin + robot.sMax),
                               robot.sMax - 0.5]:
                        s2 = robot.sectL + 0.5 * (s1 - robot.sectL)
                        k1 = th1 / s1
                        k2 = th2 / s2
                        code = cableLengthsFromCode(robot, k1, r1, s1, k2, r2, s2)
                        ref = cableLengthsNumeric(robot, k1, r1, s1, k2, r2, s2)
                        cases += 1
                        for i, (name, _, _) in enumerate(CABLES):
                            e = abs(code[i] - ref[i])
                            if e > worst:
                                worst, worstCfg = e, (name, th1, th2, r1, s1)
    check(f"all 6 cables match over {cases} configs", worst < 1e-8,
          f"worst {worst:.2e} mm at cable {worstCfg[0]}" if worstCfg else "")


def test_straight_config():
    print("straight full-length config")
    robot = ContinuumRobot()
    code = cableLengthsFromCode(robot, 0.0, 0.0, robot.sectL, 0.0, 0.0, robot.sectL)
    expect = robot.initCableLen
    worst = max(abs(code[i] - expect[i]) for i in range(6))
    check("cables at nominal lengths (motors zero)", worst < 1e-9,
          f"worst {worst:.2e} mm")


def test_bend_direction():
    print("bend-direction sanity")
    robot = ContinuumRobot()
    # bend section 1 toward cable A's azimuth (phi_A = pi/2 - OFS) -> cable A
    # should be the shortest section-1 cable, its antagonist D the longest
    rA = PI / 2 - robot.rotationalOffset
    # sectL - 0.5 rather than sectL: arcIK round-trip float noise puts s a
    # hair above sMax, which _applyLimits then "clamps" (s 148.5 -> 148.5)
    s1 = robot.sectL - 0.5
    s2 = robot.sectL - 0.5
    code = cableLengthsFromCode(robot, 1.0 / s1, rA, s1, 0.0, 0.0, s2)
    sect1Part = [code[i] - (s2 if CABLES[i][2] else 0.0) for i in range(6)]
    check("cable A shortest when bending toward A", min(sect1Part) == sect1Part[0],
          f"A={sect1Part[0]:.2f} mm")
    check("cable D longest when bending toward A", max(sect1Part) == sect1Part[3],
          f"D={sect1Part[3]:.2f} mm")
    # antagonist pairs through section 1 sum to a d-independent constant
    pairSums = [sect1Part[0] + sect1Part[3], sect1Part[1] + sect1Part[4],
                sect1Part[2] + sect1Part[5]]
    spread = max(pairSums) - min(pairSums)
    check("antagonist pair sums equal (d cancels)", spread < 1e-9,
          f"spread {spread:.2e} mm")


def main():
    test_cable_formula()
    test_straight_config()
    test_bend_direction()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    else:
        print("RESULT: all cable length checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
