"""
test_arc_kinematics.py - Validates the single-section arc math and the full
IK -> FK round trip in PCC_control.py.

Checks (all independent of the formulas under test):
  1. arcFK produces an orthonormal rotation matrix.
  2. arcFK tip position lies on the circle of radius 1/k centred at
     (1/k)(cos r, sin r, 0) in the bend plane.
  3. arcFK z-column matches the numeric derivative dp/ds (the backbone
     tangent), i.e. the rotation really is the frame of the arc.
  4. arcFK -> arcIK recovers (k, r, s) exactly.
  5. FullIK -> forwardKinematics round trip: reachable tool poses generated
     by direct forward kinematics are recovered by the solver.

Run:  python test_arc_kinematics.py
"""
import math
import random

from PCC_control import ContinuumRobot

PI = math.pi
FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vnorm(a):
    return math.sqrt(sum(v * v for v in a))


def test_arcFK_properties(robot):
    print("arcFK geometric properties")
    worstOrtho = worstCircle = worstTangent = 0.0
    for th in [0.05, 0.3, 0.8, 1.2, PI / 2]:
        for s in [robot.sMin, 110.0, robot.sMax]:
            for r in [i * PI / 4 for i in range(-3, 5)]:
                k = th / s
                p, R = robot.arcFK(k, r, s)
                # 1. orthonormality of R
                for i in range(3):
                    for j in range(3):
                        dot = sum(R[a][i] * R[a][j] for a in range(3))
                        expect = 1.0 if i == j else 0.0
                        worstOrtho = max(worstOrtho, abs(dot - expect))
                # 2. tip on the bend circle
                c = [(1 / k) * math.cos(r), (1 / k) * math.sin(r), 0.0]
                worstCircle = max(worstCircle, abs(vnorm(vsub(p, c)) - 1 / k))
                # 3. z-column of R equals dp/ds (backbone tangent)
                h = 1e-4
                pa, _ = robot.arcFK(k, r, s - h)
                pb, _ = robot.arcFK(k, r, s + h)
                dp = [(pb[i] - pa[i]) / (2 * h) for i in range(3)]
                tan = [R[0][2], R[1][2], R[2][2]]
                worstTangent = max(worstTangent, vnorm(vsub(dp, tan)))
    check("rotation matrix orthonormal", worstOrtho < 1e-12, f"worst {worstOrtho:.2e}")
    check("tip lies on bend circle radius 1/k", worstCircle < 1e-9, f"worst {worstCircle:.2e} mm")
    check("R z-column = backbone tangent dp/ds", worstTangent < 1e-6, f"worst {worstTangent:.2e}")


def test_arc_roundtrip(robot):
    print("arcFK -> arcIK round trip")
    worst = 0.0
    for th in [0.06, 0.3, 0.8, 1.2, PI / 2]:
        for s in [robot.sMin, 110.0, robot.sMax]:
            for r in [i * PI / 4 for i in range(-3, 5)]:
                k = th / s
                p, _ = robot.arcFK(k, r, s)
                k2, r2, s2 = robot.arcIK(*p)
                dr = (r2 - r + PI) % (2 * PI) - PI
                worst = max(worst, abs(k2 - k), abs(dr), abs(s2 - s) / s)
    check("parameters recovered", worst < 1e-9, f"worst rel error {worst:.2e}")
    # straight singular branch
    k2, r2, s2 = robot.arcIK(0.0, 0.0, 120.0)
    check("straight singularity -> (0, 0, z)", (k2, r2, s2) == (0.0, 0.0, 120.0))


def toolPoseFromParams(robot, s1, th1, r1, th2, r2, s2):
    """Independent forward kinematics to a tooltip pose (mirrors workspace_sim)."""
    k1, k2 = th1 / s1, th2 / s2
    p1, R1 = robot.arcFK(k1, r1, s1)
    p2, R2 = robot.arcFK(k2, r2, s2)
    p = [p1[i] + sum(R1[i][j] * p2[j] for j in range(3)) for i in range(3)]
    t2 = [R2[0][2], R2[1][2], R2[2][2]]
    t = [sum(R1[i][j] * t2[j] for j in range(3)) for i in range(3)]
    tip = [p[i] + robot.toolLength * t[i] for i in range(3)]
    Rx = math.atan2(t[1], math.hypot(t[0], t[2]))
    Ry = math.atan2(t[0], t[2])
    return [tip[0], tip[1], tip[2], Rx, Ry, 0.0]


def test_fullIK_roundtrip():
    print("FullIK -> forwardKinematics round trip (200 random reachable targets)")
    rng = random.Random(42)
    posErrs, angErrs, clamped, failed = [], [], 0, 0
    for _ in range(200):
        robot = ContinuumRobot()
        # sample inside the limits, s2 from the mechanical coupling
        s1 = rng.uniform(robot.sMin + 1, robot.sMax - 1)
        s2 = robot.sectL + 0.5 * (s1 - robot.sectL)
        th1 = rng.uniform(0.05, robot.thetaMax - 0.05)
        th2 = rng.uniform(0.05, robot.thetaMax - 0.05)
        r1 = rng.uniform(-PI, PI)
        r2 = rng.uniform(-PI, PI)
        target = toolPoseFromParams(robot, s1, th1, r1, th2, r2, s2)
        robot.FullIK(target, [0.0] * 6)
        if robot.limitsHit:
            clamped += 1
        got = robot.forwardKinematics()
        pe = vnorm(vsub(got[:3], target[:3]))
        ae = math.hypot(got[3] - target[3], got[4] - target[4])
        posErrs.append(pe)
        angErrs.append(ae)
        if pe > 1.0 or ae > 0.02:
            failed += 1
    posErrs.sort()
    angErrs.sort()
    med_p, max_p = posErrs[len(posErrs) // 2], posErrs[-1]
    med_a, max_a = angErrs[len(angErrs) // 2], angErrs[-1]
    print(f"  position error  median {med_p:.2e} mm, max {max_p:.2e} mm")
    print(f"  tilt error      median {med_a:.2e} rad, max {max_a:.2e} rad")
    print(f"  solver hit limits on {clamped}/200 targets")
    check("round trip within 1 mm / 0.02 rad on >95% of targets", failed <= 10,
          f"{failed} targets outside tolerance")


def main():
    robot = ContinuumRobot()
    test_arcFK_properties(robot)
    test_arc_roundtrip(robot)
    test_fullIK_roundtrip()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    else:
        print("RESULT: all arc kinematics checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
