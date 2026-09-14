"""
test_motor_positions.py - Validates the cable-length -> motor-position
conversion in PCC_control.PCC_Control and audits motor travel over the
traj_gen.py trajectory.

Physics: a spool of radius spoolRadius (= motorShaftDiameter/2) winds
spoolRadius mm of cable per radian of shaft rotation, so the shaft angle
needed to take up a cable length change dL is
    motor_rad = dL / spoolRadius              (radians)
    motor_rev = dL / (pi*motorShaftDiameter)  (full turns)

The conversion is checked fully independently: motor outputs from the real
FullIK pipeline are compared against (init - l_numeric)/spoolRadius, where
l_numeric comes from the chord-sum geometric cable model in
test_cable_lengths.py (no shared formulas with PCC_control's cable math).

The trajectory audit reports how much shaft travel the traj_gen.py path
actually demands and where it comes from (section length change vs
bending). Motor travel beyond 2*pi is expected and physical: the sections
are extensible, and one shaft turn only winds 2*pi*spoolRadius of cable.

Run:  python test_motor_positions.py
"""
import math

from PCC_control import ContinuumRobot
from test_cable_lengths import cableLengthsNumeric

PI = math.pi
FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def expectedMotors(robot):
    """Independent motor targets from the chord-sum cable model at the
    robot's last solved (and limit-clamped) arc parameters."""
    k1, r1, s1, k2, r2, s2 = robot.getCylPos()
    cables = cableLengthsNumeric(robot, k1, r1, s1, k2, r2, s2)
    return [(robot.initCableLen[i] - cables[i]) / robot.spoolRadius
            for i in range(6)]


def test_unit_conversion():
    print("motor output vs independent chord-sum model / spoolRadius")
    robot = ContinuumRobot()
    worst = 0.0
    for target in ([40.0, 25.0, 280.0, 0.0, 0.0, 0.0],
                   [-30.0, 50.0, 290.0, 0.2, -0.3, 0.0],
                   [0.0, 0.0, 300.0, 0.0, 0.0, 0.0],
                   [60.0, -20.0, 285.0, -0.4, 0.1, 0.0]):
        motors, _ = robot.FullIK(target, [0.0] * 6)
        expect = expectedMotors(robot)
        worst = max(worst, max(abs(motors[i] - expect[i]) for i in range(6)))
    check("shaft radians = cable take-up / spoolRadius", worst < 1e-8,
          f"worst error {worst:.2e} rad over 4 poses")


def trajectory_targets(nSamples):
    """The exact tooltip trajectory from traj_gen.py, sampled uniformly."""
    S, C = math.sin, math.cos
    acc = 200
    w = PI * 0.2 * acc
    for i in range(nSamples):
        t = 0.2 * i / (nSamples - 1)
        x, y, z = 0, 0, 250 + 40 * C(w * t)
        Rx = PI * C(w * t) * S(w * t / 8) / 3
        Ry = PI * S(w * t) * S(w * t / 8) / 3
        yield [x, y, z, Rx, Ry, 0.0]


def test_trajectory_audit():
    print("traj_gen.py trajectory audit")
    robot = ContinuumRobot()
    r_spool = robot.spoolRadius
    nSamples = 300
    maxMotor = [0.0] * 6
    maxContraction = [0.0] * 6
    maxBend = [0.0] * 6
    limitHits = 0
    worstModelErr = 0.0
    for target in trajectory_targets(nSamples):
        motors, _ = robot.FullIK(target, [0.0] * 6)
        if robot.limitsHit:
            limitHits += 1
        expect = expectedMotors(robot)
        worstModelErr = max(worstModelErr,
                            max(abs(motors[i] - expect[i]) for i in range(6)))
        _, _, s1, _, _, s2 = robot.getCylPos()
        for i in range(6):
            dL = motors[i] * r_spool                 # cable take-up in mm
            spansBoth = i in (0, 2, 4)               # cables A, C, E
            contr = (2 * robot.sectL - (s1 + s2)) if spansBoth else (robot.sectL - s1)
            maxMotor[i] = max(maxMotor[i], abs(motors[i]))
            maxContraction[i] = max(maxContraction[i], abs(contr) / r_spool)
            maxBend[i] = max(maxBend[i], abs(dL - contr) / r_spool)

    names = "ABCDEF"
    print(f"  {limitHits}/{nSamples} samples clamped by _applyLimits")
    print("  cable | shaft rad (turns)  | from s-change | from bending")
    for i in range(6):
        print(f"    {names[i]}   | {maxMotor[i]:7.2f}  ({maxMotor[i]/(2*PI):5.2f})  "
              f"|   {maxContraction[i]:7.2f}     |  {maxBend[i]:7.2f}")
    check("motors match independent model along whole trajectory",
          worstModelErr < 1e-8, f"worst error {worstModelErr:.2e} rad")
    worst = max(maxMotor)
    print(f"  note: peak demand {worst:.2f} rad = {worst/(2*PI):.2f} turns "
          f"({worst*r_spool:.1f} mm of cable; one turn winds {2*PI*r_spool:.1f} mm).")
    print("  Travel beyond 2*pi is physical - it comes mostly from section")
    print("  length change, not bending.")


def report_worst_case_budget():
    print("worst-case motor travel budget (from physical limits, informational)")
    robot = ContinuumRobot()
    r_spool = robot.spoolRadius
    s1 = robot.sMin
    s2 = robot.sectL + 0.5 * (s1 - robot.sectL)
    contrACE = 2 * robot.sectL - (s1 + s2)
    contrBDF = robot.sectL - s1
    bendPerSect = robot.thetaMax * robot.robotRadius
    combined = (contrACE + 2 * bendPerSect) / r_spool
    print(f"  contraction alone: A/C/E {contrACE:.1f} mm -> {contrACE/r_spool:.1f} rad "
          f"({contrACE/r_spool/(2*PI):.2f} turns), "
          f"B/D/F {contrBDF:.1f} mm -> {contrBDF/r_spool:.1f} rad")
    print(f"  full bend alone (both sections, toward cable): ~{2*bendPerSect:.1f} mm "
          f"-> {2*bendPerSect/r_spool:.1f} rad ({2*bendPerSect/r_spool/(2*PI):.2f} turns)")
    print(f"  combined worst case: {combined:.1f} rad = {combined/(2*PI):.2f} turns")


def main():
    test_unit_conversion()
    test_trajectory_audit()
    report_worst_case_budget()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    else:
        print("RESULT: all motor position checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
