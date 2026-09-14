"""
plot_positions.py — Plots desired vs. actual motor positions from log files.

Reads DesiredPosition.txt and CurrentPosition.txt (written by test_connection.py)
and produces a 6-panel figure, one subplot per motor, each showing the desired
and actual position curves over time.

Expected line format in both files:
  t=2.341s  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

Install dependency:
  pip install matplotlib
"""

import re
import matplotlib.pyplot as plt

TARGET_FILE = "TargetPosition.txt"
MIDPOINT_FILE = "MidPointPosition.txt"
CALCULATED_FILE = "CalculatedPosition.txt"

LINE_PATTERN = re.compile(r"t=([\d.]+)s\s+\[([^\]]+)\]")


def parse_log(filename: str):
    """
    Parse a log file into (times, [motor0_vals, motor1_vals, ..., motor5_vals]).
    """
    times = []
    motor_vals = [[] for _ in range(6)]

    with open(filename, "r") as f:
        for line in f:
            match = LINE_PATTERN.match(line.strip())
            if not match:
                continue

            t = float(match.group(1))
            vals = [float(v.strip()) for v in match.group(2).split(",")]

            if len(vals) != 6:
                continue  # skip malformed lines

            times.append(t)
            for i in range(6):
                motor_vals[i].append(vals[i])

    return times, motor_vals


def main():
    desired_times, desired_vals = parse_log(TARGET_FILE)
    current_times, current_vals = parse_log(CALCULATED_FILE)
    mid_times, mid_vals = parse_log(MIDPOINT_FILE)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motor_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]

    for i in range(6):
        ax = axes[i]
        ax.plot(desired_times, desired_vals[i], label="Target", linestyle="--", color="tab:blue")
        ax.plot(current_times, current_vals[i], label="Calculated", color="tab:red")
        ax.plot(mid_times, mid_vals[i], label="MidPoint", color="tab:orange")
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (mm/ rad)")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Tool Position")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
