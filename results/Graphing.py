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

DESIRED_FILE = "PCC/Test_46/DesiredMotorPosition.txt"#"DesiredPosition.txt"
CURRENT_FILE = "PCC/Test_46/CurrentMotorPosition.txt"#"CurrentPosition.txt"

LINE_PATTERN = re.compile(r"t=(-?[\d.]+)s\s+\[([^\]]+)\]")


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

def EgoGraph(test: int, ctrl="PCC", _DESIRED_FILE="DesiredMotorPosition.txt", _CURRENT_FILE="CurrentMotorPosition.txt"):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + _DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + _CURRENT_FILE
    desired_times, desired_vals = parse_log(DESIRED_LOCATION)
    current_times, current_vals = parse_log(CURRENT_LOCATION)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motor_names = ["A", "B", "C", "D", "E", "F"]

    for i in range(6):
        ax = axes[i]
        ax.plot(desired_times, desired_vals[i], label="Desired", linestyle="--", color="tab:blue")
        ax.plot(current_times, current_vals[i], label="Actual", color="tab:orange")
        ax.set(ylim=[0, 6])
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Desired vs Actual Motor Positions")
    print(f"Graphing Motor Data... \nTest {test} for {ctrl} control.\n")
    fig.tight_layout()
    plt.show()

def GlobGraph(test: int, ctrl="PCC", _DESIRED_FILE="DesiredTipPosition.txt", _CURRENT_FILE="CurrentTipPosition.txt"):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + _DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + _CURRENT_FILE
    desired_times, desired_vals = parse_log(DESIRED_LOCATION)
    current_times, current_vals = parse_log(CURRENT_LOCATION)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motor_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]

    for i in range(6):
        ax = axes[i]
        ax.plot(desired_times, desired_vals[i], label="Desired", linestyle="--", color="tab:blue")
        ax.plot(current_times, current_vals[i], label="Actual", color="tab:orange")
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (mm)")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Desired vs Actual Tip Positions")
    fig.tight_layout()
    print(f"Graphing Tip Data... \nTest {test} for {ctrl} control.\n")
    plt.show()
def main():
    exampleTest = 57
    EgoGraph(test = exampleTest)
    #GlobGraph(test = exampleTest)


if __name__ == "__main__":
    main()
