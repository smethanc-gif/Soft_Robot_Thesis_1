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
import numpy
import statistics as stat
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

def EgoGraph(test: int, ctrl="PCC", _DESIRED_FILE="DesiredMotorPosition.txt", _CURRENT_FILE="CurrentMotorPosition.txt", vel=False):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + _DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + _CURRENT_FILE
    desiredTimes, desiredPos = parse_log(DESIRED_LOCATION)
    currentTimes, currentPos = parse_log(CURRENT_LOCATION)
    desiredVel = numpy.zeros((len(desiredPos[0]),6))
    currentVel = numpy.zeros((len(desiredPos[0]),6))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motor_names = ["A", "B", "C", "D", "E", "F"]
    for i in range(6):
        for j in range(len(desiredPos[0])-1):
            desiredVel[j+1,i] = desiredPos[i][j+1] - desiredPos[i][j]
            currentVel[j+1,i] = currentPos[i][j+1] - currentPos[i][j]

    for i in range(6):
        ax = axes[i]
        
        if vel:
            ax.plot(desiredTimes, desiredVel[:,i], label="Desired Velocity", linestyle="--", color="tab:purple")
            ax.plot(currentTimes, currentVel[:,i], label="Actual", color="tab:red")
            ax.set_ylabel("Velocity (rad/s))")
        else:
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
            ax.set_ylabel("Position (rad)")
            #ax.set(ylim=, max(desiredPos[i])+1])
        
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        
        ax.legend()
        ax.grid(True)
    if vel:
        fig.suptitle("Desired vs Actual Motor Velocities")
    else:
        fig.suptitle("Desired vs Actual Motor Positions")
    print(f"Graphing Motor Data... \nTest {test} for {ctrl} control.\n")
    deltaTime = numpy.zeros(len(desiredTimes))
    for i in range(1,len(desiredTimes)-2):
        deltaTime[i] = desiredTimes[i+1]-desiredTimes[i]
    

    print(f"\n-----\nTIME STATISTICS\n-----\n\n")
    print(f"Total timesteps:        {len(deltaTime)}\n")
    print(f"Average Time Taken:     {stat.mean(deltaTime)}s\n")
    print(f"Standard Deviation:      {stat.stdev(deltaTime)}")
    fig.tight_layout()
    plt.show()

def GlobGraph(test: int, ctrl="PCC", _DESIRED_FILE="DesiredTipPosition.txt", _CURRENT_FILE="CurrentTipPosition.txt", vel=False):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + _DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + _CURRENT_FILE
    desiredTimes, desiredPos = parse_log(DESIRED_LOCATION)
    currentTimes, currentPos = parse_log(CURRENT_LOCATION)
    desiredVel = numpy.zeros((len(desiredPos),6))
    currentVel = numpy.zeros((len(desiredPos),6))   
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motor_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
    for i in range(6):
            for j in range(len(desiredPos)-1):
                desiredVel[j+1,i] = desiredPos[j+1][(i)] - desiredPos[j][i]
                currentVel[j+1, i] = currentPos[j+1][i] - currentPos[j][i]

    for i in range(6):
        ax = axes[i]
        if vel:
            ax.plot(desiredTimes, desiredVel[i], label="Desired Velocity", linestyle="--", color="tab:purple")
            ax.plot(currentTimes, currentVel[i], label="Actual", color="tab:red")
        else:
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (mm)")
        ax.legend()
        ax.grid(True)

    if vel:
        fig.suptitle("Desired vs Actual Tip Velocities")
    else:
        fig.suptitle("Desired vs Actual Tip Positions")
    fig.tight_layout()
    print(f"Graphing Tip Data... \nTest {test} for {ctrl} control.\n")
    plt.show()
def main():
    exampleTest = 65
    EgoGraph(test = exampleTest)
    #EgoGraph(test = exampleTest+1)
    #GlobGraph(test = exampleTest)


if __name__ == "__main__":
    main()
