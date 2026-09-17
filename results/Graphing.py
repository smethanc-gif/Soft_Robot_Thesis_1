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
    getTimeimeStatistics(desiredTimes)
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

def getTimeimeStatistics(timeSteps: list[float]):
    '''Returns: \nThe total timesteps, 
                \nThe Mean step length, 
                \nThe minimum step length, 
                \nAnd the maximum step length of a single test.'''
    
    deltaTime = numpy.zeros(len(timeSteps))
    for i in range(1,len(timeSteps)-2):
        deltaTime[i] = timeSteps[i+1]-timeSteps[i]
    _NoTimeSteps = len(deltaTime)
    _meanTimeSteps = stat.mean(deltaTime)
    _minTimeStep = min(deltaTime)
    _maxTimeStep = max(deltaTime)

        
    print(f"\n-----\nTIME STATISTICS\n-----\n\n")
    print(f"Total timesteps:        {_NoTimeSteps}\n")
    print(f"Mean Step Size:     {_meanTimeSteps}s\n")
    print(f"Min & Max Step Size:      {_minTimeStep}s | {_maxTimeStep}s")
    return _NoTimeSteps, _meanTimeSteps, _minTimeStep, _maxTimeStep

def timeStatistics(testRange: list[int], testPeriods = None, ctrl="PCC"):
    if testRange is None:
        print("Failure to provide testRange")
        return None
    
    rangeDiff = testRange[1] - testRange[0]
    timeStepN = numpy.zeros(rangeDiff)
    timeStepAvg = numpy.zeros(rangeDiff)
    timeStepMin = numpy.zeros(rangeDiff)
    timeStepMax = numpy.zeros(rangeDiff)
    for i in range(testRange[0], testRange[1]):
        point = i - testRange[0]
        FILE = f"{ctrl}/Test_{i}/CurrentMotorPosition.txt" 
        times, _ = parse_log(FILE)
        timeStepN[point], timeStepAvg[point], timeStepMin[point], timeStepMax[point] = getTimeimeStatistics(times)
    print(f"\n------------------------------------------------------------------"+
          f"\n TIME STATISTICS FOR {ctrl} TESTS IN RANGE: [{testRange[0]} : {testRange[1]}]" +
          f"\n------------------------------------------------------------------\n")
    
    
    numNorm = stat.NormalDist.from_samples(timeStepN)
    numSuccessfulTests = numNorm.cdf(300)
    maxNorm = stat.NormalDist.from_samples(timeStepMax)
    
    print(f"NUMBER OF TESTS ANALYSIS:")
    print(f"Proportion of Completed Tests: {numSuccessfulTests*100}%")


    print(f"\nMAX NORM ANALYSIS:")
    if testPeriods is None:
        print(f"Suggested Time Range.\n")
        print(f"Time for 50% Success: {maxNorm.inv_cdf(0.5)}s")
        print(f"Time for 80% Success: {maxNorm.inv_cdf(0.8)}s")
        print(f"Time for 90% Success: {maxNorm.inv_cdf(0.9)}s")
        print(f"Time for 95% Success: {maxNorm.inv_cdf(0.95)}s")
        print(f"Time for 99% Success: {maxNorm.inv_cdf(0.99)}s")
    

def main():
    exampleTest = 84
    #timeStatistics([60, exampleTest])
    #EgoGraph(test = exampleTest)
    EgoGraph(test = exampleTest, vel=True)
    #GlobGraph(test = exampleTest)


if __name__ == "__main__":
    main()
