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
import numpy as np
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D
_DESIRED_FILE = "PCC/Test_46/DesiredMotorPosition.txt"#"DesiredPosition.txt"
_CURRENTFILE = "PCC/Test_46/CurrentMotorPosition.txt"#"CurrentPosition.txt"

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

def EgoGraph(test: int, ctrl="PCC", __DESIRED_FILE="DesiredMotorPosition.txt", __CURRENTFILE="CurrentMotorPosition.txt", vel=False):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + __DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + __CURRENTFILE
    desiredTimes, desiredPos = parse_log(DESIRED_LOCATION)
    currentTimes, currentPos = parse_log(CURRENT_LOCATION)
    errPos = np.array(desiredPos) - np.array(currentPos)
    desiredVel = np.zeros((len(desiredPos[0]),6))
    currentVel = np.zeros((len(desiredPos[0]),6))
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
            ax.plot(desiredTimes, errPos[i], label="Position Error", linestyle=":", color="tab:green")
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
            ax.set_ylabel("Position (rad)")
            #ax.set(xlim = [0,2], ylim=[5, 5.6])
        
        ax.set_title(f"{motor_names[i]}")
        ax.set_xlabel("Time (s)")
        
        ax.legend()
        ax.grid(True)
    if vel:
        fig.suptitle("Desired vs Actual Motor Velocities")
    else:
        fig.suptitle("Desired vs Actual Motor Positions")
    print(f"Graphing Motor Data... \nTest {test} for {ctrl} control.\n")
    getTimeStatistics(desiredTimes)
    fig.tight_layout()
    plt.show()

def GlobGraph(test: int, ctrl="PCC", __DESIRED_FILE="DesiredTipPosition.txt", __CURRENTFILE="CurrentTipPosition.txt", vel=False):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + __DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + __CURRENTFILE
    desiredTimes, desiredPos = parse_log(DESIRED_LOCATION)
    currentTimes, currentPos = parse_log(CURRENT_LOCATION)
    desiredVel = np.zeros((len(desiredPos),6))
    currentVel = np.zeros((len(desiredPos),6))   
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    axis_names = ["X", "Y", "Z"]
    for i in range(3):
            for j in range(len(desiredPos)-1):
                desiredVel[j+1,i] = desiredPos[j+1][(i)] - desiredPos[j][i]
                currentVel[j+1, i] = currentPos[j+1][i] - currentPos[j][i]

    for i in range(3):
        ax = axes[i]
        if vel:
            ax.plot(desiredTimes, desiredVel[i], label="Desired Velocity", linestyle="--", color="tab:purple")
            ax.plot(currentTimes, currentVel[i], label="Actual", color="tab:red")
        else:
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
        ax.set_title(f"{axis_names[i]}")
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

def GlobGraph3D(test: int, ctrl="PCC", __DESIRED_FILE="DesiredTipPosition.txt", __CURRENTFILE="CurrentTipPosition.txt", __CALCULATEDFILE="CalculatedTipPosition.txt", vel=False):
    DESIRED_LOCATION = f"{ctrl}/Test_{test}/" + __DESIRED_FILE 
    CURRENT_LOCATION = f"{ctrl}/Test_{test}/" + __CURRENTFILE
    CALCULATED_LOCATION = f"{ctrl}/Test_{test}/" + __CALCULATEDFILE
    _, desiredPos = parse_log(DESIRED_LOCATION)
    _, currentPos = parse_log(CURRENT_LOCATION)
    _, calculatedPos = parse_log(CALCULATED_LOCATION)
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_subplot(111, projection='3d')
    axes.plot(desiredPos[0],desiredPos[1],desiredPos[2],label="Desired Position", linestyle="--", color="tab:blue")
    axes.plot(currentPos[0],currentPos[1],currentPos[2],label="Actual Position", color="tab:orange")
    axes.plot(calculatedPos[0],calculatedPos[1],calculatedPos[2],label="Calculated Position", color="tab:green")
    x = np.linspace(-200,200,20)
    y = np.linspace(-200,200,20)
    X, Y= np.meshgrid(x,y)
    Z = np.zeros_like(X+Y)
    axes.plot_surface(X,Y,Z,cmap="viridis", alpha=0.8)
    #axes.scatter(0, 0, 0, color="black", s=40, label="base")
    axes.scatter(0, 0, 320, color="black", s=40, label="Starting Point")

    axes.set_xlim(-200, 200)
    axes.set_ylim(-200, 200)
    axes.set_zlim(0, 350)

    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Y (mm)")
    axes.set_zlabel("Z (mm)")
    axes.legend()
    plt.show()

def GlobErrorMetrics(test: int, ctrl: str = "PCC", _DESIRED_FILE: str = "DesiredTipPosition.txt", _CURRENTFILE: str = "CurrentTipPosition.txt") -> dict:
    ''' Cartesian tip-tracking error for one test: per-timestep error norm,
        RMSE, mean, and max error, comparing desired vs actual tip position.'''
    desiredTimes, desiredPos = parse_log(f"{ctrl}/Test_{test}/{_DESIRED_FILE}")
    _, currentPos = parse_log(f"{ctrl}/Test_{test}/{_CURRENTFILE}")
    desired = np.array(desiredPos)[:3].T    # parse_log is axis-major [axis][sample] -> [sample][axis]
    current = np.array(currentPos)[:3].T
    err = np.linalg.norm(desired - current, axis=1)
    return {
        "times": desiredTimes,
        "error": err,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_error": float(np.mean(err)),
        "max_error": float(np.max(err)),
    }
def plotGlobError(test: int, ctrl: str = "PCC"):
    ''' Error-norm-over-time plot for one test, with RMSE/max annotated. '''
    m = GlobErrorMetrics(test, ctrl)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m["times"], m["error"], color="tab:red")
    ax.axhline(m["rmse"], color="tab:blue", linestyle="--", label=f"RMSE = {m['rmse']:.2f} mm")
    ax.axhline(m["max_error"], color="tab:orange", linestyle=":", label=f"Max = {m['max_error']:.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cartesian tracking error (mm)")
    ax.set_title(f"Tip Tracking Error - {ctrl} Test {test}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()

def getTimeStatistics(timeSteps: list[float]):
    '''Returns: \nThe total timesteps, 
                \nThe Mean step length, 
                \nThe minimum step length, 
                \nAnd the maximum step length of a single test.'''
    
    deltaTime = np.zeros(len(timeSteps))
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

def runTimeStatistics(testRange: list[int], testPeriods = None, ctrl="PCC"):
    if testRange is None:
        print("Failure to provide testRange")
        return None
    
    rangeDiff = testRange[1] - testRange[0]
    timeStepN = np.zeros(rangeDiff)
    timeStepAvg = np.zeros(rangeDiff)
    timeStepMin = np.zeros(rangeDiff)
    timeStepMax = np.zeros(rangeDiff)
    for i in range(testRange[0], testRange[1]):
        point = i - testRange[0]
        FILE = f"{ctrl}/Test_{i}/CurrentMotorPosition.txt" 
        times, _ = parse_log(FILE)
        timeStepN[point], timeStepAvg[point], timeStepMin[point], timeStepMax[point] = getTimeStatistics(times)
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
    exampleTest = 115
    
    #timeStatistics([60, exampleTest])
    EgoGraph(test = exampleTest)
    #EgoGraph(test = exampleTest, vel=True)
    #GlobGraph3D(test = exampleTest)
    #plotGlobError(test=exampleTest)


if __name__ == "__main__":
    main()
