'''Quick plots of one test's logs: desired vs actual motor positions (or velocities), desired vs
actual tip position over time and in 3D, and the tip error over time. Also loop-timing statistics.

Run from anywhere:  python Results/analysis/graphing.py   (plots the test in main())
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so this file can be run directly
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
from Results.logReader import parseLog, getLogPath


def plotEgoTracking(_test: int, _ctrl: str = "PCC", _vel: bool = False):
    '''Desired vs actual position (and their difference) of each motor over time, or the
        per-sample velocities with _vel=True. Prints the loop-timing statistics.'''
    desiredTimes, desiredPos = parseLog(getLogPath(_test, _ctrl, "desiredEgo"))
    currentTimes, currentPos = parseLog(getLogPath(_test, _ctrl, "currentEgo"))
    errPos = np.array(desiredPos) - np.array(currentPos)
    desiredVel = np.zeros((len(desiredPos[0]),6))
    currentVel = np.zeros((len(desiredPos[0]),6))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    motorNames = ["A", "B", "C", "D", "E", "F"]
    for i in range(6):
        for j in range(len(desiredPos[0])-1):
            desiredVel[j+1,i] = desiredPos[i][j+1] - desiredPos[i][j]
            currentVel[j+1,i] = currentPos[i][j+1] - currentPos[i][j]

    for i in range(6):
        ax = axes[i]
        if _vel:
            ax.plot(desiredTimes, desiredVel[:,i], label="Desired Velocity", linestyle="--", color="tab:purple")
            ax.plot(currentTimes, currentVel[:,i], label="Actual", color="tab:red")
            ax.set_ylabel("Velocity (rad/s))")
        else:
            ax.plot(desiredTimes, errPos[i], label="Position Error", linestyle=":", color="tab:green")
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
            ax.set_ylabel("Position (rad)")
        ax.set_title(f"{motorNames[i]}")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True)
    if _vel:
        fig.suptitle("Desired vs Actual Motor Velocities")
    else:
        fig.suptitle("Desired vs Actual Motor Positions")
    print(f"Graphing Motor Data... \nTest {_test} for {_ctrl} control.\n")
    calcTimeStatistics(desiredTimes)
    fig.tight_layout()
    plt.show()


def plotGlobTracking(_test: int, _ctrl: str = "PCC", _vel: bool = False):
    '''Desired vs actual tip X, Y and Z over time (or per-sample differences with _vel=True).'''
    desiredTimes, desiredPos = parseLog(getLogPath(_test, _ctrl, "desiredGlob"))
    currentTimes, currentPos = parseLog(getLogPath(_test, _ctrl, "currentGlob"))
    desiredVel = np.zeros((len(desiredPos),6))
    currentVel = np.zeros((len(desiredPos),6))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    axisNames = ["X", "Y", "Z"]
    for i in range(3):
        for j in range(len(desiredPos)-1):
            desiredVel[j+1,i] = desiredPos[j+1][(i)] - desiredPos[j][i]
            currentVel[j+1, i] = currentPos[j+1][i] - currentPos[j][i]

    for i in range(3):
        ax = axes[i]
        if _vel:
            ax.plot(desiredTimes, desiredVel[i], label="Desired Velocity", linestyle="--", color="tab:purple")
            ax.plot(currentTimes, currentVel[i], label="Actual", color="tab:red")
        else:
            ax.plot(desiredTimes, desiredPos[i], label="Desired Position", linestyle="--", color="tab:blue")
            ax.plot(currentTimes, currentPos[i], label="Actual Position", color="tab:orange")
        ax.set_title(f"{axisNames[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (mm)")
        ax.legend()
        ax.grid(True)

    if _vel:
        fig.suptitle("Desired vs Actual Tip Velocities")
    else:
        fig.suptitle("Desired vs Actual Tip Positions")
    fig.tight_layout()
    print(f"Graphing Tip Data... \nTest {_test} for {_ctrl} control.\n")
    plt.show()


def plotGlobTracking3D(_test: int, _ctrl: str = "PCC"):
    '''Desired, actual and model-calculated tip paths in 3D above the base plane.'''
    _, desiredPos = parseLog(getLogPath(_test, _ctrl, "desiredGlob"))
    _, currentPos = parseLog(getLogPath(_test, _ctrl, "currentGlob"))
    _, calculatedPos = parseLog(getLogPath(_test, _ctrl, "calculatedGlob"))
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
    axes.scatter(0, 0, 320, color="black", s=40, label="Starting i")

    axes.set_xlim(-200, 200)
    axes.set_ylim(-200, 200)
    axes.set_zlim(0, 350)

    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Y (mm)")
    axes.set_zlabel("Z (mm)")
    axes.legend()
    plt.show()


def calcGlobErrorMetrics(_test: int, _ctrl: str = "PCC") -> dict:
    '''Cartesian tip-tracking error for one test: per-timestep error norm, RMSE, mean and max
        error, comparing desired vs actual tip position.'''
    desiredTimes, desiredPos = parseLog(getLogPath(_test, _ctrl, "desiredGlob"))
    _, currentPos = parseLog(getLogPath(_test, _ctrl, "currentGlob"))
    desired = np.array(desiredPos)[:3].T    # parseLog is axis-major [axis][sample] -> [sample][axis]
    current = np.array(currentPos)[:3].T
    err = np.linalg.norm(desired - current, axis=1)
    return {
        "times": desiredTimes,
        "error": err,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_error": float(np.mean(err)),
        "max_error": float(np.max(err)),
    }


def plotGlobError(_test: int, _ctrl: str = "PCC"):
    '''Error-norm-over-time plot for one test, with RMSE/max annotated.'''
    m = calcGlobErrorMetrics(_test, _ctrl)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m["times"], m["error"], color="tab:red")
    ax.axhline(m["rmse"], color="tab:blue", linestyle="--", label=f"RMSE = {m['rmse']:.2f} mm")
    ax.axhline(m["max_error"], color="tab:orange", linestyle=":", label=f"Max = {m['max_error']:.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cartesian tracking error (mm)")
    ax.set_title(f"Tip Tracking Error - {_ctrl} Test {_test}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def calcTimeStatistics(_timeSteps: list[float]) -> tuple[int, float, float, float]:
    '''Prints and returns the number of time steps and the mean, minimum and maximum step length of one test.'''
    deltaTime = np.zeros(len(_timeSteps))
    for i in range(1,len(_timeSteps)-2):
        deltaTime[i] = _timeSteps[i+1]-_timeSteps[i]
    nTimeSteps = len(deltaTime)
    meanTimeStep = stat.mean(deltaTime)
    minTimeStep = min(deltaTime)
    maxTimeStep = max(deltaTime)

    print(f"\n-----\nTIME STATISTICS\n-----\n\n")
    print(f"Total timesteps:        {nTimeSteps}\n")
    print(f"Mean Step Size:     {meanTimeStep}s\n")
    print(f"Min & Max Step Size:      {minTimeStep}s | {maxTimeStep}s")
    return nTimeSteps, meanTimeStep, minTimeStep, maxTimeStep


def printTimeStatistics(_testRange: list[int], _testPeriods=None, _ctrl: str = "PCC"):
    '''Loop-timing statistics over the tests in [_testRange[0], _testRange[1]): proportion of
        completed tests and the step time needed for 50-99 % of tests to never go overtime.'''
    if _testRange is None:
        print("Failure to provide testRange")
        return None

    rangeDiff = len(_testRange)
    timeStepN = np.zeros(rangeDiff)
    timeStepAvg = np.zeros(rangeDiff)
    timeStepMin = np.zeros(rangeDiff)
    timeStepMax = np.zeros(rangeDiff)
    for i in range(rangeDiff):
        
        times, _ = parseLog(getLogPath(_testRange[i], _ctrl, "currentEgo"))
        timeStepN[i], timeStepAvg[i], timeStepMin[i], timeStepMax[i] = calcTimeStatistics(times)
    print(f"\n------------------------------------------------------------------"+
          f"\n TIME STATISTICS FOR {_ctrl} TESTS IN RANGE: [{_testRange[0]} : ]" +
          f"\n------------------------------------------------------------------\n")

    numNorm = stat.NormalDist.from_samples(timeStepN)
    numSuccessfulTests = numNorm.cdf(300)
    maxNorm = stat.NormalDist.from_samples(timeStepMax)

    print(f"NUMBER OF TESTS ANALYSIS:")
    print(f"Proportion of Completed Tests: {numSuccessfulTests*100}%")

    print(f"\nMAX NORM ANALYSIS:")
    if _testPeriods is None:
        print(f"Suggested Time Range.\n")
        print(f"Time for 50% Success: {maxNorm.inv_cdf(0.5)}s")
        print(f"Time for 80% Success: {maxNorm.inv_cdf(0.8)}s")
        print(f"Time for 90% Success: {maxNorm.inv_cdf(0.9)}s")
        print(f"Time for 95% Success: {maxNorm.inv_cdf(0.95)}s")
        print(f"Time for 99% Success: {maxNorm.inv_cdf(0.99)}s")


def main():
    '''Plots one example test.'''
    PCCTests_Traj4 = [248, 256, 274, 249, 250, 251, 253, 267, 254, 255, 259, 260, 268]
    CableTests_Traj4 = [44, 43, 66, 44, 45, 46, 53, 47, 52, 68, 48, 51, 49, 50, 67, 59, 55, 57, 58]
    #printTimeStatistics(PCCTests_Traj4)
    #printTimeStatistics(CableTests_Traj4, _ctrl = "PCC_Cable")
    #plotEgoTracking(_test=exampleTest)
    #plotEgoTracking(_test=exampleTest, _vel=True)
    #plotGlobTracking3D(_test=exampleTest, _ctrl = "PCC")
    #plotGlobError(_test=exampleTest)
    plotEgoTracking(237)

if __name__ == "__main__":
    main()
