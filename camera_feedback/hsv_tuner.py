import cv2
import numpy as np

# Written using Copilot 9/3/26
# Edited by Ethan C 10/3/26

def nothing(x):
    pass

# Open camera (change index if needed)
cap = cv2.VideoCapture(2)

# Create a window for trackbars
cv2.namedWindow("HSV Tuner")

# Create trackbars for lower/upper HSV
cv2.createTrackbar("H Min", "HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("H Max", "HSV Tuner", 179, 179, nothing)

cv2.createTrackbar("S Min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)

cv2.createTrackbar("V Min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read trackbar positions
    h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
    s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
    s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
    v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
    v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Print current bounds (optional)
    print("Lower:", lower, "Upper:", upper, end="\r")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()