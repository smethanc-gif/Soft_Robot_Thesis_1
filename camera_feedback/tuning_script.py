import cv2
import numpy as np

# Written by Ethan 9/3/26

def nothing(x):
    pass
# Establish Camera
capture = cv2.VideoCapture(2)

# Set Limits
# lower = np.array([163, 45, 101])
# higher = np.array([179, 255, 255])

# Create a window for trackbars
cv2.namedWindow("Noise Tuner")


# Create trackbars for lower/upper HSV
cv2.createTrackbar("Kernel Sz 1", "Noise Tuner", 1, 25, nothing)

cv2.createTrackbar("Erode 1", "Noise Tuner", 1, 5, nothing)

cv2.createTrackbar("Dilate 1", "Noise Tuner", 1, 5, nothing)

cv2.createTrackbar("Kernel Sz 2", "Noise Tuner", 1, 25, nothing)

cv2.createTrackbar("Erode 2", "Noise Tuner", 1, 5, nothing)

cv2.createTrackbar("Dilate 2", "Noise Tuner", 1, 5, nothing)

cv2.createTrackbar("Kernel Sz 3", "Noise Tuner", 1, 25, nothing)

cv2.createTrackbar("Erode 3", "Noise Tuner", 1, 5, nothing)

cv2.createTrackbar("Dilate 3", "Noise Tuner", 1, 5, nothing)

cv2.resizeWindow("Noise Tuner", 320, 150)

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

    # Get Data
    ret, frame = capture.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get Data out of the Wiindow
    kSz1 = cv2.getTrackbarPos("Kernel Sz 1", "Noise Tuner")

    eIt1 = cv2.getTrackbarPos("Erode 1", "Noise Tuner")
    dIt1 = cv2.getTrackbarPos("Dilate 1", "Noise Tuner")

    kSz2 = cv2.getTrackbarPos("Kernel Sz 2", "Noise Tuner")

    eIt2 = cv2.getTrackbarPos("Erode 2", "Noise Tuner")
    dIt2 = cv2.getTrackbarPos("Dilate 2", "Noise Tuner")

    kSz3 = cv2.getTrackbarPos("Kernel Sz 3", "Noise Tuner")

    eIt3 = cv2.getTrackbarPos("Erode 3", "Noise Tuner")
    dIt3 = cv2.getTrackbarPos("Dilate 3", "Noise Tuner")

    # Read trackbar positions
    h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
    s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
    s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
    v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
    v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # Inital Mask
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean out the noise

    kernel1 = np.ones((kSz1, kSz1), np.uint8)
    kernel2= np.ones((kSz2, kSz2), np.uint8)
    kernel3 = np.ones((kSz3, kSz3), np.uint8)

    mask = cv2.GaussianBlur(mask, (7,7), 0)

    mask = cv2.erode(mask, kernel1, iterations=eIt1)
    mask = cv2.dilate(mask, kernel1, iterations=dIt1)

    mask = cv2.erode(mask, kernel2, iterations=eIt2)
    mask = cv2.dilate(mask, kernel2, iterations=dIt2)

    mask = cv2.erode(mask, kernel3, iterations=eIt3)
    mask = cv2.dilate(mask, kernel3, iterations=dIt3)


    # Identify largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute centre
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Visualise
    cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("mask 2", mask2)
    # cv2.imshow("mask 3", mask3)
    cv2.imshow("mask", mask)

capture.release()
cv2.destroyAllWindows()
