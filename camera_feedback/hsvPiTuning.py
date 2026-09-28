"""
hsv_pi_tuning.py - HSV threshold tuner for the tip tracker, built for
tuning directly on the Raspberry Pi + RealSense rig.

Auto-exposure/white-balance (and even raw image quality over a
bandwidth-limited USB link) differ machine to machine - a band tuned
against a plain webcam capture on a different PC won't transfer here (see
robot_tracker.py's HSV_LOWER/HSV_UPPER comment). This script instead:

  - opens the RealSense colour+depth streams the same way robot_tracker.py
    does, so USB bandwidth/negotiation matches production
  - prints the negotiated USB connection type at startup - "2.x" means the
    link is bandwidth-limited and the image may be noisy/compressed no
    matter how HSV is tuned
  - locks exposure/white balance the same way robot_tracker.py does
    (pin MANUAL_EXPOSURE/MANUAL_WHITE_BALANCE below, or leave None to let
    auto settle then freeze it), so the band tuned here is tied to a
    specific, known, reproducible exposure
  - live H/S/V min/max trackbars over the same open/close mask cleanup
    robot_tracker.py applies, plus HSV-at-cursor on hover
  - prints a ready-to-paste HSV_LOWER/HSV_UPPER/MANUAL_EXPOSURE/
    MANUAL_WHITE_BALANCE block when you quit (q)

Install dependency:
  pip install opencv-python pyrealsense2 numpy
"""
import cv2
import numpy as np
import pyrealsense2 as rs

# --- Match robot_tracker.py, or override to explore ---
FRAME_W, FRAME_H, FPS = 640, 480, 30
OPEN_KERNEL = 3
CLOSE_KERNEL = 5
SETTLE_FRAMES = 30

# None = let auto exposure/white balance settle then freeze whatever it
# lands on this run. Set a number to force that value instead (e.g. once
# you've found a good pair on this rig and want repeatable tuning runs).
MANUAL_EXPOSURE = None
MANUAL_WHITE_BALANCE = None

# Starting HSV window for the sliders - whatever's currently in
# robot_tracker.py is a reasonable seed even if it needs adjusting here.
INIT_LOWER = [15, 130, 50]
INIT_UPPER = [35, 255, 255]


def nothing(_):
    pass


def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)
    profile = pipeline.start(config)

    dev = profile.get_device()
    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
    print(f"Camera: {dev.get_info(rs.camera_info.name)}  USB {usb_type}")
    if not usb_type.startswith("3"):
        print("WARNING: not on a USB 3 link - image quality/noise may be "
              "bandwidth-limited no matter how HSV is tuned. Try a "
              "different port/cable before spending time here.")

    color_sensor = dev.first_color_sensor()
    for _ in range(10):            # let the stream actually start
        pipeline.wait_for_frames()
    color_sensor.set_option(rs.option.auto_exposure_priority, 0)
    return pipeline, color_sensor


def lock_colour_sensor(pipeline, color_sensor):
    # Mirrors robot_tracker.py's _lock_colour_sensor exactly, so whatever
    # gets tuned here matches what production will actually see.
    print(f"letting exposure settle for {SETTLE_FRAMES} frames...")
    for _ in range(SETTLE_FRAMES):
        pipeline.wait_for_frames()
    exposure = MANUAL_EXPOSURE or color_sensor.get_option(rs.option.exposure)
    white_bal = MANUAL_WHITE_BALANCE or color_sensor.get_option(rs.option.white_balance)
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, exposure)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    color_sensor.set_option(rs.option.white_balance, white_bal)
    print(f"locked exposure={exposure:.0f} white_balance={white_bal:.0f}")
    return exposure, white_bal


def main():
    pipeline, color_sensor = start_camera()
    exposure, white_bal = lock_colour_sensor(pipeline, color_sensor)

    open_kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)

    cv2.namedWindow("Frame")
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H Min", "HSV Tuner", INIT_LOWER[0], 179, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", INIT_UPPER[0], 179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", INIT_LOWER[1], 255, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", INIT_UPPER[1], 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", INIT_LOWER[2], 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", INIT_UPPER[2], 255, nothing)

    mouse_pos = {"xy": None}
    cv2.setMouseCallback("Frame", lambda e, x, y, f, p: p.update(xy=(x, y)), mouse_pos)

    print("q to quit and print the final values; hover over 'Frame' to "
          "read the HSV under the cursor\n")

    lower = np.array(INIT_LOWER)
    upper = np.array(INIT_UPPER)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower = np.array([cv2.getTrackbarPos("H Min", "HSV Tuner"),
                              cv2.getTrackbarPos("S Min", "HSV Tuner"),
                              cv2.getTrackbarPos("V Min", "HSV Tuner")])
            upper = np.array([cv2.getTrackbarPos("H Max", "HSV Tuner"),
                              cv2.getTrackbarPos("S Max", "HSV Tuner"),
                              cv2.getTrackbarPos("V Max", "HSV Tuner")])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            if mouse_pos["xy"] is not None:
                mx, my = mouse_pos["xy"]
                if 0 <= my < hsv.shape[0] and 0 <= mx < hsv.shape[1]:
                    h, s, v = hsv[my, mx]
                    cv2.putText(frame, f"HSV @ cursor: {h}, {s}, {v}", (10, FRAME_H - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            print(f"HSV_LOWER = {list(lower)}  HSV_UPPER = {list(upper)}", end="\r")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n\n--- paste into robot_tracker.py ---")
        print(f"HSV_LOWER = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
        print(f"HSV_UPPER = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
        print(f"MANUAL_EXPOSURE = {exposure:.1f}")
        print(f"MANUAL_WHITE_BALANCE = {white_bal:.1f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
