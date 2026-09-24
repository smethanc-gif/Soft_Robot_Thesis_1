import cv2
import numpy as np
import pyrealsense2 as rs

# Created by Ethan 12/3/26
# Updated 17/7/26 - RealSense depth + ArUco global reference frame
# Updated 17/7/26 - locked colour sensor, gentler mask cleanup, frozen marker
#                   pose, smoothed output; refactored into TipTracker class

# ============================================================
# CONFIG - everything you should need to touch lives here
# ============================================================

# --- Colour segmentation limits ---
# Measured from the ball under LOCKED exposure/white balance (exp 156, wb 4600).
# Note: limits tuned with hsv_tuner.py (auto exposure) won't match this stream.
# High S min is what keeps the cardboard/desk/clothing out of the mask.
#HSV_LOWER = np.array([15, 81, 71])
#HSV_UPPER = np.array([32, 234, 255])
#HSV_LOWER = np.array([15, 130, 50])
#HSV_UPPER = np.array([45, 255, 255])
    ## Green ball tip:
#HSV_LOWER = np.array([22, 131, 83])
#HSV_UPPER = np.array([32, 255, 255])
    ## Matt red tip:
##HSV_LOWER = np.array([0, 129, 141])
#HSV_UPPER = np.array([8, 255, 255])
    ## MATT red tip w D455F
#HSV_LOWER = np.array([0, 126, 131])
#HSV_UPPER = np.array([10, 229, 255])
    ## Florescent Green w D455F
HSV_LOWER = np.array([15, 130, 148])
HSV_UPPER = np.array([35, 255, 255])
# --- Mask cleanup ---
# Open removes small specks, close fills small holes in the blob.
OPEN_KERNEL = 0
CLOSE_KERNEL = 0
MIN_BLOB_AREA = 0   # ignore blobs smaller than this (px) - noise rejection

# --- Colour sensor locking ---
# Auto exposure/white balance hunting shifts the hue of the tip and makes the
# mask flicker, so we let them settle for a moment and then freeze them.
LOCK_COLOR_SENSOR = True
SETTLE_FRAMES = 30        # frames of auto exposure before locking
MANUAL_EXPOSURE = 166.0    # e.g. 156.0 to force a value, None = lock the settled value
MANUAL_WHITE_BALANCE = 4600  # e.g. 4600.0, None = lock the settled value

# --- ArUco marker (defines the global reference frame) ---
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # dictionary the marker was generated from
MARKER_ID = 0                       # ID of the marker to look for
MARKER_SIZE_MM = 121.0              # printed side length of the black square - measure it!

# Camera and marker are both fixed, so the pose is averaged over this many
# detections at startup and then frozen - re-solving it every frame just adds
# noise to every coordinate.
POSE_SAMPLES = 30

# Position of the global origin relative to the MARKER CENTRE, in marker-frame mm.
# Marker frame: X = right along the marker, Y = up along the marker, Z = out of
# the marker towards the camera. Leave as zeros to use the marker centre itself.
# e.g. if the workspace corner is 100mm to the right of the marker centre:
# ORIGIN_OFFSET_MM = np.array([100.0, 0.0, 0.0])
ORIGIN_OFFSET_MM = np.array([117.75, 95.25, 0.0])

# --- Output smoothing ---
# Exponential moving average: weight of the newest reading (1.0 = no smoothing,
# smaller = smoother but laggier).
# Kept at 1.0 so the logged tip position has no filter lag - the phase-lag /
# speed study relies on this (0.3 added ~0.1s of apparent lag). Smooth offline.
SMOOTHING = 1.0

# --- RealSense stream settings ---
FRAME_W, FRAME_H, FPS = 640, 480, 30

# Half-width of the pixel window sampled for depth around the tip centroid
DEPTH_WIN = 1
#DEPTH_ERODE_PX = 2

# ============================================================
# Tracker
# ============================================================

class TipTracker:
    """Tracks the robot tip in 3D relative to an ArUco marker.

    Use from another script:
        tracker = TipTracker()
        while ...:
            result = tracker.update()
            if result["position"] is not None:
                x, y, z = result["position"]
        tracker.stop()

    Or run this file directly for the live viewer windows.
    """

    def __init__(self):
        self.pose_rvecs = []
        self.pose_tvecs = []
        self.pose_locked = False
        self.marker_R = None
        self.marker_t = None
        self.rvec_avg = None
        self.tvec_avg = None
        self.tip_smooth = None
        self.position = None
        self.open_kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
        self.close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)

        half = MARKER_SIZE_MM / 2
        self.marker_obj_points = np.array([[-half,  half, 0],
                                           [ half,  half, 0],
                                           [ half, -half, 0],
                                           [-half, -half, 0]], dtype=np.float32)

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv2.aruco.DetectorParameters())

        self._start_camera()
        if LOCK_COLOR_SENSOR:
            self._lock_colour_sensor()

        #self.depth_erode_kernel = np.ones((DEPTH_ERODE_PX, DEPTH_ERODE_PX), np.uint8)
    # ---------------- camera ----------------

    def _start_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)
        profile = self.pipeline.start(config)

        # Align depth frames onto the colour frames
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.color_sensor = profile.get_device().first_color_sensor()

        # Colour camera intrinsics (factory calibrated - no checkerboard needed)
        self.intr = (profile.get_stream(rs.stream.color)
                     .as_video_stream_profile().get_intrinsics())
        self.camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]])
        self.dist_coeffs = np.array(self.intr.coeffs)

        # Let the stream actually start before touching sensor options
        for _ in range(10):
            self.pipeline.wait_for_frames()

        # Stop auto exposure dropping the frame rate below FPS in dim light
        self.color_sensor.set_option(rs.option.auto_exposure_priority, 0)

    def _lock_colour_sensor(self):
        # Let auto exposure/white balance settle, then freeze them so the tip
        # colour stays inside the HSV band
        print(f"letting exposure settle for {SETTLE_FRAMES} frames...")
        for _ in range(SETTLE_FRAMES):
            self.pipeline.wait_for_frames()
        exposure = MANUAL_EXPOSURE or self.color_sensor.get_option(rs.option.exposure)
        white_bal = (MANUAL_WHITE_BALANCE
                     or self.color_sensor.get_option(rs.option.white_balance))
        self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.color_sensor.set_option(rs.option.exposure, exposure)
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        self.color_sensor.set_option(rs.option.white_balance, white_bal)
        print(f"locked exposure={exposure:.0f} white_balance={white_bal:.0f}")

    def get_frames(self):
        # Returns (colour image, depth image) or (None, None) if a frame dropped
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return (np.asanyarray(color_frame.get_data()),
                np.asanyarray(depth_frame.get_data()))

    # ---------------- marker pose ----------------

    def update_marker_pose(self, frame):
        # Accumulate detections until POSE_SAMPLES, then average and freeze
        if self.pose_locked:
            return
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if marker_id == MARKER_ID:
                    ok, rvec, tvec = cv2.solvePnP(
                        self.marker_obj_points, marker_corners.reshape(4, 2),
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        self.pose_rvecs.append(rvec)
                        self.pose_tvecs.append(tvec)
        if len(self.pose_rvecs) >= POSE_SAMPLES:
            self.rvec_avg = np.mean(self.pose_rvecs, axis=0)
            self.tvec_avg = np.mean(self.pose_tvecs, axis=0)
            self.marker_R, _ = cv2.Rodrigues(self.rvec_avg)
            self.marker_t = self.tvec_avg.flatten()
            self.pose_locked = True
            print(f"\nmarker pose locked from {POSE_SAMPLES} samples")

    # ---------------- segmentation ----------------

    def segment_tip(self, frame):
        # Returns (mask, centroid) - centroid is None if no valid blob
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Clean out the noise - open drops specks, close fills holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        # Largest blob that is big enough to be the tip
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_BLOB_AREA]
        if not contours:
            return mask, None
        
        M = cv2.moments(max(contours, key=cv2.contourArea))
        if M["m00"] == 0:
            return mask, None
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        return mask, centroid

    def tip_depth_mm(self, depth_image, cx, cy):
        window = depth_image[max(cy - DEPTH_WIN, 0):cy + DEPTH_WIN + 1, max(cx - DEPTH_WIN, 0):cx + DEPTH_WIN +1]
        # Median of valid depth pixels in a small window - single pixels are noisy

        valid = window[window > 0]
        if valid.size == 0:
            return None
        depth = float((np.median(valid)) * self.depth_scale * 1000)
        #print(f"Tip depth from camera: {depth}mm\n")
        return depth

    # ---------------- 3D position ----------------

    def to_global(self, cx, cy, d_mm):
        # Pixel + depth -> camera frame -> marker frame -> global frame (mm)
        p_cam = np.array(rs.rs2_deproject_pixel_to_point(
            self.intr, [cx, cy], d_mm / 1000)) * 1000
        p_marker = self.marker_R.T @ (p_cam - self.marker_t)
        return p_marker - ORIGIN_OFFSET_MM

    def update(self):
        """Process one frame. Returns a dict with:
        position - smoothed np.array [x, y, z] in mm, or None
        status   - human-readable state string
        frame    - colour image (for display)
        mask     - segmentation mask (for display)
        centroid - tip pixel (cx, cy), or None
        """
        frame, depth_image = self.get_frames()
        if frame is None:
            return {"position": None, "status": "frame dropped",
                    "frame": None, "mask": None, "centroid": None}

        self.update_marker_pose(frame)
        mask, centroid = self.segment_tip(frame)

        position = None
        status = "no tip found"
        if centroid is not None:
            cx, cy = centroid
            d_mm = self.tip_depth_mm(depth_image, cx, cy)
            if d_mm is None:
                status = "tip found, no depth data"
            elif not self.pose_locked:
                status = f"locking marker pose {len(self.pose_rvecs)}/{POSE_SAMPLES}"
            else:
                p_global = self.to_global(cx, cy, d_mm)

                # Smooth the output
                if self.tip_smooth is None:
                    self.tip_smooth = p_global
                else:
                    self.tip_smooth = (SMOOTHING * p_global
                                       + (1 - SMOOTHING) * self.tip_smooth)
                position = self.tip_smooth
                x, y, z = position
                status = f"X {x:7.1f}  Y {y:7.1f}  Z {z:7.1f} mm"
                
        return {"position": position, "status": status,
                "frame": frame, "mask": mask, "centroid": centroid}

    def stop(self):
        self.pipeline.stop()


    # ---------------- live viewer ----------------

    def annotate(self, result):
        frame = result["frame"]
        if result["centroid"] is not None:
            cv2.circle(frame, result["centroid"], 5, (0, 255, 0), -1)
        if self.marker_R is not None:
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                              self.rvec_avg, self.tvec_avg, MARKER_SIZE_MM * 0.75)
        if not self.pose_locked:
            cv2.putText(frame,
                        f"locking marker pose {len(self.pose_rvecs)}/{POSE_SAMPLES}",
                        (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        cv2.putText(frame, result["status"], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        # Live viewer loop - q to quit
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            result = self.update()
            if result["frame"] is None:
                continue
            if result["position"] is not None:
                print(result["status"], end="\r")

            self.annotate(result)
            cv2.imshow("frame", result["frame"])
            cv2.imshow("mask", result["mask"])
    def getPoseLocked(self) -> bool:
        return self.pose_locked

if __name__ == "__main__":
    tracker = TipTracker()
    try:
        tracker.run()
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
