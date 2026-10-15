'''Tracks the robot's tip in 3D with the RealSense camera: colour segmentation of the tip, its depth,
and an ArUco marker that defines the global reference frame. Used by main.py (TipTracker.update);
run this file directly for the live viewer windows.
'''
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
ORIGIN_OFFSET_MM = np.array([103.75, 95.25, 0.0])

# --- Output smoothing ---
# Exponential moving average: weight of the newest reading (1.0 = no smoothing,
# smaller = smoother but laggier).
# Kept at 1.0 so the logged tip position has no filter lag - the phase-lag /
# speed study relies on this (0.3 added ~0.1s of apparent lag). Smooth offline.
SMOOTHING = 1.0

# --- RealSense stream settings ---
FRAME_W, FRAME_H, FPS = 640, 480, 30

# --- Depth sampling ---
# Tip depth = median depth over the whole tip blob. Pixels this far in from the
# blob's edge are dropped first, since rim pixels often pick up the depth of the
# background behind the tip (0 = use the whole blob).
DEPTH_ERODE_PX = 2

# ============================================================
# Tracker
# ============================================================

class TipTracker:
    '''Tracks the robot tip in 3D relative to an ArUco marker.

    Use from another script:
        tracker = TipTracker()
        while ...:
            result = tracker.update()
            if result["position"] is not None:
                x, y, z = result["position"]
        tracker.stop()

    Or run this file directly for the live viewer windows.
    '''

    def __init__(self):
        '''Starts the camera (and locks its colour sensor); the marker pose is locked from the first POSE_SAMPLES detections.'''
        self.poseRvecs = []
        self.poseTvecs = []
        self.poseLocked = False
        self.markerR = None
        self.markerT = None
        self.rvecAvg = None
        self.tvecAvg = None
        self.tipSmooth = None
        self.position = None
        self.openKernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
        self.closeKernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
        self.depthErodeKernel = np.ones((2*DEPTH_ERODE_PX + 1, 2*DEPTH_ERODE_PX + 1), np.uint8)

        half = MARKER_SIZE_MM / 2
        self.markerObjPoints = np.array([[-half,  half, 0],
                                           [ half,  half, 0],
                                           [ half, -half, 0],
                                           [-half, -half, 0]], dtype=np.float32)

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv2.aruco.DetectorParameters())

        self._startCamera()
        if LOCK_COLOR_SENSOR:
            self._lockColourSensor()

    # ---------------- camera ----------------

    def _startCamera(self):
        '''Starts the colour + depth streams, aligned, and reads the factory intrinsics.'''
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)
        profile = self.pipeline.start(config)

        # Align depth frames onto the colour frames
        self.align = rs.align(rs.stream.color)
        self.depthScale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.colorSensor = profile.get_device().first_color_sensor()

        # Colour camera intrinsics (factory calibrated - no checkerboard needed)
        self.intr = (profile.get_stream(rs.stream.color)
                     .as_video_stream_profile().get_intrinsics())
        self.cameraMatrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]])
        self.distCoeffs = np.array(self.intr.coeffs)

        # Let the stream actually start before touching sensor options
        for _ in range(10):
            self.pipeline.wait_for_frames()

        # Stop auto exposure dropping the frame rate below FPS in dim light
        self.colorSensor.set_option(rs.option.auto_exposure_priority, 0)

    def _lockColourSensor(self):
        '''Lets auto exposure/white balance settle, then freezes them so the tip colour stays inside the HSV band.'''
        print(f"letting exposure settle for {SETTLE_FRAMES} frames...")
        for _ in range(SETTLE_FRAMES):
            self.pipeline.wait_for_frames()
        exposure = MANUAL_EXPOSURE or self.colorSensor.get_option(rs.option.exposure)
        whiteBal = (MANUAL_WHITE_BALANCE
                     or self.colorSensor.get_option(rs.option.white_balance))
        self.colorSensor.set_option(rs.option.enable_auto_exposure, 0)
        self.colorSensor.set_option(rs.option.exposure, exposure)
        self.colorSensor.set_option(rs.option.enable_auto_white_balance, 0)
        self.colorSensor.set_option(rs.option.white_balance, whiteBal)
        print(f"locked exposure={exposure:.0f} white_balance={whiteBal:.0f}")

    def getFrames(self):
        '''Returns (colour image, depth image), or (None, None) if a frame dropped.'''
        frames = self.align.process(self.pipeline.wait_for_frames())
        colorFrame = frames.get_color_frame()
        depthFrame = frames.get_depth_frame()
        if not colorFrame or not depthFrame:
            return None, None
        return (np.asanyarray(colorFrame.get_data()),
                np.asanyarray(depthFrame.get_data()))

    # ---------------- marker pose ----------------

    def updateMarkerPose(self, _frame):
        '''Accumulates marker detections until POSE_SAMPLES, then averages and freezes the marker pose.'''
        if self.poseLocked:
            return
        corners, ids, _ = self.detector.detectMarkers(_frame)
        if ids is not None:
            for markerCorners, markerId in zip(corners, ids.flatten()):
                if markerId == MARKER_ID:
                    ok, rvec, tvec = cv2.solvePnP(
                        self.markerObjPoints, markerCorners.reshape(4, 2),
                        self.cameraMatrix, self.distCoeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        self.poseRvecs.append(rvec)
                        self.poseTvecs.append(tvec)
        if len(self.poseRvecs) >= POSE_SAMPLES:
            self.rvecAvg = np.mean(self.poseRvecs, axis=0)
            self.tvecAvg = np.mean(self.poseTvecs, axis=0)
            self.markerR, _ = cv2.Rodrigues(self.rvecAvg)
            self.markerT = self.tvecAvg.flatten()
            self.poseLocked = True
            print(f"\nmarker pose locked from {POSE_SAMPLES} samples")

    # ---------------- segmentation ----------------

    def segmentTip(self, _frame):
        '''Returns (mask, centroid, blob) - blob is a filled mask of just the tip blob; centroid and
            blob are None if there's no valid blob.'''
        hsv = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Clean out the noise - open drops specks, close fills holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.openKernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.closeKernel)

        # Largest blob that is big enough to be the tip
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_BLOB_AREA]
        if not contours:
            return mask, None, None

        tip = max(contours, key=cv2.contourArea)
        M = cv2.moments(tip)
        if M["m00"] == 0:
            return mask, None, None
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [tip], -1, 255, thickness=cv2.FILLED)
        return mask, centroid, blob

    def calcTipDepth(self, _depthImage, _blob):
        '''Tip depth (mm): median of the valid depth pixels over the tip blob (depth is aligned to the
            colour frame, so the blob indexes it directly), rim eroded off. None if there's no depth data.'''
        core = cv2.erode(_blob, self.depthErodeKernel) if DEPTH_ERODE_PX > 0 else _blob
        valid = _depthImage[(core > 0) & (_depthImage > 0)]
        if valid.size == 0:     # blob too small to survive the erosion - use all of it
            valid = _depthImage[(_blob > 0) & (_depthImage > 0)]
        if valid.size == 0:
            return None
        depth = float(np.median(valid) * self.depthScale * 1000)
        return depth

    # ---------------- 3D position ----------------

    def convertToGlobal(self, _cx, _cy, _depthMm):
        '''Pixel + depth -> camera frame -> marker frame -> global frame (mm).'''
        pCam = np.array(rs.rs2_deproject_pixel_to_point(
            self.intr, [_cx, _cy], _depthMm / 1000)) * 1000
        pMarker = self.markerR.T @ (pCam - self.markerT)
        return pMarker - ORIGIN_OFFSET_MM

    def update(self):
        '''Processes one frame. Returns a dict with:
        position - smoothed np.array [x, y, z] in mm, or None
        status   - human-readable state string
        frame    - colour image (for display)
        mask     - segmentation mask (for display)
        centroid - tip pixel (cx, cy), or None
        '''
        frame, depthImage = self.getFrames()
        if frame is None:
            return {"position": None, "status": "frame dropped",
                    "frame": None, "mask": None, "centroid": None}

        self.updateMarkerPose(frame)
        mask, centroid, blob = self.segmentTip(frame)

        position = None
        status = "no tip found"
        if centroid is not None:
            cx, cy = centroid
            depthMm = self.calcTipDepth(depthImage, blob)
            if depthMm is None:
                status = "tip found, no depth data"
            elif not self.poseLocked:
                status = f"locking marker pose {len(self.poseRvecs)}/{POSE_SAMPLES}"
            else:
                pGlobal = self.convertToGlobal(cx, cy, depthMm)

                # Smooth the output
                if self.tipSmooth is None:
                    self.tipSmooth = pGlobal
                else:
                    self.tipSmooth = (SMOOTHING * pGlobal
                                       + (1 - SMOOTHING) * self.tipSmooth)
                position = self.tipSmooth
                x, y, z = position
                status = f"X {x:7.1f}  Y {y:7.1f}  Z {z:7.1f} mm"
                
        return {"position": position, "status": status,
                "frame": frame, "mask": mask, "centroid": centroid}

    def stop(self):
        '''Stops the camera stream.'''
        self.pipeline.stop()

    # ---------------- live viewer ----------------

    def annotate(self, _result):
        '''Draws the tip centroid, marker axes and status text onto the frame for the live viewer.'''
        frame = _result["frame"]
        if _result["centroid"] is not None:
            cv2.circle(frame, _result["centroid"], 5, (0, 255, 0), -1)
        if self.markerR is not None:
            cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs,
                              self.rvecAvg, self.tvecAvg, MARKER_SIZE_MM * 0.75)
        if not self.poseLocked:
            cv2.putText(frame,
                        f"locking marker pose {len(self.poseRvecs)}/{POSE_SAMPLES}",
                        (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        cv2.putText(frame, _result["status"], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        '''Live viewer loop - q to quit.'''
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
        '''True once the marker pose has been averaged and frozen.'''
        return self.poseLocked

if __name__ == "__main__":
    tracker = TipTracker()
    try:
        tracker.run()
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
