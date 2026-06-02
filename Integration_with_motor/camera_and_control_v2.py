"""
camera_and_control_v2.py

Live camera + ball detection + prediction + Arduino paddle control.

──────────────────────────────────────────────────────────────────────
v2 ADDS: COMMIT-ZONE STATE MACHINE
──────────────────────────────────────────────────────────────────────
The Director now runs a three-state state machine to prevent the
motor from being whipsawed by post-bounce or noise-driven detections:

    IDLE        — no active trajectory; waiting for a ball
    TRACKING    — ball moving toward rail; predicting + commanding
    COMMITTED   — ball past commit line; target FROZEN, no commands

The commit line sits at `--commit-fraction` of the play-area width
(default 0.60). Once the ball has been detected past that line on a
few consecutive frames, the Director sends one final motor command
to its best prediction, then stops issuing any further commands.
This means: if the ball bounces back off the paddle, no spurious
commands. If random hand motion appears in the frame, no spurious
commands. The only way out of COMMITTED is the ball being lost for
hold_after_lost_s seconds, OR a manual reset (spacebar).

This is the fix for "belt skips after ball bounces" type problems.

──────────────────────────────────────────────────────────────────────

NO REAL-WORLD UNITS REQUIRED. The system never needs to know how many
meters wide the rectangle is, because the only quantity that matters
in the camera->motor pipeline is steps-per-pixel, which is captured
directly during rail calibration:

    steps_per_pixel = |step_bot - step_top| / |y_bot_px - y_top_px|

Workflow:
    1. Press 'v', click two points on the rail in the camera image.
    2. Press 'h', click two points for the horizontal extent.
    3. Press 'r' to enter RAIL CALIBRATION mode. Arrow-key jog the
       paddle to the TOP of the rectangle, press 't'. Jog to the
       BOTTOM, press 'b'.
    4. Press 'k' to LOCK and start tracking.

Keyboard:
    v          define vertical (rail) line
    h          define horizontal extent
    r          enter rail-calibration mode
    t          (in rail-cal mode) capture top step count
    b          (in rail-cal mode) capture bottom step count
    UP/DOWN    jog paddle by 100 steps
    PgUp/PgDn  jog by 1000 steps
    HOME       goto step 0
    Z          zero paddle here
    k          lock and start live control
    u          unlock (back to setup)
    m          toggle recording (saves to recordings/rec_HHMM_MM_DD_YY.mp4)
    SPACE      reset Director state machine (force back to IDLE)
    q          quit
"""

import argparse
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from paddle_client import PaddleClient


# ============================================================================
# Globals
# ============================================================================
running = True
latest_raw = None
raw_lock = threading.Lock()
frame_seq = 0


def signal_handler(sig, frame):
    global running
    print("\nForce quit detected.")
    running = False
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# Recorder
#
# Records the on-screen display (post-overlay) to a video file at the
# CORRECT real-world playback speed. The trick:
#
#   1. cv2.VideoWriter requires a fixed declared fps at file-open time,
#      but we don't know our real frame-write rate until we're done.
#      Declaring a guess (e.g. "60 fps") and writing at a different
#      actual rate produces a sped-up or slowed-down playback.
#
#   2. Solution: write a temporary AVI with a placeholder fps, count
#      frames and elapsed time, and on stop() ALSO know the true fps
#      (frames / elapsed). Then remux the AVI to a final MP4 using
#      ffmpeg, telling ffmpeg the true fps. ffmpeg can do this with
#      stream-copy (-c copy) so there is NO re-encoding -- the JPEG
#      frames are repackaged into MP4 with a corrected timing track.
#      This is fast (a couple seconds for a long recording) and lossless.
#
#   3. If ffmpeg isn't available, we keep the AVI but warn the user.
#      The AVI still plays back, just at an incorrect speed; ffmpeg
#      can be run by hand later to fix it.
#
# Filename format: rec_HHMM_MM_DD_YY.mp4 (per user request)
# ============================================================================
class Recorder:
    # The placeholder fps written into the temp AVI. It just has to
    # be a legal positive number; the final MP4's fps is overwritten.
    _PLACEHOLDER_FPS = 30.0

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.writer: Optional[cv2.VideoWriter] = None
        self.tmp_path: Optional[Path] = None
        self.final_path: Optional[Path] = None
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self._ffmpeg_checked = False
        self._ffmpeg_available = False

    @property
    def is_active(self) -> bool:
        return self.writer is not None

    def _check_ffmpeg(self) -> bool:
        """Cache whether ffmpeg is on PATH. Only checked once."""
        if self._ffmpeg_checked:
            return self._ffmpeg_available
        import shutil
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        self._ffmpeg_checked = True
        return self._ffmpeg_available

    def _make_path(self) -> Tuple[Path, Path]:
        """Returns (tmp_avi_path, final_mp4_path).
        Format: rec_HHMM_MM_DD_YY.{avi,mp4}"""
        now = datetime.now()
        stamp = now.strftime("%H%M_%m_%d_%y")
        avi = self.output_dir / f"rec_{stamp}.avi"
        mp4 = self.output_dir / f"rec_{stamp}.mp4"
        return (avi, mp4)

    def start(self, frame_shape: Tuple[int, int], fps_hint: float = 30.0) -> bool:
        """Begin recording. The fps_hint is unused except as a placeholder
        in the AVI header; the real fps is measured at stop time and
        baked into the final MP4."""
        if self.is_active:
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path, self.final_path = self._make_path()
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(
            str(self.tmp_path), fourcc, self._PLACEHOLDER_FPS, (w, h))
        if not self.writer.isOpened():
            print(f"Failed to open video writer for {self.tmp_path}")
            self.writer = None
            self.tmp_path = None
            self.final_path = None
            return False
        self.frame_count = 0
        self.start_time = time.time()
        print(f"Recording started: {self.final_path}  ({w}x{h})")
        if not self._check_ffmpeg():
            print("  WARNING: ffmpeg not found on PATH. The recording")
            print("  will be saved as AVI with incorrect playback speed.")
            print("  Install ffmpeg (sudo apt install ffmpeg) for correct")
            print("  speed playback.")
        return True

    def write(self, frame) -> None:
        if not self.is_active or self.writer is None:
            return
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.writer.write(frame)
        self.frame_count += 1

    def stop(self) -> None:
        """Close the temp AVI, then remux to MP4 with the true fps so
        playback speed is correct."""
        if not self.is_active or self.writer is None:
            return
        try:
            self.writer.release()
        except Exception as exc:
            print(f"Recorder release error: {exc}")
        elapsed = time.time() - (self.start_time or time.time())
        # The "true" fps is the rate frames were actually fed to the
        # writer. This is what we want playback to use so 1 second of
        # recording is 1 second of playback.
        actual_fps = (self.frame_count / elapsed) if elapsed > 0 else self._PLACEHOLDER_FPS
        if actual_fps < 1.0:
            actual_fps = self._PLACEHOLDER_FPS

        tmp_path = self.tmp_path
        final_path = self.final_path
        frame_count = self.frame_count
        # Reset state before doing the (slow) remux so further calls
        # to start() can succeed even while we wait.
        self.writer = None
        self.tmp_path = None
        self.final_path = None
        self.frame_count = 0
        self.start_time = None

        print(f"Recording stopped.")
        print(f"  Wrote {frame_count} frames in {elapsed:.1f} s "
              f"(true fps {actual_fps:.2f})")

        if tmp_path is None or final_path is None:
            return

        if self._check_ffmpeg():
            self._remux(tmp_path, final_path, actual_fps)
        else:
            print(f"  Saved as AVI (no ffmpeg): {tmp_path}")
            print(f"  Playback will be incorrect speed. Fix with:")
            print(f"    ffmpeg -r {actual_fps:.3f} -i {tmp_path.name} "
                  f"-c copy {final_path.name}")

    @staticmethod
    def _remux(tmp_avi: Path, final_mp4: Path, true_fps: float) -> None:
        """Use ffmpeg to repackage the temp AVI as MP4 with the
        corrected fps. Stream-copy (-c copy) means no re-encoding;
        the JPEG-compressed video stream is just placed in a new
        container with a corrected timing track. Fast and lossless."""
        import subprocess
        # The -r before -i tells ffmpeg "interpret the input as having
        # this fps", which overrides the placeholder fps in the AVI
        # header. The -c copy then preserves the JPEG bitstream as-is.
        cmd = [
            "ffmpeg", "-y",
            "-r", f"{true_fps:.6f}",
            "-i", str(tmp_avi),
            "-c", "copy",
            str(final_mp4),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30.0)
            if result.returncode != 0:
                print(f"  ffmpeg failed (returncode={result.returncode}):")
                print(f"    {result.stderr.strip().splitlines()[-1] if result.stderr else ''}")
                print(f"  Keeping AVI: {tmp_avi}")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"  ffmpeg invocation error: {exc}")
            print(f"  Keeping AVI: {tmp_avi}")
            return

        # Success -- delete the temp AVI to keep the recordings folder tidy.
        try:
            tmp_avi.unlink()
        except OSError:
            pass
        print(f"  Saved: {final_mp4}")


# ============================================================================
# BallTracker (unchanged)
# ============================================================================
class BallTracker:
    def __init__(self, threshold_k=1.5, motion_thresh=18, min_area=50,
                 max_area=3000, lost_timeout=0.40, roi=None):
        self.threshold_k = threshold_k
        self.motion_thresh = motion_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.lost_timeout = lost_timeout
        self.roi = roi
        self.center = None
        self.radius = 0
        self.last_radius = 15
        self.last_seen_time = None
        self.prev_gray = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_bright_T = None
        self.lock = threading.Lock()

    def set_roi(self, roi):
        with self.lock:
            self.roi = roi

    def detect_ball(self, frame):
        try:
            if frame is None:
                return None, 0, None
            gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frame.shape) == 3 else frame)
            if self.roi is not None:
                xL, yT, xR, yB = self.roi
                xL = max(0, int(xL)); yT = max(0, int(yT))
                xR = min(gray.shape[1], int(xR))
                yB = min(gray.shape[0], int(yB))
                if xR <= xL or yB <= yT:
                    return None, 0, None
                gray_crop = gray[yT:yB, xL:xR]
                offset = (xL, yT)
            else:
                gray_crop = gray
                offset = (0, 0)
            gray_eq = self.clahe.apply(gray_crop)
            if self.prev_gray is None or self.prev_gray.shape != gray_eq.shape:
                self.prev_gray = gray_eq
                return None, 0, None
            diff = cv2.absdiff(gray_eq, self.prev_gray)
            self.prev_gray = gray_eq
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            _, motion = cv2.threshold(
                diff_blur, self.motion_thresh, 255, cv2.THRESH_BINARY)
            mean, std = cv2.meanStdDev(gray_eq)
            threshold = float(mean.item()) + self.threshold_k * float(std.item())
            threshold = max(60.0, min(240.0, threshold))
            self.last_bright_T = threshold
            _, bright = cv2.threshold(gray_eq, threshold, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_and(motion, bright)
            kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_slow = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_5, 1)
            mask_slow = cv2.morphologyEx(mask_slow, cv2.MORPH_CLOSE, kernel_5, 2)
            mask_fast = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_3, 1)
            mask_fast = cv2.dilate(mask_fast, kernel_3, iterations=1)
            contours_slow, _ = cv2.findContours(
                mask_slow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_fast, _ = cv2.findContours(
                mask_fast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_center = None; best_radius = 0; best_score = -1.0
            for contour in contours_slow:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > self.max_area:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.45:
                    continue
                (_, _), radius = cv2.minEnclosingCircle(contour)
                if radius <= 0:
                    continue
                m = cv2.moments(contour)
                if m["m00"] <= 0:
                    continue
                cx = int(m["m10"] / m["m00"]) + offset[0]
                cy = int(m["m01"] / m["m00"]) + offset[1]
                score = circularity * circularity * area
                if score > best_score:
                    best_score = score
                    best_center = (cx, cy); best_radius = int(radius)
            max_area_fast = self.max_area * 5
            for contour in contours_fast:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > max_area_fast:
                    continue
                rect = cv2.minAreaRect(contour)
                (rcx, rcy), (rw, rh), angle = rect
                minor_axis = min(rw, rh); major_axis = max(rw, rh)
                if major_axis <= 0 or minor_axis <= 0:
                    continue
                aspect_ratio = major_axis / minor_axis
                if minor_axis < 3 or minor_axis > 40:
                    continue
                if aspect_ratio < 1.5:
                    continue
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0:
                    continue
                solidity = area / hull_area
                if solidity < 0.3:
                    continue
                m = cv2.moments(contour)
                if m["m00"] <= 0:
                    continue
                cx = int(m["m10"] / m["m00"]) + offset[0]
                cy = int(m["m01"] / m["m00"]) + offset[1]
                expected_minor = 12.0
                minor_match = 1.0 / (1.0 + abs(minor_axis - expected_minor)
                                     / expected_minor)
                score = solidity * minor_match * area * 0.5
                if score > best_score:
                    best_score = score
                    best_center = (cx, cy)
                    best_radius = max(int(minor_axis / 2), 3)
            return best_center, best_radius, mask_fast
        except Exception as exc:
            print(f"detect_ball error: {exc}")
            return None, 0, None

    def update(self, center, radius, t):
        with self.lock:
            if center is not None:
                self.center = center
                self.radius = radius
                if radius > 0:
                    self.last_radius = radius
                self.last_seen_time = t
            else:
                if (self.last_seen_time is not None
                        and (t - self.last_seen_time) > self.lost_timeout):
                    self.center = None


# ============================================================================
# Director — units-free version
#
# Works entirely in image pixels and motor steps. Does linear regression
# on (t, x_px) and (t, y_px), predicts intersection with the rail's
# x-line, computes the predicted y in pixels, converts directly to a
# step target via:
#
#     step_target = step_center + (y_pred_px - y_center_px) * steps_per_pixel
#
# steps_per_pixel may be negative; that's fine, it just encodes which
# physical direction is increasing image-y.
# ============================================================================
class Director:
    # State machine values for the commit-zone logic
    STATE_IDLE      = "IDLE"        # no ball seen recently
    STATE_TRACKING  = "TRACKING"    # ball in observation zone, predicting + commanding
    STATE_COMMITTED = "COMMITTED"   # ball past commit line, target FROZEN

    def __init__(self, client: PaddleClient,
                 x_rail_px: float, y_center_px: float,
                 steps_per_pixel: float,
                 step_min: int, step_max: int,
                 # X-coordinate of the LEFT edge of the play area, used to
                 # compute the commit line. The Director receives x_rail_px
                 # (right edge); we need the left edge to know the span.
                 x_left_px: float = 0.0,
                 # Y-coordinates of the top and bottom edges of the play
                 # area, in image pixels. Used for bounce prediction:
                 # if the predicted trajectory hits one of these walls
                 # before reaching the rail, we reflect the y-velocity
                 # and continue the trajectory from the bounce point.
                 # When 0, bounce prediction is disabled.
                 y_top_px: float = 0.0,
                 y_bot_px: float = 0.0,
                 # Callable returning the current paddle position in motor
                 # steps. The main loop polls the Arduino at 20 Hz; this
                 # lets the Director read that value without doing its
                 # own serial round-trip per command. Defaults to a
                 # constant 0 if not provided, which makes the very first
                 # move use full-speed scaling (correct behavior since
                 # the paddle IS at step 0 right after rail calibration).
                 paddle_steps_getter=None,
                 history_len: int = 6, min_fit_points: int = 2,
                 min_flight_time_s: float = 0.02,
                 max_flight_time_s: float = 3.0,
                 ema_alpha: float = 0.55,
                 # Speed/acceleration envelope (steps per s, steps per s^2).
                 slow_speed_sps: int = 800, fast_speed_sps: int = 2500,
                 slow_accel_sps2: int = 2500, fast_accel_sps2: int = 8000,
                 # ── PHYSICAL-DISTANCE THRESHOLDS (steps) ──────────────
                 # Speed scaling now branches on the actual physical
                 # remaining travel distance, not on prediction change.
                 # These thresholds are in motor STEPS, the right unit
                 # because they're invariant to camera pixel scale.
                 #
                 # big_move_threshold_steps: at or above this, use full
                 #   speed/accel. Below, interpolate.
                 # small_move_dead_band_steps: at or below this, suppress
                 #   the command entirely (no point asking AccelStepper
                 #   to plan a 5-step move; it can't even ramp up).
                 big_move_threshold_steps: int = 400,
                 small_move_dead_band_steps: int = 80,
                 # ── COMMIT-ZONE STATE MACHINE ──────────────────────────
                 # Fraction of the play-area x-span (left -> rail) at which
                 # the prediction freezes. 0.6 = freeze when ball has
                 # traversed 60% of the way to the rail.
                 commit_fraction: float = 0.60,
                 # After the ball is lost, how long to stay in COMMITTED
                 # before returning to IDLE. Prevents a bounced ball from
                 # restarting the predictor too eagerly.
                 hold_after_lost_s: float = 0.30,
                 # Number of consecutive observations needed past the
                 # commit line to actually transition. Defends against
                 # a single noisy detection triggering a false freeze.
                 commit_consecutive_required: int = 1,
                 ):
        self.client = client
        self.x_rail_px = x_rail_px
        self.y_center_px = y_center_px
        self.steps_per_pixel = steps_per_pixel
        self.step_min = step_min
        self.step_max = step_max
        self.min_fit_points = min_fit_points
        self.min_flight_time_s = min_flight_time_s
        self.max_flight_time_s = max_flight_time_s
        self.big_move_threshold_steps = big_move_threshold_steps
        self.small_move_dead_band_steps = small_move_dead_band_steps
        self.ema_alpha = ema_alpha
        self.slow_speed_sps = slow_speed_sps
        self.fast_speed_sps = fast_speed_sps
        self.slow_accel_sps2 = slow_accel_sps2
        self.fast_accel_sps2 = fast_accel_sps2
        self.paddle_steps_getter = paddle_steps_getter or (lambda: 0)
        self.ts = deque(maxlen=history_len)
        self.xs = deque(maxlen=history_len)
        self.ys = deque(maxlen=history_len)
        self.last_prediction_y_px: Optional[float] = None
        self.last_commanded_y_px: Optional[float] = None
        self.y_pred_smooth_px: Optional[float] = None
        self.last_obs_t: Optional[float] = None

        # ── Commit-zone state ─────────────────────────────────────────
        self.x_left_px = x_left_px
        self.y_top_px = y_top_px
        self.y_bot_px = y_bot_px
        # Compute the absolute pixel x of the commit line. If the ball is
        # moving rightward toward x_rail_px, observations with x_px past
        # this value transition us into COMMITTED.
        self.commit_line_px = (
            x_left_px + commit_fraction * (x_rail_px - x_left_px)
        )
        self.hold_after_lost_s = hold_after_lost_s
        self.commit_consecutive_required = commit_consecutive_required
        self.state = self.STATE_IDLE
        # Counter for consecutive observations past commit_line_px.
        # Reset whenever an observation is on the wrong side.
        self._past_commit_count = 0

        # ── Bounce path for drawing ──────────────────────────────────
        # List of (x_px, y_px) tuples representing the predicted
        # trajectory: starts at the current ball position, includes
        # each bounce point, ends at the rail intercept. Drawn by the
        # overlay code to visualize the predicted bouncing path.
        # None means "no current prediction" (drawing skipped).
        self.predicted_path_px: Optional[list] = None

    def y_px_to_steps(self, y_px: float) -> int:
        """Convert an image-y to absolute motor steps. step 0 is at
        y_center_px after the rail calibration's recentering."""
        steps = int(round((y_px - self.y_center_px) * self.steps_per_pixel))
        if steps < self.step_min:
            steps = self.step_min
        if steps > self.step_max:
            steps = self.step_max
        return steps

    def clear(self):
        """Reset the trajectory history but keep the state machine alone.
        Called automatically when the ball is lost; the state machine
        decides separately when to transition states."""
        self.ts.clear(); self.xs.clear(); self.ys.clear()
        self.last_prediction_y_px = None
        self.last_commanded_y_px = None
        self.y_pred_smooth_px = None
        self._past_commit_count = 0
        self.predicted_path_px = None

    def reset(self):
        """Hard reset: clears history AND forces state back to IDLE.
        Use when the user wants to abort the current trajectory (spacebar)."""
        self.clear()
        self.state = self.STATE_IDLE
        self.last_obs_t = None
        print(f"[Director] manual reset → {self.state}")

    def _predict_intercept_with_bounces(self, x: float, y: float,
                                        vx: float, vy: float,
                                        max_bounces: int = 5):
        """Predict the y-coordinate where the ball will reach the rail
        (x = x_rail_px), accounting for elastic bounces off the top
        and bottom edges of the play area (and the left edge if the
        ball reverses x-direction).

        Args:
            x, y:   starting position in pixels
            vx, vy: velocity in pixels/sec
            max_bounces: safety cap to prevent infinite loops from
                         numerical edge cases.

        Returns:
            (y_intercept_px, path) where path is a list of (x, y)
            tuples beginning at the starting point, including each
            bounce position, and ending at the rail intercept.
            Returns (None, []) if the trajectory is invalid (not
            moving toward rail, or runaway bounce count).

        Bounce model: perfectly elastic, axis-aligned walls.
        At top (y=y_top) or bottom (y=y_bot) wall: vy → -vy.
        At left (x=x_left) wall: vx → -vx.
        Speed magnitude is preserved (no energy loss). The
        intercept position on the rail is what matters; bounce
        timing inaccuracy from real-world energy loss does not
        change the geometric prediction.
        """
        # If bounce bounds aren't configured, fall back to the simple
        # single-segment intercept (current behavior).
        bounces_configured = (self.y_top_px > 0 and self.y_bot_px > 0
                              and self.y_bot_px > self.y_top_px)

        path = [(x, y)]

        # Safety: a tiny positive epsilon to avoid re-detecting the wall
        # we just bounced off due to floating-point error placing the
        # post-bounce position exactly on the wall.
        EPS = 1e-4

        for _ in range(max_bounces + 1):
            # Compute candidate event times. Each is "time until x or y
            # reaches the corresponding wall," only valid (positive) if
            # the ball is moving toward that wall.
            candidates = []  # list of (t, "wall_label")

            if vx > 0:
                t_rail = (self.x_rail_px - x) / vx
                if t_rail > EPS:
                    candidates.append((t_rail, "rail"))
            elif vx < 0 and bounces_configured:
                t_left = (self.x_left_px - x) / vx
                if t_left > EPS:
                    candidates.append((t_left, "left"))

            if bounces_configured:
                if vy < 0:
                    t_top = (self.y_top_px - y) / vy
                    if t_top > EPS:
                        candidates.append((t_top, "top"))
                elif vy > 0:
                    t_bot = (self.y_bot_px - y) / vy
                    if t_bot > EPS:
                        candidates.append((t_bot, "bot"))

            if not candidates:
                # No valid event ahead. Could happen if vx=0 and bounces
                # not configured, or if numerical edge cases eat all
                # candidates. Bail out.
                return (None, path)

            # The next event is the smallest positive t.
            t_next, wall = min(candidates, key=lambda c: c[0])

            # Advance the trajectory to the event point.
            x = x + vx * t_next
            y = y + vy * t_next

            if wall == "rail":
                # Reached the rail. Done. Append final point and return.
                path.append((x, y))
                return (y, path)

            # It's a bounce. Append bounce point and reflect velocity.
            path.append((x, y))
            if wall == "top" or wall == "bot":
                vy = -vy
            elif wall == "left":
                vx = -vx
            # Loop continues with new state.

        # If we exit the loop without hitting the rail, something is
        # wrong (e.g., parallel-to-rail trajectory). Return failure.
        return (None, path)

    def observe(self, x_px: float, y_px: float, t: float):
        # ── State machine: handle lost-ball timeout ──────────────────
        # If we haven't seen an observation in a while, decide whether
        # to advance the state machine. Long absence (> hold_after_lost_s)
        # means the trajectory is over and we're ready for a new ball.
        if self.last_obs_t is not None:
            gap = t - self.last_obs_t
            if gap > self.hold_after_lost_s:
                # Long enough absence: trajectory is over, return to IDLE
                # so the next ball can start a fresh TRACKING cycle.
                if self.state != self.STATE_IDLE:
                    print(f"[Director] {self.state} → IDLE (lost {gap*1000:.0f} ms)")
                self.state = self.STATE_IDLE
                self.clear()
            elif gap > 0.30:
                # Shorter absence: ball just dropped a few frames. Clear
                # the trajectory window so the next sample doesn't get
                # blended with stale data, but keep the state.
                self.clear()
        self.last_obs_t = t

        # If we're in COMMITTED state, we DO NOT update the prediction
        # or send any motor commands no matter what we see. The ball
        # could be bouncing back, a hand could be moving in the frame,
        # noise could be triggering detections — we don't care, the
        # paddle is locked in until the trajectory ends.
        if self.state == self.STATE_COMMITTED:
            return

        # ── Append observation and check for trajectory fit ──────────
        self.ts.append(t); self.xs.append(x_px); self.ys.append(y_px)
        if len(self.ts) < self.min_fit_points:
            return
        t_arr = np.fromiter(self.ts, dtype=float)
        x_arr = np.fromiter(self.xs, dtype=float)
        y_arr = np.fromiter(self.ys, dtype=float)
        t_mean = t_arr.mean()
        t_c = t_arr - t_mean
        tvar = float(np.dot(t_c, t_c) / t_c.size)
        if tvar < 1e-10:
            return
        b_x = float(np.dot(t_c, x_arr) / (t_c.size * tvar))
        b_y = float(np.dot(t_c, y_arr) / (t_c.size * tvar))
        a_x = float(x_arr.mean()); a_y = float(y_arr.mean())
        t_latest = t_arr[-1]
        x_now = a_x + b_x * (t_latest - t_mean)
        y_now = a_y + b_y * (t_latest - t_mean)
        if b_x <= 50:
            return

        # ── Predict intercept, accounting for bounces ───────────────
        # b_x and b_y are velocities in px/sec. Use the bounce-aware
        # solver to find where the ball will hit the rail. If bounce
        # bounds aren't configured (y_top/y_bot are 0), this falls
        # back to the simple straight-line intercept automatically.
        y_pred_px, predicted_path = self._predict_intercept_with_bounces(
            x_now, y_now, b_x, b_y)
        if y_pred_px is None:
            return
        # Validate the result is within the rail's reachable range.
        # If the ball would intercept above the top edge or below the
        # bottom edge of our play area, the bounce model is telling us
        # something is wrong with the prediction; reject it.
        if (self.y_top_px > 0 and self.y_bot_px > 0):
            if y_pred_px < self.y_top_px - 50 or y_pred_px > self.y_bot_px + 50:
                return
        # Sanity check on flight time: estimate total tau as the sum of
        # segment times along the path. If absurdly large or negative,
        # reject. We approximate using x-component only (always positive
        # progress for forward-moving balls, since bounces don't change
        # that here unless bouncing off the left wall).
        # Approximate total flight time as the path's x-extent / |b_x|;
        # for paths that bounce off the left wall this underestimates
        # but it's a sanity check, not a precise calculation.
        total_x_distance = sum(
            abs(predicted_path[i+1][0] - predicted_path[i][0])
            for i in range(len(predicted_path) - 1)
        )
        approx_tau = total_x_distance / max(abs(b_x), 1.0)
        if approx_tau < self.min_flight_time_s or approx_tau > self.max_flight_time_s:
            return

        # Store the path so the overlay can draw it.
        self.predicted_path_px = predicted_path

        # We have a valid forward-moving prediction. If we were IDLE,
        # we're now TRACKING.
        if self.state == self.STATE_IDLE:
            self.state = self.STATE_TRACKING
            print(f"[Director] IDLE → TRACKING")

        # ── Check for the IDLE→TRACKING→COMMITTED transition ─────────
        # When the latest observation is past the commit line, count it.
        # We require N consecutive past-the-line observations before
        # actually freezing, to defend against a single noisy detection.
        if x_px >= self.commit_line_px:
            self._past_commit_count += 1
        else:
            self._past_commit_count = 0

        if (self.state == self.STATE_TRACKING
                and self._past_commit_count >= self.commit_consecutive_required):
            # FREEZE. Compute the final prediction here and commit it.
            # No more updates until the trajectory ends.
            if self.y_pred_smooth_px is None:
                # Edge case: only just transitioned from IDLE; use raw.
                final_target_px = y_pred_px
            else:
                # Use the existing smoothed value; refining further is
                # exactly what we DON'T want to do in COMMITTED.
                final_target_px = self.y_pred_smooth_px
            # Send one last command at this final target before freezing.
            # This guarantees the motor is heading to the committed value
            # even if the dead band would have suppressed it otherwise.
            self._send_motor_command(final_target_px, force=True)
            self.last_prediction_y_px = final_target_px
            self.state = self.STATE_COMMITTED
            print(f"[Director] TRACKING → COMMITTED at y={final_target_px:.0f} px")
            return

        # ── EMA smoothing on the predicted y (in pixels) ─────────────
        if self.y_pred_smooth_px is None:
            self.y_pred_smooth_px = y_pred_px
        else:
            self.y_pred_smooth_px = (
                self.ema_alpha * y_pred_px
                + (1.0 - self.ema_alpha) * self.y_pred_smooth_px)
        y_target_px = self.y_pred_smooth_px
        self.last_prediction_y_px = y_target_px

        # Issue a motor command if outside the dead band. This is the
        # hierarchical-control path used during normal TRACKING.
        self._send_motor_command(y_target_px, force=False)

    def _send_motor_command(self, y_target_px: float, force: bool = False):
        """Common motor-command logic used by both ongoing TRACKING
        updates and the one-shot final command at COMMIT time.

        force=True bypasses the dead-band check (used at commit time
        so we always send the final target even if it's a small move).

        ── KEY DESIGN POINT ─────────────────────────────────────────
        Speed and acceleration scale with the PHYSICAL DISTANCE the
        paddle has to travel right now (from its actual current
        position to the target). Earlier versions scaled with the
        prediction CHANGE between consecutive frames, which had
        nothing to do with how far the paddle actually had to move
        and produced "slow when it should be fast" behavior:
        prediction refinements of a few pixels mid-flight would
        kick the speed scaling into low gear even though the paddle
        was only halfway through a large move.

        Now: query the current paddle position via the getter, compute
        |target - current|, and scale based on that. Result: motor
        runs fast when far from target, gentle when close, regardless
        of how much the prediction wobbles.
        """
        target_steps = self.y_px_to_steps(y_target_px)
        # Read the most-recently-polled paddle position.
        try:
            current_steps = int(self.paddle_steps_getter())
        except Exception:
            current_steps = 0
        move_distance_steps = abs(target_steps - current_steps)

        # Dead band on PHYSICAL remaining distance. If the paddle is
        # already nearly where it needs to be, don't bother sending
        # another command; AccelStepper can't usefully execute a tiny
        # move anyway, and re-commanding mid-ramp can interfere with
        # the existing motion profile.
        if not force and move_distance_steps < self.small_move_dead_band_steps:
            return

        # ── Speed/acceleration envelope ──────────────────────────────
        # Branch on physical distance (in motor steps), invariant to
        # camera resolution / pixel scale.
        if move_distance_steps >= self.big_move_threshold_steps:
            speed = self.fast_speed_sps
            accel = self.fast_accel_sps2
        else:
            # Interpolation factor (0..1) between dead-band edge and
            # big-move threshold.
            t_norm = ((move_distance_steps - self.small_move_dead_band_steps)
                      / max(1, (self.big_move_threshold_steps
                                - self.small_move_dead_band_steps)))
            t_norm = max(0.0, min(1.0, t_norm))
            speed = int(self.slow_speed_sps
                        + t_norm * (self.fast_speed_sps - self.slow_speed_sps))
            accel = int(self.slow_accel_sps2
                        + t_norm * (self.fast_accel_sps2 - self.slow_accel_sps2))

        # Final safeguard: cap the achievable peak velocity in this
        # move to the requested max_speed. For a triangular profile,
        # v_peak = sqrt(a*d). If that exceeds our chosen max speed,
        # reduce accel so v_peak == speed exactly. This keeps short
        # moves from feeling jerky even when accel is set high.
        if move_distance_steps > 0:
            v_peak = (accel * move_distance_steps) ** 0.5
            if v_peak > speed:
                accel = max(100, int(speed * speed / move_distance_steps))

        self.client.set_max_speed(speed)
        self.client.set_acceleration(accel)
        self.client.goto_steps(target_steps)
        self.last_commanded_y_px = y_target_px


# ============================================================================
# Geometry + setup state
# ============================================================================
@dataclass
class Geometry:
    v1: Optional[Tuple[int, int]] = None
    v2: Optional[Tuple[int, int]] = None
    h1: Optional[Tuple[int, int]] = None
    h2: Optional[Tuple[int, int]] = None
    mode: str = "idle"
    locked: bool = False
    step_top: Optional[int] = None
    step_bot: Optional[int] = None
    current_steps: int = 0

    def vertical_ok(self) -> bool:
        return self.v1 is not None and self.v2 is not None

    def horizontal_ok(self) -> bool:
        return self.h1 is not None and self.h2 is not None

    def rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not (self.vertical_ok() and self.horizontal_ok()):
            return None
        x_rail = (self.v1[0] + self.v2[0]) // 2
        y_top = min(self.v1[1], self.v2[1])
        y_bot = max(self.v1[1], self.v2[1])
        x_left = min(self.h1[0], self.h2[0])
        x_right = x_rail
        if x_right <= x_left:
            return None
        return (x_left, y_top, x_right, y_bot)

    def x_rail_px(self) -> Optional[float]:
        if not self.vertical_ok():
            return None
        return (self.v1[0] + self.v2[0]) / 2.0

    def y_center_px(self) -> Optional[float]:
        r = self.rect()
        if r is None:
            return None
        return (r[1] + r[3]) / 2.0

    def y_top_px(self) -> Optional[int]:
        r = self.rect()
        return None if r is None else r[1]

    def y_bot_px(self) -> Optional[int]:
        r = self.rect()
        return None if r is None else r[3]

    def rail_calibration_ready(self) -> bool:
        return self.step_top is not None and self.step_bot is not None


# ============================================================================
# Threads
# ============================================================================
def capture_thread_fn(cap):
    global running, latest_raw, frame_seq
    print("Capture thread started.")
    err = 0
    while running:
        try:
            if cap.grab():
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    with raw_lock:
                        latest_raw = frame
                        frame_seq += 1
                    err = 0
                else:
                    err += 1
            else:
                err += 1
            if err > 30:
                print("Too many capture errors; stopping.")
                break
        except Exception as exc:
            print(f"Capture error: {exc}")
            err += 1
            time.sleep(0.01)
    print("Capture thread ended.")


def processing_thread_fn(tracker, director_holder):
    global running
    last_seq = -1
    proc_count = 0
    proc_t0 = time.time()
    while running:
        frame = None
        with raw_lock:
            if latest_raw is not None and frame_seq != last_seq:
                frame = latest_raw.copy()
                last_seq = frame_seq
        if frame is None:
            time.sleep(0.001)
            continue
        now = time.time()
        center, radius, _ = tracker.detect_ball(frame)
        tracker.update(center, radius, now)
        director = director_holder[0]
        if director is not None and center is not None:
            director.observe(center[0], center[1], now)
        proc_count += 1
        if time.time() - proc_t0 >= 1.0:
            print(f"  [process] {proc_count:.0f} fps  "
                  f"detected={center is not None}")
            proc_count = 0
            proc_t0 = time.time()
    print("Processing thread ended.")


# ============================================================================
# Drawing
# ============================================================================
def draw_overlay(frame, tracker, geom, paddle_steps, director, display_fps,
                 recording: bool = False):
    disp = frame.copy() if len(frame.shape) == 3 \
           else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    rect = geom.rect()
    if rect is not None:
        xL, yT, xR, yB = rect
        if geom.locked:
            color = (0, 255, 0)
        elif geom.mode == "rail_cal":
            color = (255, 200, 0)
        else:
            color = (0, 200, 255)
        cv2.rectangle(disp, (xL, yT), (xR, yB), color, 2)

    if geom.vertical_ok():
        cv2.line(disp, geom.v1, geom.v2, (0, 200, 255), 2)
        for p in (geom.v1, geom.v2):
            cv2.circle(disp, p, 5, (0, 255, 255), -1)
    elif geom.v1 is not None:
        cv2.circle(disp, geom.v1, 5, (0, 255, 255), -1)
    if geom.horizontal_ok():
        cv2.line(disp, geom.h1, geom.h2, (255, 200, 0), 2)
        for p in (geom.h1, geom.h2):
            cv2.circle(disp, p, 5, (255, 255, 0), -1)
    elif geom.h1 is not None:
        cv2.circle(disp, geom.h1, 5, (255, 255, 0), -1)

    with tracker.lock:
        c = tracker.center
        r = tracker.radius if tracker.radius > 0 else tracker.last_radius
    if c is not None:
        cv2.circle(disp, c, r, (0, 255, 255), 2)
        cv2.circle(disp, c, 4, (0, 255, 0), -1)

    # Draw the commit line (vertical, dashed-style by stitching segments)
    # and indicate the current state machine state with color.
    if director is not None and rect is not None:
        commit_x = int(director.commit_line_px)
        # State color: TRACKING = orange, COMMITTED = bright red, IDLE = gray
        if director.state == director.STATE_COMMITTED:
            commit_color = (0, 0, 255)        # red — frozen
        elif director.state == director.STATE_TRACKING:
            commit_color = (0, 165, 255)      # orange — actively predicting
        else:
            commit_color = (140, 140, 140)    # gray — waiting
        # Draw vertical "dashed" line as a series of short segments
        y_start = rect[1]
        y_end = rect[3]
        seg_len = 12
        gap_len = 8
        y = y_start
        while y < y_end:
            y2 = min(y + seg_len, y_end)
            cv2.line(disp, (commit_x, y), (commit_x, y2),
                     commit_color, 2)
            y = y2 + gap_len

    # Draw the predicted bouncing path: line segments connecting
    # the bounce points. Drawn with a softer color than the live
    # ball trail so it's clearly the *prediction* rather than the
    # actual observed positions.
    if (director is not None
            and director.predicted_path_px is not None
            and len(director.predicted_path_px) >= 2):
        # Color: a dim cyan when tracking, a dim red when committed.
        if director.state == director.STATE_COMMITTED:
            path_color = (80, 80, 200)   # dim red
        else:
            path_color = (200, 200, 80)  # dim cyan
        path = director.predicted_path_px
        for i in range(len(path) - 1):
            p1 = (int(path[i][0]), int(path[i][1]))
            p2 = (int(path[i + 1][0]), int(path[i + 1][1]))
            cv2.line(disp, p1, p2, path_color, 1)
        # Mark each bounce point with a small circle (skip first
        # point, which is the ball's current position, and last,
        # which is the rail intercept already marked separately).
        for bx, by in path[1:-1]:
            cv2.circle(disp, (int(bx), int(by)), 4, path_color, 1)

    if (director is not None
            and director.last_prediction_y_px is not None
            and rect is not None):
        y_pred = int(director.last_prediction_y_px)
        x_rail = rect[2]
        # In COMMITTED state, draw the prediction marker in red so the
        # operator can see the target is frozen. Otherwise keep yellow.
        if director.state == director.STATE_COMMITTED:
            pred_color = (0, 0, 255)
            pred_thick = 3
        else:
            pred_color = (0, 255, 255)
            pred_thick = 2
        cv2.line(disp, (x_rail - 30, y_pred), (x_rail + 30, y_pred),
                 pred_color, pred_thick)
        cv2.circle(disp, (x_rail, y_pred), 6, pred_color, pred_thick)

    # Draw current paddle position (purple tick) on the rail when the
    # Director is live (so we have a steps_per_pixel mapping).
    if (director is not None and paddle_steps is not None
            and rect is not None):
        # Inverse mapping: y_px = y_center_px + steps / steps_per_pixel
        if abs(director.steps_per_pixel) > 1e-9:
            y_paddle_px = int(director.y_center_px
                              + paddle_steps / director.steps_per_pixel)
            x_rail = rect[2]
            cv2.line(disp, (x_rail - 50, y_paddle_px),
                     (x_rail + 50, y_paddle_px), (255, 0, 255), 4)

    if geom.mode == "rail_cal" and rect is not None:
        x_rail = rect[2]
        if geom.step_top is not None:
            cv2.putText(disp, f"TOP captured: {geom.step_top}",
                        (x_rail - 240, rect[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if geom.step_bot is not None:
            cv2.putText(disp, f"BOT captured: {geom.step_bot}",
                        (x_rail - 240, rect[3] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    panel_x, panel_y, pw, ph = 10, 10, 480, 220
    cv2.rectangle(disp, (panel_x, panel_y),
                  (panel_x + pw, panel_y + ph), (0, 0, 0), -1)
    cv2.rectangle(disp, (panel_x, panel_y),
                  (panel_x + pw, panel_y + ph), (255, 255, 255), 1)

    def text(line, msg, color=(255, 255, 255), size=0.5, thick=1):
        cv2.putText(disp, msg, (panel_x + 8, panel_y + 22 + 22 * line),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)

    text(0, f"Mode: {geom.mode}   Locked: {geom.locked}",
         (200, 255, 200), 0.55, 2)
    text(1, f"Vert: {'set' if geom.vertical_ok() else 'unset'}    "
            f"Horiz: {'set' if geom.horizontal_ok() else 'unset'}")
    if geom.mode == "rail_cal":
        text(2, f"Step pos: {geom.current_steps}    "
                f"Top: {geom.step_top}    Bot: {geom.step_bot}",
             (255, 200, 0), 0.5, 1)
        text(3, "UP/DN=jog 100   PgUp/PgDn=jog 1000   HOME=goto 0",
             (180, 180, 180), 0.42)
        text(4, "Z=zero here   T=capture top   B=capture bot   K=lock",
             (180, 180, 180), 0.42)
    else:
        text(2, "[v] vert  [h] horiz  [r] rail-cal  [k] lock  "
                "[u] unlock  [m] rec  [SPACE] reset  [q] quit",
             (180, 180, 180), 0.42)

    # Show the Director's state machine status (only when locked + live)
    if director is not None:
        state_colors = {
            director.STATE_IDLE:      (160, 160, 160),  # gray
            director.STATE_TRACKING:  (0, 165, 255),    # orange
            director.STATE_COMMITTED: (0, 0, 255),      # red
        }
        c = state_colors.get(director.state, (255, 255, 255))
        text(4, f"State: {director.state}", c, 0.55, 2)

    text(6, f"Display: {display_fps:.0f} fps", (200, 200, 200))
    if paddle_steps is not None:
        text(7, f"Paddle: {paddle_steps:+d} steps",
             (255, 0, 255), 0.55, 2)

    # Recording indicator: pulsing red dot in the top-right corner.
    # Pulses at ~1 Hz for unmistakable visibility on demo video.
    if recording:
        h, w = disp.shape[:2]
        # Pulse via sine of wall-clock seconds; 0.5..1.0 amplitude
        pulse = 0.75 + 0.25 * np.sin(time.time() * 2 * np.pi)
        red = (0, 0, int(255 * pulse))
        cv2.circle(disp, (w - 40, 40), 14, red, -1)
        cv2.putText(disp, "REC", (w - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return disp


# ============================================================================
# Mouse callback
# ============================================================================
def make_mouse_cb(geom: Geometry):
    def cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if geom.mode == "vert":
            if geom.v1 is None:
                geom.v1 = (x, y)
            elif geom.v2 is None:
                geom.v2 = (x, y); geom.mode = "idle"
            else:
                geom.v1 = (x, y); geom.v2 = None
        elif geom.mode == "horiz":
            if geom.h1 is None:
                geom.h1 = (x, y)
            elif geom.h2 is None:
                geom.h2 = (x, y); geom.mode = "idle"
            else:
                geom.h1 = (x, y); geom.h2 = None
    return cb


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--exposure", type=int, default=80)
    parser.add_argument("--port", required=True)
    parser.add_argument("--debug-keys", action="store_true",
                        help="Print every keycode received")
    # ── Motion tuning ────────────────────────────────────────────────
    # These values control the motor's speed and acceleration envelope.
    # If you see the belt skip teeth (calibration drifts after motion),
    # LOWER these. The DM542 + NEMA17 + GT2 belt combination is fairly
    # forgiving below ~3000 steps/s peak with accel under 10000 sps^2,
    # but your specific build may differ. Start conservative, raise
    # gradually, verify by hand each time.
    parser.add_argument("--max-speed", type=int, default=4000,
                        help="Peak motor speed in steps/s (default: 4000)")
    parser.add_argument("--max-accel", type=int, default=8000,
                        help="Peak motor acceleration in steps/s^2 "
                             "(default: 8000)")
    parser.add_argument("--min-speed", type=int, default=800,
                        help="Speed for small moves (default: 800)")
    parser.add_argument("--min-accel", type=int, default=2500,
                        help="Acceleration for small moves (default: 2500)")
    # ── Commit-zone state machine ────────────────────────────────────
    # Where in the play area the prediction freezes. 0.6 means the
    # paddle's target locks in once the ball has crossed 60% of the way
    # from the left edge to the rail. After that, no more motor commands
    # until the trajectory ends (ball lost for hold_after_lost_s).
    parser.add_argument("--commit-fraction", type=float, default=0.60,
                        help="Fraction of the play area at which the "
                             "predicted target freezes. 0.5 = midpoint, "
                             "0.7 = three-quarters of the way (default: 0.60)")
    args = parser.parse_args()

    # ---- Paddle (no calibration file required) ----
    client = PaddleClient(args.port)
    # Use a CONSERVATIVE speed during initial jogging so the user
    # can't accidentally crash the carriage. The Director will set
    # higher per-move speeds at runtime once the user locks in.
    client.set_max_speed(min(args.min_speed, 1500))
    client.set_acceleration(min(args.min_accel, 3000))

    # ---- Camera ----
    try:
        src = int(args.src)
    except ValueError:
        src = args.src
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open camera {src}")
        client.close(); return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    except Exception:
        pass

    # ---- Threads ----
    tracker = BallTracker()
    geom = Geometry()
    director_holder = [None]
    cap_thread = threading.Thread(
        target=capture_thread_fn, args=(cap,), daemon=True)
    proc_thread = threading.Thread(
        target=processing_thread_fn,
        args=(tracker, director_holder), daemon=True)
    cap_thread.start()
    proc_thread.start()

    # ---- GUI ----
    win = "Camera + Control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, make_mouse_cb(geom))

    last_query = 0.0
    paddle_steps: Optional[int] = None
    # Mutable single-element holder so a lambda passed to the Director
    # can always read the LATEST polled paddle position. Lambdas capture
    # local variables by reference, but `paddle_steps` is rebound on
    # each query (not mutated in place), so capturing it directly would
    # freeze the lambda at the value it had at construction time. The
    # one-element list works because we mutate the [0] element rather
    # than reassigning the whole variable.
    paddle_pos_holder = [0]
    disp_count = 0
    disp_t0 = time.time()
    disp_fps = 0.0
    last_jog_target = 0
    recorder = Recorder(output_dir="recordings")

    def jog(delta_steps: int):
        nonlocal last_jog_target
        last_jog_target += delta_steps
        client.goto_steps(last_jog_target, dedupe=False)

    init = client.query()
    if init is not None:
        last_jog_target = init.position_steps
        geom.current_steps = init.position_steps
        paddle_steps = init.position_steps
        paddle_pos_holder[0] = init.position_steps

    global running
    try:
        while running:
            frame = None
            with raw_lock:
                if latest_raw is not None:
                    frame = latest_raw.copy()
            if frame is None:
                time.sleep(0.005); continue

            now = time.time()
            if now - last_query > 0.05:
                last_query = now
                st = client.query()
                if st is not None:
                    paddle_steps = st.position_steps
                    geom.current_steps = st.position_steps
                    paddle_pos_holder[0] = st.position_steps

            disp = draw_overlay(frame, tracker, geom, paddle_steps,
                                director_holder[0], disp_fps,
                                recording=recorder.is_active)
            cv2.imshow(win, disp)

            # If recording, write the post-overlay frame so the user
            # captures the HUD/predictions/paddle position alongside
            # the camera feed.
            if recorder.is_active:
                recorder.write(disp)

            disp_count += 1
            if time.time() - disp_t0 >= 0.5:
                disp_fps = disp_count / (time.time() - disp_t0)
                disp_count = 0
                disp_t0 = time.time()

            # waitKeyEx returns extended keycodes including arrow keys.
            key_full = cv2.waitKeyEx(1)
            if args.debug_keys and key_full != -1:
                print(f"keycode raw=0x{key_full:08X}  ({key_full})")
            key = key_full & 0xFF if key_full != -1 else 255

            # ---- rail_cal ----
            if geom.mode == "rail_cal":
                # Try multiple known arrow-key encodings (Linux Qt, GTK,
                # Windows). If yours differs, run with --debug-keys.
                if key_full in (0xFF52, 65362, 2490368):    # UP
                    jog(-100)
                elif key_full in (0xFF54, 65364, 2621440):  # DOWN
                    jog(+100)
                elif key_full in (0xFF55, 65365, 2162688):  # PgUp
                    jog(-1000)
                elif key_full in (0xFF56, 65366, 2228224):  # PgDn
                    jog(+1000)
                elif key_full in (0xFF50, 65360, 2359296):  # Home
                    last_jog_target = 0
                    client.goto_steps(0, dedupe=False)
                elif key in (ord('z'), ord('Z')):
                    client.zero_here()
                    last_jog_target = 0
                    geom.current_steps = 0
                    print("Zeroed at current position.")
                elif key in (ord('t'), ord('T')):
                    geom.step_top = geom.current_steps
                    print(f"TOP captured at {geom.step_top}")
                elif key in (ord('b'), ord('B')):
                    geom.step_bot = geom.current_steps
                    print(f"BOT captured at {geom.step_bot}")
                elif key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    geom.mode = "idle"
                elif key == ord('m') or key == ord('M'):
                    if recorder.is_active:
                        recorder.stop()
                    else:
                        recorder.start(disp.shape,
                                       disp_fps if disp_fps > 0 else 30.0)
                elif key == ord('k'):
                    if not geom.rail_calibration_ready():
                        print("Cannot lock: capture both T and B first.")
                    elif geom.rect() is None:
                        print("Cannot lock: rectangle is incomplete.")
                    else:
                        # The math: y_top_px corresponds to step_top,
                        # y_bot_px to step_bot. We want a linear map
                        #     steps = (y_px - y_center_px) * steps_per_pixel
                        # such that steps_top ↔ y_top_px and
                        # steps_bot ↔ y_bot_px after we re-zero at the
                        # midpoint.
                        y_top_px = geom.y_top_px()
                        y_bot_px = geom.y_bot_px()
                        # Midpoint mapping is built in via the recenter.
                        # steps_per_pixel from the captured span:
                        spp = ((geom.step_bot - geom.step_top)
                               / (y_bot_px - y_top_px))
                        # After we re-zero at the midpoint, "step_top"
                        # becomes (step_top - midpoint), and likewise
                        # for bot. The DIFFERENCE step_bot - step_top
                        # is unchanged, so spp is already correct.

                        midpoint_steps = (geom.step_top + geom.step_bot) // 2
                        client.goto_steps(midpoint_steps, dedupe=False)
                        for _ in range(40):
                            st = client.query()
                            if st is not None and not st.busy:
                                break
                            time.sleep(0.05)
                        client.zero_here()
                        last_jog_target = 0

                        # Soft limits in steps relative to new zero
                        step_min = -abs(geom.step_bot - geom.step_top) // 2
                        step_max = +abs(geom.step_bot - geom.step_top) // 2

                        tracker.set_roi(geom.rect())
                        rect = geom.rect()
                        director_holder[0] = Director(
                            client=client,
                            x_rail_px=geom.x_rail_px(),
                            y_center_px=geom.y_center_px(),
                            steps_per_pixel=spp,
                            step_min=step_min,
                            step_max=step_max,
                            x_left_px=float(rect[0]),
                            y_top_px=float(rect[1]),
                            y_bot_px=float(rect[3]),
                            paddle_steps_getter=lambda: paddle_pos_holder[0],
                            slow_speed_sps=args.min_speed,
                            fast_speed_sps=args.max_speed,
                            slow_accel_sps2=args.min_accel,
                            fast_accel_sps2=args.max_accel,
                            commit_fraction=args.commit_fraction,
                        )
                        geom.locked = True
                        geom.mode = "idle"
                        print(f"LOCKED.")
                        print(f"  rect             = {geom.rect()}")
                        print(f"  steps_per_pixel  = {spp:+.3f}")
                        print(f"  step soft limits = "
                              f"{step_min} to {step_max}")
                        print(f"  speed envelope   = "
                              f"{args.min_speed}..{args.max_speed} steps/s")
                        print(f"  accel envelope   = "
                              f"{args.min_accel}..{args.max_accel} steps/s^2")
                        print(f"  commit fraction  = {args.commit_fraction:.2f}  "
                              f"(commit line at x={director_holder[0].commit_line_px:.0f} px)")
                continue

            # ---- idle ----
            if key == ord('q') or key == 27:
                break
            elif key == ord('v'):
                geom.mode = "vert"; geom.v1 = None; geom.v2 = None
                geom.locked = False; director_holder[0] = None
                tracker.set_roi(None)
            elif key == ord('h'):
                geom.mode = "horiz"; geom.h1 = None; geom.h2 = None
                geom.locked = False; director_holder[0] = None
                tracker.set_roi(None)
            elif key == ord('r'):
                if not (geom.vertical_ok() and geom.horizontal_ok()):
                    print("Set vertical and horizontal lines first.")
                else:
                    geom.mode = "rail_cal"
                    geom.step_top = None
                    geom.step_bot = None
                    director_holder[0] = None
                    geom.locked = False
                    tracker.set_roi(None)
                    print("Rail-cal mode: arrows to jog, T/B to capture.")
            elif key == ord('u'):
                geom.locked = False
                director_holder[0] = None
                tracker.set_roi(None)
                print("UNLOCKED.")
            elif key == ord('m') or key == ord('M'):
                # Toggle recording. We pass `disp` shape because we're
                # recording the post-overlay frame. Use the most recent
                # measured display fps (or 30 as a fallback if we
                # haven't measured yet).
                if recorder.is_active:
                    recorder.stop()
                else:
                    recorder.start(disp.shape, disp_fps if disp_fps > 0 else 30.0)
            elif key == ord(' '):
                # Spacebar: hard reset of the Director's state machine.
                # Useful when the system is stuck in COMMITTED and you
                # want to clear it without waiting for the lost-ball
                # timeout. Has no effect if there's no active director.
                if director_holder[0] is not None:
                    director_holder[0].reset()

    finally:
        running = False
        # Close any active recording FIRST so the file is finalized
        # even if other shutdown steps misbehave.
        try: recorder.stop()
        except Exception: pass
        for t in (cap_thread, proc_thread):
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: client.close()
        except Exception: pass
        print("Done.")


if __name__ == "__main__":
    main()