import argparse
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Global state shared between threads.
#
# The ARCHITECTURE has changed from v3.0.  Previously there were two
# threads:
#   1. capture_thread:  grab frame → detect_ball → copy to latest_frame
#   2. main thread:     read latest_frame → draw overlay → imshow
#
# The problem: detect_ball() takes ~15-20ms per frame (CLAHE, blur,
# morphology, contour search on a 1280×800 image).  Since it ran
# INSIDE the capture loop, the loop period was:
#
#   T_loop = T_grab + T_retrieve + T_detect + T_copy
#          ≈  0     +   4ms      +  15ms    +  1ms   = 20ms
#
#   ⟹  f_loop = 1 / 20ms = 50 fps   ← exactly what you measured!
#
# The fix: THREE threads now:
#   1. capture_thread:   grab → retrieve → copy to latest_raw   (FAST)
#   2. processing_thread: read latest_raw → detect_ball → update tracker
#   3. main thread:       read latest_raw → draw overlay → imshow
#
# The capture thread now does ONLY grab+retrieve+copy — no processing.
# This means its loop period is just:
#
#   T_loop = T_grab + T_retrieve + T_copy ≈ 0 + 4ms + 1ms = 5ms
#
#   ⟹  f_loop = 1 / 5ms = 200 fps (faster than camera, so camera
#       is the bottleneck → we get the full 100-120 fps from the sensor)
#
# The processing thread runs detect_ball() at whatever rate it can.
# If it's slower than 100fps, that's fine — we just skip frames for
# detection, but we still CAPTURE at full rate.  The tracker's velocity
# math still works because it uses wall-clock timestamps, not frame
# indices.
# ──────────────────────────────────────────────────────────────────────

running = True

# latest_raw: the most recent frame from the camera, untouched.
# Written by capture_thread, read by processing_thread and main thread.
latest_raw = None
raw_lock = threading.Lock()

# frame_seq: incremented every time capture_thread stores a new frame.
# processing_thread uses this to detect when a genuinely new frame is
# available, so it doesn't re-process the same frame.
frame_seq = 0


def signal_handler(sig, frame):
    global running
    print("\n\nForce quit detected! Exiting...")
    running = False
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class BallTracker:
    def __init__(
        self,
        threshold_k=1.5,
        motion_thresh=18,
        min_area=50,
        max_area=3000,
        circularity_min=0.65,
        history_len=120,
        lost_timeout=0.40,
        trail_duration=1.0,
    ):
        self.threshold_k = threshold_k
        self.motion_thresh = motion_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_min = circularity_min
        self.lost_timeout = lost_timeout
        self.trail_duration = trail_duration

        self.positions = deque(maxlen=history_len)
        self.times = deque(maxlen=history_len)

        self.center = None
        self.radius = 0
        self.last_radius = 15

        self.velocity_px = (0.0, 0.0)
        self.speed_px = 0.0
        self.velocity_m = (0.0, 0.0)
        self.speed_m = 0.0
        self.direction_deg = 0.0

        self.last_seen_time = None
        self.prev_gray = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_bright_T = None

        # Thread-safety lock for tracker state that is read by main
        # thread (for drawing) and written by processing thread.
        self.lock = threading.Lock()

    def detect_ball(self, frame):
        """Detect a ping pong ball that may be stationary, slow, OR fast.

        The v3.0/3.1 detector only found slow-moving balls because it
        required high circularity (≥ 0.65).  A fast-moving ball at 120fps
        creates a motion-blurred streak that is far from circular.

        PHYSICS OF THE PROBLEM
        ──────────────────────
        A table tennis ball is ~40mm diameter ≈ 10-20 px on this camera.
        A standard hit crosses the 1280px frame in ~9 frames at 120fps.
        Per-frame displacement:

            d = 1280 px / 9 frames ≈ 142 px/frame

        The ball diameter D ≈ 15 px, so the aspect ratio of the
        motion-blurred streak is:

            r = (D + d) / D = (15 + 142) / 15 ≈ 10.5

        Circularity of a rectangle with aspect ratio r:

            C = π·r / (r + 1)²

        At r=10.5:  C ≈ 0.25   (vs. the old threshold of 0.65 → REJECTED)
        At r=5:     C ≈ 0.44   (still rejected)
        At r=2:     C ≈ 0.70   (barely passes — only very slow balls)

        Additionally, morphological OPEN with a 5×5 kernel erodes thin
        streaks (width ~5-8px) completely, destroying the contour before
        it's even found.

        SOLUTION: TWO-PATH DETECTION
        ────────────────────────────
        Path A (slow ball): High circularity, morphological cleanup.
            Same as v3.0.  Works for balls moving < ~3× their diameter
            per frame.

        Path B (fast ball): Low circularity, use minAreaRect to check
            that the streak has a plausible width (close to ball diameter)
            and is not too wide (which would indicate a hand or large
            object).  Skip the morphological OPEN to preserve thin
            streaks; use a smaller 3×3 kernel for CLOSE only.

        The best candidate across both paths wins.

        Returns:
            (center, radius, mask) where center is (x,y) tuple or None.
        """
        try:
            if frame is None:
                return None, 0, None

            # ── Step 1: Grayscale conversion ──
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # ── Step 2: CLAHE for local contrast normalization ──
            gray_eq = self.clahe.apply(gray)

            # ── Step 3: Frame differencing ──
            if self.prev_gray is None:
                self.prev_gray = gray_eq
                return None, 0, None

            diff = cv2.absdiff(gray_eq, self.prev_gray)
            self.prev_gray = gray_eq

            # ── Step 4: Blur + threshold → motion mask ──
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            _, motion = cv2.threshold(
                diff_blur, self.motion_thresh, 255, cv2.THRESH_BINARY
            )

            # ── Step 5: Adaptive brightness threshold ──
            mean, std = cv2.meanStdDev(gray_eq)
            threshold = float(mean.item()) + self.threshold_k * float(std.item())
            threshold = max(60.0, min(240.0, threshold))
            self.last_bright_T = threshold

            _, bright = cv2.threshold(
                gray_eq, threshold, 255, cv2.THRESH_BINARY
            )

            # ── Step 6: Combine motion + brightness ──
            combined = cv2.bitwise_and(motion, bright)

            # ── Step 7: TWO morphological paths ──
            #
            # Path A (slow ball):
            #   Full OPEN (5×5) to remove noise, then CLOSE to fill gaps.
            #   This destroys thin streaks but cleans up the mask nicely
            #   for round blobs.
            #
            # Path B (fast ball):
            #   NO open (preserves thin streaks).  Only a small CLOSE
            #   (3×3) to bridge tiny gaps in the streak.  Then dilate
            #   slightly to thicken the streak for more robust contour
            #   detection.

            kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # Path A mask: aggressive cleanup
            mask_slow = cv2.morphologyEx(
                combined, cv2.MORPH_OPEN, kernel_5, iterations=1
            )
            mask_slow = cv2.morphologyEx(
                mask_slow, cv2.MORPH_CLOSE, kernel_5, iterations=2
            )

            # Path B mask: gentle — preserve thin structures
            mask_fast = cv2.morphologyEx(
                combined, cv2.MORPH_CLOSE, kernel_3, iterations=1
            )
            mask_fast = cv2.dilate(mask_fast, kernel_3, iterations=1)

            # ── Step 8: Find contours on BOTH masks ──
            contours_slow, _ = cv2.findContours(
                mask_slow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_fast, _ = cv2.findContours(
                mask_fast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best_center = None
            best_radius = 0
            best_score = -1.0

            # ── Path A scoring: slow/stationary ball (circular) ──
            for contour in contours_slow:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > self.max_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.45:
                    # Still need SOME circularity for the slow path,
                    # but much more relaxed than the old 0.65.
                    continue

                (_, _), radius = cv2.minEnclosingCircle(contour)
                if radius <= 0:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue

                center = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )

                # Score: heavily reward circularity for the slow path.
                score = circularity * circularity * area
                if score > best_score:
                    best_score = score
                    best_center = center
                    best_radius = int(radius)

            # ── Path B scoring: fast ball (elongated streak) ──
            #
            # For a fast-moving ball, the blob is a thin elongated
            # streak.  We use minAreaRect to measure:
            #   - minor_axis (width): should be close to ball diameter
            #     (roughly 5–30 px depending on distance)
            #   - major_axis (length): can be very long (the streak)
            #   - aspect_ratio = major/minor: high for fast balls
            #
            # We ACCEPT blobs with:
            #   - minor axis in [3, 40] px (plausible ball width)
            #   - aspect ratio > 1.5  (clearly elongated = fast motion)
            #   - area in [min_area, max_area_fast]
            #
            # max_area is raised for the fast path because a streak
            # has much more area than a stationary ball:
            #   streak_area ≈ D × (D + d) ≈ 15 × 157 ≈ 2355 px²

            max_area_fast = self.max_area * 5  # allow up to 15000 px²

            for contour in contours_fast:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > max_area_fast:
                    continue

                # minAreaRect returns ((cx,cy), (w,h), angle)
                # where w and h are NOT guaranteed to be width < height.
                rect = cv2.minAreaRect(contour)
                (rcx, rcy), (rw, rh), angle = rect

                # Ensure minor_axis < major_axis
                minor_axis = min(rw, rh)
                major_axis = max(rw, rh)

                if major_axis <= 0 or minor_axis <= 0:
                    continue

                aspect_ratio = major_axis / minor_axis

                # ── Width filter ──
                # The minor axis should be plausible for a ball:
                # too thin (< 3px) = noise, too wide (> 40px) = not a ball.
                if minor_axis < 3 or minor_axis > 40:
                    continue

                # ── Elongation check ──
                # aspect_ratio > 1.5 means clearly elongated.
                # If it's nearly round (aspect < 1.5), Path A handles it.
                if aspect_ratio < 1.5:
                    continue

                # ── Solidity check ──
                # Solidity = area / convex_hull_area.  A streak should
                # fill most of its convex hull (solidity > 0.3).
                # Random noise blobs tend to have low solidity.
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0:
                    continue
                solidity = area / hull_area
                if solidity < 0.3:
                    continue

                # ── Compute center via moments ──
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                center = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )

                # ── Scoring for fast path ──
                # We want to reward:
                #   - Higher solidity (fills its bounding rect well)
                #   - Plausible minor axis (close to expected ball size)
                #   - Larger area (more confident detection)
                #
                # We DON'T reward extreme aspect ratios per se — a very
                # long streak and a moderately long streak are both valid.
                #
                # Approximate expected minor axis (ball diameter in px).
                # Without calibration, assume ~12px.  With calibration
                # this could be refined.
                expected_minor = 12.0
                minor_match = 1.0 / (1.0 + abs(minor_axis - expected_minor) / expected_minor)

                score = solidity * minor_match * area * 0.5
                # The 0.5 factor slightly deprioritizes the fast path
                # relative to the slow path, so if both find a candidate,
                # the round one wins (it's a more confident detection).

                if score > best_score:
                    best_score = score
                    best_center = center
                    # For a streak, "radius" is half the minor axis
                    # (the ball's actual radius, not the streak length).
                    best_radius = max(int(minor_axis / 2), 3)

            # Return the combined mask (for visualization/debugging).
            # Use the fast mask since it preserves more structure.
            return best_center, best_radius, mask_fast

        except Exception as exc:
            print(f"Error in detect_ball: {exc}")
            return None, 0, None

    def update_tracking(
        self, center, radius, current_time,
        px_per_meter_x=None, px_per_meter_y=None
    ):
        """Update position history and compute velocity.

        Uses wall-clock timestamps so the velocity math is correct
        even if frames are skipped (which WILL happen now that capture
        and processing are decoupled).

        Velocity in pixels/s:
            v_x = Δx / Δt,   v_y = Δy / Δt

        Speed in m/s (if calibrated):
            v_x^m = v_x^{px} / (px/m)_x
            speed = √( (v_x^m)² + (v_y^m)² )
        """
        try:
            with self.lock:
                if center is not None:
                    self.positions.append(center)
                    self.times.append(current_time)

                    self.center = center
                    self.radius = radius
                    if radius > 0:
                        self.last_radius = radius

                    self.last_seen_time = current_time

                    if len(self.positions) >= 2:
                        (x1, y1) = self.positions[-2]
                        (x2, y2) = self.positions[-1]
                        t1 = self.times[-2]
                        t2 = self.times[-1]

                        dt = t2 - t1
                        if dt > 1e-6:
                            vx_px = (x2 - x1) / dt
                            vy_px = (y2 - y1) / dt
                            self.velocity_px = (vx_px, vy_px)
                            self.speed_px = float(np.hypot(vx_px, vy_px))

                            if px_per_meter_x and px_per_meter_y:
                                vx_m = vx_px / px_per_meter_x
                                vy_m = vy_px / px_per_meter_y
                                self.velocity_m = (vx_m, vy_m)
                                self.speed_m = float(np.hypot(vx_m, vy_m))

                            ang = np.degrees(np.arctan2(vy_px, vx_px))
                            if ang < 0:
                                ang += 360.0
                            self.direction_deg = float(ang)
                else:
                    self.center = None
                    self.radius = 0

                    if (
                        self.last_seen_time is not None
                        and (current_time - self.last_seen_time) > self.lost_timeout
                    ):
                        self.positions.clear()
                        self.times.clear()
                        self.velocity_px = (0.0, 0.0)
                        self.speed_px = 0.0
                        self.velocity_m = (0.0, 0.0)
                        self.speed_m = 0.0

        except Exception as exc:
            print(f"Error in update_tracking: {exc}")


# ──────────────────────────────────────────────────────────────────────
# SPIN ANALYZER — classifies topspin vs backspin from trajectory
# ──────────────────────────────────────────────────────────────────────
class SpinAnalyzer:
    """Classify ball spin by analyzing the vertical curvature of its
    trajectory relative to a pure gravitational parabola.

    PHYSICS
    ───────
    From a side-view camera at table height, the ball's vertical
    position during free flight obeys:

        y(t) = y₀ + v_{y0}·t + ½·a_y·t²

    where a_y is the total vertical acceleration (positive = downward
    in image coordinates, since the image y-axis points down).

    With NO spin (flat hit), only gravity acts:

        a_y = g = 9.81 m/s²

    With TOPSPIN, the Magnus force adds to gravity:

        a_y = g + a_Magnus > g       (ball dives)

    With BACKSPIN, the Magnus force opposes gravity:

        a_y = g - a_Magnus < g       (ball floats)

    ALGORITHM
    ─────────
    1. Accumulate (position, time) pairs during a detected flight.
    2. When the ball is lost (exits frame), analyze the collected
       trajectory as a completed "shot."
    3. Fit a quadratic to the vertical (y) positions vs time:
           y(t) = c₀ + c₁·t + c₂·t²
       The fitted c₂ = ½·a_y, so a_y = 2·c₂.
    4. Convert a_y from px/s² to m/s² using calibration.
    5. Compare a_y to g:
           a_y > g + threshold  →  TOPSPIN
           a_y < g - threshold  →  BACKSPIN
           otherwise            →  FLAT / UNCLEAR

    The threshold accounts for measurement noise.  With ~3px position
    noise over a ~100ms flight, the uncertainty in a_y is roughly:

        σ_a ≈ 2·σ_y / (N·Δt²)

    For N=10 points, Δt=8.3ms, σ_y=3px ≈ 0.01m:

        σ_a ≈ 2·0.01 / (10·0.0083²) ≈ 29 m/s²

    That's quite noisy for short flights!  Longer flights (N=20+,
    spanning 150ms+) bring this down to manageable levels.  We
    require a minimum flight duration and point count before
    making a classification.

    COORDINATE CONVENTION
    ─────────────────────
    Image y-axis: DOWN is positive (standard OpenCV).
    Gravity: pulls ball DOWN = positive y acceleration.
    Topspin: Magnus force points DOWN = a_y > g.
    Backspin: Magnus force points UP = a_y < g.
    """

    # Classification thresholds (in m/s²).
    # These define the "dead zone" around g where we say FLAT/UNCLEAR.
    # Tuned conservatively — better to say UNCLEAR than to misclassify.
    TOPSPIN_THRESHOLD = 3.0   # a_y > g + 3 → topspin
    BACKSPIN_THRESHOLD = 3.0  # a_y < g - 3 → backspin

    # Minimum flight requirements for a reliable fit.
    MIN_POINTS = 6            # need at least 6 data points
    MIN_DURATION_S = 0.04     # need at least 40ms of flight

    # Gravity in m/s² (positive = downward in image coords)
    G = 9.81

    def __init__(self):
        # ── Current flight accumulator ──
        # These collect data for the shot currently in progress.
        self.flight_xs = []       # x positions in pixels
        self.flight_ys = []       # y positions in pixels
        self.flight_ts = []       # timestamps (seconds)
        self.flight_active = False

        # ── Most recent completed shot result ──
        # These hold the result of the last analyzed shot and persist
        # until a new shot is analyzed, so the HUD can display them.
        self.last_spin_label = ""      # "TOPSPIN", "BACKSPIN", "FLAT", ""
        self.last_a_y_real = None      # fitted vertical accel in m/s²
        self.last_shot_time = None     # when the shot was analyzed
        self.last_confidence = ""      # "low" or "high" based on fit quality
        self.display_duration = 3.0    # seconds to show result on HUD

        # Thread lock (read by display thread, written by processing)
        self.lock = threading.Lock()

    def update(self, center, current_time, px_per_meter_x, px_per_meter_y):
        """Called every processing frame with the current detection.

        Parameters
        ----------
        center : tuple (x, y) or None
            Ball position in pixels, or None if not detected.
        current_time : float
            Wall-clock time from time.time().
        px_per_meter_x : float or None
            Calibration: pixels per meter in horizontal direction.
        px_per_meter_y : float or None
            Calibration: pixels per meter in vertical direction.
            Required for spin classification — if None, we still
            accumulate the flight but can't classify.
        """
        if center is not None:
            # ── Ball detected: accumulate flight data ──
            self.flight_xs.append(center[0])
            self.flight_ys.append(center[1])
            self.flight_ts.append(current_time)
            self.flight_active = True

        else:
            # ── Ball lost: analyze the completed flight ──
            if self.flight_active and len(self.flight_ts) >= self.MIN_POINTS:
                self._analyze_flight(px_per_meter_x, px_per_meter_y)

            # Reset for the next flight
            self.flight_xs.clear()
            self.flight_ys.clear()
            self.flight_ts.clear()
            self.flight_active = False

    def _analyze_flight(self, px_per_meter_x, px_per_meter_y):
        """Fit a quadratic to the vertical trajectory and classify spin.

        Performs a least-squares fit of:
            y(t) = c₀ + c₁·t + c₂·t²

        where t is time relative to the start of the flight.

        The vertical acceleration is a_y = 2·c₂ (in px/s²).
        Converting to m/s²:  a_y_real = a_y_px / px_per_meter_y.

        We also fit x(t) to check that horizontal motion is roughly
        linear (constant velocity), which validates that the ball is
        in free flight and not being hit or bouncing.
        """
        ts = np.array(self.flight_ts)
        xs = np.array(self.flight_xs, dtype=np.float64)
        ys = np.array(self.flight_ys, dtype=np.float64)

        # Shift time to start at 0 for numerical stability
        t0 = ts[0]
        t_rel = ts - t0

        duration = t_rel[-1] - t_rel[0]
        if duration < self.MIN_DURATION_S:
            return  # flight too short for reliable fit

        # ── Fit y(t) = c0 + c1*t + c2*t² ──
        # np.polyfit returns coefficients [c2, c1, c0] (highest degree first)
        try:
            y_coeffs = np.polyfit(t_rel, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            return  # degenerate data

        c2_y = y_coeffs[0]  # coefficient of t² → ½·a_y in px/s²
        a_y_px = 2.0 * c2_y  # vertical acceleration in px/s²

        # ── Compute fit residuals for confidence assessment ──
        y_fitted = np.polyval(y_coeffs, t_rel)
        residuals = ys - y_fitted
        rmse_px = float(np.sqrt(np.mean(residuals ** 2)))

        # ── Also fit x(t) to verify free flight ──
        # In free flight, x should be roughly linear (constant v_x).
        # If the x-fit has large quadratic component, the ball may be
        # bouncing or being hit — skip the analysis.
        try:
            x_coeffs = np.polyfit(t_rel, xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            return

        # Quadratic coefficient in x: should be small compared to
        # the linear coefficient.  a_x should be near zero in free flight
        # (no horizontal force except weak air drag).
        c2_x = x_coeffs[0]
        c1_x = x_coeffs[1]
        a_x_px = 2.0 * c2_x

        # If horizontal acceleration is more than 30% of vertical,
        # something unusual is happening — skip.
        if abs(a_y_px) > 0 and abs(a_x_px) / abs(a_y_px) > 0.3:
            # Could be a bounce, a hit, or sidespin artifact.
            # Don't classify — too ambiguous.
            pass  # we'll still store the result but mark low confidence

        # ── Convert to real-world units ──
        if px_per_meter_y is None or px_per_meter_y <= 0:
            # Can't classify without calibration
            return

        # In image coords, y increases downward.
        # Gravity pulls the ball downward = positive a_y.
        # px_per_meter_y converts: meters = pixels / px_per_meter_y
        # So: a_y_real (m/s²) = a_y_px (px/s²) / px_per_meter_y (px/m)
        a_y_real = a_y_px / px_per_meter_y

        # ── Classify ──
        if a_y_real > self.G + self.TOPSPIN_THRESHOLD:
            label = "TOPSPIN"
        elif a_y_real < self.G - self.BACKSPIN_THRESHOLD:
            label = "BACKSPIN"
        else:
            label = "FLAT"

        # ── Confidence assessment ──
        # Based on: number of points, duration, and fit RMSE.
        n = len(self.flight_ts)
        rmse_m = rmse_px / px_per_meter_y
        if n >= 12 and duration >= 0.08 and rmse_m < 0.02:
            confidence = "high"
        else:
            confidence = "low"

        # ── Store result ──
        with self.lock:
            self.last_spin_label = label
            self.last_a_y_real = float(a_y_real)
            self.last_shot_time = time.time()
            self.last_confidence = confidence

        # Console output
        print(
            f"  [spin] {label} | a_y={a_y_real:.1f} m/s² "
            f"(g={self.G:.1f}) | {n} pts over {duration*1000:.0f}ms | "
            f"RMSE={rmse_m*1000:.1f}mm | conf={confidence}"
        )

    def get_display_info(self):
        """Return the current spin label and info for HUD rendering.

        Returns (label, a_y_real, confidence) or ("", None, "") if
        no recent result or the display has expired.
        """
        with self.lock:
            if self.last_shot_time is None:
                return "", None, ""

            age = time.time() - self.last_shot_time
            if age > self.display_duration:
                return "", None, ""

            return (
                self.last_spin_label,
                self.last_a_y_real,
                self.last_confidence,
            )

# ──────────────────────────────────────────────────────────────────────
# Thread 1: CAPTURE (fast — only grabs frames)
# ──────────────────────────────────────────────────────────────────────
def capture_thread_fn(cap):
    """Grab frames from the camera as fast as possible.

    This thread does NOTHING except:
      1. cap.grab()      — tells V4L2 to dequeue the next buffer
      2. cap.retrieve()  — decodes MJPG → numpy array
      3. Store the frame in the global `latest_raw`

    No image processing happens here.  This is critical: every
    millisecond spent in this loop directly subtracts from the
    achievable capture FPS.  At 100 fps the frame budget is 10ms;
    grab+retrieve+copy takes ~5ms, leaving 5ms of slack.

    The old code put detect_ball() here (~15ms), blowing the budget
    to ~20ms → 50 fps.
    """
    global running, latest_raw, frame_seq

    print("Capture thread started...")
    frame_count = 0
    start_time = time.time()
    error_count = 0
    capture_fps = 0.0

    while running:
        try:
            # grab() is a non-blocking call that dequeues the next
            # V4L2 buffer.  It returns True if a buffer was available.
            if cap.grab():
                # retrieve() decodes the MJPG buffer into a numpy
                # array (BGR format, even for a mono camera).
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    # Store the raw frame for the other threads.
                    # We take the lock for the absolute minimum time
                    # — just the pointer swap / shallow copy.
                    with raw_lock:
                        latest_raw = frame  # no .copy() needed here —
                        # retrieve() already gives us a new array each
                        # time, so we just swap the reference.
                        # The processing thread and main thread will
                        # .copy() when they READ, so there's no race.
                        frame_seq += 1

                    frame_count += 1
                    error_count = 0
                else:
                    error_count += 1
            else:
                error_count += 1

            if error_count > 30:
                print("Too many capture errors; stopping capture thread.")
                break

            # FPS reporting — once per second.
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                capture_fps = frame_count / elapsed
                print(f"  [capture] {capture_fps:.1f} FPS")
                frame_count = 0
                start_time = time.time()

        except Exception as exc:
            error_count += 1
            print(f"Error in capture thread: {exc}")
            time.sleep(0.01)

    print("Capture thread ended")


# ──────────────────────────────────────────────────────────────────────
# Thread 2: PROCESSING (runs detect_ball at its own rate)
# ──────────────────────────────────────────────────────────────────────
def processing_thread_fn(tracker, calibration_state, spin_analyzer):
    """Run ball detection on the latest captured frame.

    This thread:
      1. Reads latest_raw (taking a copy under the lock)
      2. Runs tracker.detect_ball() — the expensive part
      3. Calls tracker.update_tracking() with wall-clock time
      4. Feeds position data to spin_analyzer for trajectory analysis

    If it can't keep up with 100 fps, it simply skips frames —
    the next iteration grabs whatever the latest frame is.  This
    is fine because:
      - The velocity math uses timestamps, not frame indices
      - Skipping frames doesn't affect capture FPS at all
      - We'd rather have accurate capture timestamps than
        process every single frame

    Expected processing rate: ~60-80 fps on a modern laptop
    (the detection pipeline takes ~12-17ms per frame).
    """
    global running

    last_processed_seq = -1
    proc_count = 0
    proc_start = time.time()

    while running:
        # Grab latest frame
        frame = None
        seq = -1
        with raw_lock:
            if latest_raw is not None and frame_seq != last_processed_seq:
                frame = latest_raw.copy()
                seq = frame_seq
                last_processed_seq = seq

        if frame is None:
            # No new frame yet — sleep briefly to avoid busy-waiting.
            # 1ms is short enough to not miss frames at 100fps (10ms
            # frame interval), but long enough to not waste CPU.
            time.sleep(0.001)
            continue

        now = time.time()

        px_per_meter_x = calibration_state.get("px_per_meter_x")
        px_per_meter_y = calibration_state.get("px_per_meter_y")

        center, radius, _ = tracker.detect_ball(frame)
        tracker.update_tracking(
            center, radius, now, px_per_meter_x, px_per_meter_y
        )

        # Feed position data to spin analyzer.
        # The analyzer accumulates positions during a flight, then
        # analyzes the trajectory when the ball is lost.
        spin_analyzer.update(center, now, px_per_meter_x, px_per_meter_y)

        proc_count += 1
        elapsed = time.time() - proc_start
        if elapsed >= 1.0:
            proc_fps = proc_count / elapsed
            bt = tracker.last_bright_T
            bt_str = f"{bt:.1f}" if bt is not None else "N/A"
            calibrated = (
                calibration_state.get("px_per_meter_x") is not None
                and calibration_state.get("px_per_meter_y") is not None
            )
            speed_str = f"{tracker.speed_m:.2f} m/s" if calibrated else "uncalibrated"
            detected = center is not None
            print(
                f"  [process] {proc_fps:.1f} FPS | "
                f"Ball: {detected} | BrightT: {bt_str} | "
                f"Speed: {speed_str}"
            )
            proc_count = 0
            proc_start = time.time()

    print("Processing thread ended")


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers (unchanged from v3.0)
# ──────────────────────────────────────────────────────────────────────
def draw_tracking_overlay(frame, tracker, calibration_state, spin_analyzer=None, show_hud=True):
    try:
        if frame is None:
            return None

        if len(frame.shape) == 2:
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            disp = frame.copy()

        # Read tracker state under the lock to avoid tearing.
        with tracker.lock:
            positions = list(tracker.positions)
            times = list(tracker.times)
            center = tracker.center
            radius = tracker.radius
            last_radius = tracker.last_radius
            speed_px = tracker.speed_px
            speed_m = tracker.speed_m
            direction_deg = tracker.direction_deg
            trail_duration = tracker.trail_duration

        recent_points = list(zip(positions, times))
        if times:
            newest_time = times[-1]
            recent_points = [
                (pos, ts)
                for pos, ts in zip(positions, times)
                if (newest_time - ts) <= trail_duration
            ]

        if len(recent_points) > 1:
            newest_time = recent_points[-1][1]
            for i in range(1, len(recent_points)):
                p0, t0 = recent_points[i - 1]
                p1, t1 = recent_points[i]
                age = newest_time - t1
                alpha = max(0.15, 1.0 - (age / max(trail_duration, 1e-6)))
                color_intensity = int(255 * alpha)
                thickness = max(1, int(4 * alpha))
                cv2.line(disp, p0, p1, (0, color_intensity, 255), thickness)

        if center is not None:
            display_radius = radius if radius > 0 else last_radius
            cv2.circle(disp, center, display_radius, (0, 255, 255), 2)
            cv2.circle(disp, center, 5, (0, 255, 0), -1)
            cv2.circle(disp, center, 8, (0, 255, 0), 2)

            if speed_px > 1:
                arrow_length = min(120, max(20, speed_px * 0.08))
                ang = np.radians(direction_deg)
                end = (
                    int(center[0] + arrow_length * np.cos(ang)),
                    int(center[1] + arrow_length * np.sin(ang)),
                )
                cv2.arrowedLine(
                    disp, center, end, (255, 0, 255), 3, tipLength=0.3
                )

        if show_hud:
            panel_x, panel_y = 10, 10
            panel_w, panel_h = 310, 150
            cv2.rectangle(
                disp,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                disp,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (255, 255, 255),
                1,
            )

            detected = center is not None
            status_text = "DETECTED" if detected else "NOT FOUND"
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.putText(
                disp,
                f"Ball: {status_text}",
                (panel_x + 8, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )

            cv2.putText(
                disp,
                f"Speed: {speed_px:.0f} px/s",
                (panel_x + 8, panel_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            calibrated = (
                calibration_state.get("px_per_meter_x") is not None
                and calibration_state.get("px_per_meter_y") is not None
            )
            speed_m_text = f"{speed_m:.2f} m/s" if calibrated else "uncalibrated"
            cv2.putText(
                disp,
                f"Real speed: {speed_m_text}",
                (panel_x + 8, panel_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            cv2.putText(
                disp,
                f"Dir: {direction_deg:.0f} deg",
                (panel_x + 8, panel_y + 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            # ── Spin classification display ──
            spin_label, spin_ay, spin_conf = "", None, ""
            if spin_analyzer is not None:
                spin_label, spin_ay, spin_conf = spin_analyzer.get_display_info()

            if spin_label:
                # Color code by spin type
                spin_colors = {
                    "TOPSPIN": (0, 140, 255),    # orange
                    "BACKSPIN": (255, 200, 0),    # cyan-ish
                    "FLAT": (200, 200, 200),      # gray
                }
                spin_color = spin_colors.get(spin_label, (200, 200, 200))
                conf_marker = "*" if spin_conf == "low" else ""
                ay_str = f" (a={spin_ay:.1f})" if spin_ay is not None else ""
                cv2.putText(
                    disp,
                    f"Spin: {spin_label}{conf_marker}{ay_str}",
                    (panel_x + 8, panel_y + 122),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    spin_color,
                    2,
                )
            else:
                cv2.putText(
                    disp,
                    "Spin: ---",
                    (panel_x + 8, panel_y + 122),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 120, 120),
                    1,
                )

            hud_hint = "H: hide HUD"
            cv2.putText(
                disp,
                hud_hint,
                (panel_x + 190, panel_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )
        else:
            cv2.putText(
                disp,
                "H: show HUD",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )

        return disp

    except Exception as exc:
        print(f"Error in draw_tracking_overlay: {exc}")
        return frame


def draw_calibration_quad(disp, calibration_state):
    points = calibration_state.get("points")
    editing = calibration_state.get("editing", False)
    if points is None:
        return disp

    for i in range(4):
        p0 = tuple(int(v) for v in points[i])
        p1 = tuple(int(v) for v in points[(i + 1) % 4])
        cv2.line(disp, p0, p1, (0, 200, 255), 2)

    for i, point in enumerate(points):
        color = (0, 255, 0) if editing else (0, 200, 200)
        center = (int(point[0]), int(point[1]))
        cv2.circle(disp, center, calibration_state["node_radius"], color, -1)
        cv2.circle(
            disp, center, calibration_state["node_radius"] + 2, (0, 0, 0), 2
        )
        cv2.putText(
            disp,
            str(i + 1),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if editing:
        cv2.putText(
            disp,
            "Edit mode: drag corners to real table edges, then press Enter to lock.",
            (10, disp.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

    return disp


def draw_dimension_input_overlay(disp, calibration_state):
    if not calibration_state.get("input_mode", False):
        return disp

    panel_x, panel_y = 80, 80
    panel_w, panel_h = 420, 180
    cv2.rectangle(
        disp,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        disp,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (255, 255, 255),
        2,
    )

    cv2.putText(
        disp,
        "Calibration Dimensions (cm)",
        (panel_x + 15, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    fields = [
        ("width_cm", "Width"),
        ("height_cm", "Height"),
    ]
    active_field = calibration_state.get("active_input_field", "width_cm")

    for idx, (field_name, label) in enumerate(fields):
        y = panel_y + 70 + idx * 45
        is_active = field_name == active_field
        box_color = (0, 200, 255) if is_active else (180, 180, 180)
        value = calibration_state.get(field_name, "")

        cv2.putText(
            disp,
            f"{label}:",
            (panel_x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
        )
        cv2.rectangle(
            disp,
            (panel_x + 120, y - 22),
            (panel_x + 280, y + 8),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            disp,
            (panel_x + 120, y - 22),
            (panel_x + 280, y + 8),
            box_color,
            2,
        )
        cv2.putText(
            disp,
            value,
            (panel_x + 130, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        disp,
        "Type numbers, Tab switches field, Enter saves, Esc cancels",
        (panel_x + 15, panel_y + 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    return disp


def handle_dimension_input(key, calibration_state):
    if not calibration_state.get("input_mode", False):
        return False

    if key == 255:
        return True

    if key == 27:
        calibration_state["input_mode"] = False
        return True

    if key == 9:
        if calibration_state["active_input_field"] == "width_cm":
            calibration_state["active_input_field"] = "height_cm"
        else:
            calibration_state["active_input_field"] = "width_cm"
        return True

    if key in (10, 13):
        try:
            width_cm = float(calibration_state["width_cm"])
            height_cm = float(calibration_state["height_cm"])
        except ValueError:
            print("Invalid dimensions. Enter numeric values in cm.")
            return True

        if width_cm <= 0 or height_cm <= 0:
            print("Dimensions must be positive.")
            return True

        calibration_state["table_width_m"] = width_cm / 100.0
        calibration_state["table_height_m"] = height_cm / 100.0
        points = calibration_state.get("points")
        if points is not None:
            px_per_meter_x, px_per_meter_y = compute_pixels_per_meter(
                points,
                calibration_state["table_width_m"],
                calibration_state["table_height_m"],
            )
            calibration_state["px_per_meter_x"] = px_per_meter_x
            calibration_state["px_per_meter_y"] = px_per_meter_y
            calibration_state["locked"] = bool(px_per_meter_x and px_per_meter_y)
        calibration_state["input_mode"] = False
        print(
            f"Calibration dimensions set: {width_cm:.1f} cm x {height_cm:.1f} cm"
        )
        if calibration_state.get("locked"):
            print(
                f"Calibration updated: {calibration_state['px_per_meter_x']:.1f} px/m x, "
                f"{calibration_state['px_per_meter_y']:.1f} px/m y"
            )
        return True

    active_field = calibration_state.get("active_input_field", "width_cm")
    if key in (8, 127):
        calibration_state[active_field] = calibration_state[active_field][:-1]
        return True

    if 48 <= key <= 57 or key == ord("."):
        calibration_state[active_field] += chr(key)
        return True

    return True


def compute_pixels_per_meter(points, table_width_m, table_height_m):
    if points is None or table_width_m <= 0 or table_height_m <= 0:
        return None, None

    top = np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    bottom = np.hypot(
        points[2][0] - points[3][0], points[2][1] - points[3][1]
    )
    left = np.hypot(
        points[3][0] - points[0][0], points[3][1] - points[0][1]
    )
    right = np.hypot(
        points[2][0] - points[1][0], points[2][1] - points[1][1]
    )

    px_per_meter_x = ((top + bottom) * 0.5) / table_width_m
    px_per_meter_y = ((left + right) * 0.5) / table_height_m
    return px_per_meter_x, px_per_meter_y


def init_rect_from_frame(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half_w = int(w * 0.2)
    half_h = int(h * 0.15)
    return [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]


def rect_mouse_callback(event, x, y, flags, calibration_state):
    if not calibration_state.get("editing", False):
        return

    points = calibration_state.get("points")
    if points is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        best_idx = -1
        best_dist = None
        for i, point in enumerate(points):
            dx = x - point[0]
            dy = y - point[1]
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist is not None and best_dist <= (
            calibration_state["node_radius"] * 2
        ) ** 2:
            calibration_state["drag_idx"] = best_idx

    elif event == cv2.EVENT_MOUSEMOVE:
        idx = calibration_state.get("drag_idx", -1)
        if idx >= 0:
            calibration_state["points"][idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        calibration_state["drag_idx"] = -1


def fourcc_to_str(value):
    value = int(value)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def create_recording_writer(output_dir, frame_size, fps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ball_tracker_{timestamp}.avi"

    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, float(fps), (int(width), int(height))
    )

    if not writer.isOpened():
        return None, None

    return writer, output_path


# ──────────────────────────────────────────────────────────────────────
# Camera configuration via v4l2-ctl
# ──────────────────────────────────────────────────────────────────────
def configure_camera_v4l2(device_index, exposure=80, fps=120):
    """Configure the OV9281 via v4l2-ctl for maximum frame rate.

    This MUST be called AFTER cv2.VideoCapture() opens the device,
    because OpenCV may reset controls during initialization.

    Parameters
    ----------
    device_index : int
        The /dev/videoN index (e.g. 4 for /dev/video4).
    exposure : int
        Exposure time in units of 0.1 ms.
        For 100 FPS: must be < 100  (< 10 ms frame period).
        For 120 FPS: must be < 83   (< 8.3 ms frame period).
        Default 80 = 8.0 ms — works for both 100 and 120 fps.
    fps : int
        Target frame rate to negotiate with the V4L2 driver.

    Physics reminder:
        f_max = 1 / t_exp
        At exposure=80 → t_exp = 8.0ms → f_max = 125 fps ✓
        At exposure=157 (factory default) → t_exp = 15.7ms → f_max = 63 fps
    """
    dev = f"/dev/video{device_index}"

    commands = [
        # 1. Switch to manual exposure (1 = Manual Mode).
        #    Must be done BEFORE setting exposure_absolute, because
        #    the control is locked (flags=inactive) in auto mode.
        ["v4l2-ctl", "-d", dev, "-c", "exposure_auto=1"],

        # 2. Set absolute exposure time.
        ["v4l2-ctl", "-d", dev, "-c", f"exposure_absolute={exposure}"],

        # 3. Set the V4L2 stream frame rate.
        #    This calls VIDIOC_S_PARM with timeperframe = 1/fps.
        ["v4l2-ctl", "-d", dev, f"--set-parm={fps}"],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                print(f"  v4l2-ctl warning: {' '.join(cmd)}")
                if result.stderr.strip():
                    print(f"    stderr: {result.stderr.strip()}")
        except FileNotFoundError:
            print(
                "  WARNING: v4l2-ctl not found. Install with: "
                "sudo apt install v4l-utils"
            )
            return False
        except Exception as e:
            print(f"  v4l2-ctl error: {e}")
            return False

    # Verify
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "-C", "auto_exposure,exposure_time_absolute"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"  v4l2-ctl verify: {result.stdout.strip()}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--get-parm"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if "Frames per second" in line:
                print(f"  v4l2-ctl verify: {line.strip()}")
    except Exception:
        pass

    return True


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────
def ball_tracker_live(
    source=0,
    width=1280,
    height=800,
    fps=120,
    backend="auto",
    threshold_k=1.5,
    motion_thresh=18,
    min_area=50,
    max_area=3000,
    circularity_min=0.65,
    table_width_m=2.74,
    table_height_m=1.525,
    exposure=80,
    record_dir="recordings",
):
    global running

    print("=" * 80)
    print("Ping Pong Ball Tracker 3.1 - Decoupled capture/processing")
    print("=" * 80)

    # ── Select backend ──
    if backend == "auto":
        if sys.platform.startswith("win"):
            api = cv2.CAP_DSHOW
            backend_name = "CAP_DSHOW"
        else:
            api = cv2.CAP_V4L2
            backend_name = "CAP_V4L2"
    else:
        backend_map = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "v4l2": cv2.CAP_V4L2,
            "any": cv2.CAP_ANY,
        }
        api = backend_map.get(backend.lower(), cv2.CAP_ANY)
        backend_name = backend

    cap = None
    cap_thread = None
    proc_thread = None
    recorder = None
    recorder_path = None
    recording = False

    try:
        # ── Open camera ──
        cap = cv2.VideoCapture(source, api)
        if not cap.isOpened():
            print(f"Cannot open camera source={source} backend={backend_name}")
            return

        # ── Set properties ──
        # Order: FOURCC → Resolution → FPS → Buffer
        # Each set() triggers a V4L2 renegotiation; setting FOURCC first
        # ensures the driver knows we want MJPG before it tries to
        # validate the resolution and FPS combo.
        if not sys.platform.startswith("win"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        # DO NOT set CAP_PROP_BUFFERSIZE = 1.
        # With a single V4L2 buffer, the driver can't double-buffer:
        # while your code reads one frame, there's nowhere to put the
        # next one, so it gets dropped.  This halves the effective FPS:
        #   BUFFERSIZE=1 → 60 FPS,  default (4) → 120 FPS.
        # The tradeoff is ~25ms extra latency (3 frames at 120fps),
        # which is negligible for ball tracking.

        # ── Configure exposure via v4l2-ctl ──
        # This MUST happen after VideoCapture opens the device.
        # OpenCV's CAP_PROP_AUTO_EXPOSURE / CAP_PROP_EXPOSURE mapping
        # is unreliable, so we shell out to v4l2-ctl directly.
        if not sys.platform.startswith("win") and isinstance(source, int):
            print("\nConfiguring camera via v4l2-ctl...")
            configure_camera_v4l2(source, exposure=exposure, fps=fps)

        # ── Report negotiated settings ──
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        print("\nCamera Negotiated Settings:")
        print(f"  Backend  : {backend_name}")
        print(f"  Source   : /dev/video{source}")
        print(f"  FOURCC   : {actual_fourcc}")
        print(f"  Res      : {int(actual_w)} x {int(actual_h)}")
        print(f"  FPS (rep): {actual_fps:.1f}")
        print(f"  Exposure : {exposure} (= {exposure * 0.1:.1f} ms)")

        print("\nCalibration Settings:")
        print(f"  Table width  : {table_width_m:.3f} m")
        print(f"  Table height : {table_height_m:.3f} m")

        print("\nControls:")
        print("  q / ESC : quit")
        print("  d       : edit calibration rectangle")
        print("  t       : edit real rectangle dimensions")
        print("  Enter   : lock calibration rectangle")
        print("  h       : show/hide info HUD")
        print("  f       : start/stop recording")
        print("  [ / ]   : decrease/increase motion threshold")
        print("  - / =   : decrease/increase brightness k")
        print("=" * 80)

        # ── Create tracker and calibration state ──
        tracker = BallTracker(
            threshold_k=threshold_k,
            motion_thresh=motion_thresh,
            min_area=min_area,
            max_area=max_area,
            circularity_min=circularity_min,
        )

        window_name = "Ping Pong Ball Tracker 3.1"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(actual_w), int(actual_h))

        calibration_state = {
            "editing": False,
            "locked": False,
            "input_mode": False,
            "points": None,
            "drag_idx": -1,
            "node_radius": 8,
            "px_per_meter_x": None,
            "px_per_meter_y": None,
            "table_width_m": table_width_m,
            "table_height_m": table_height_m,
            "width_cm": f"{table_width_m * 100.0:.1f}",
            "height_cm": f"{table_height_m * 100.0:.1f}",
            "active_input_field": "width_cm",
        }
        cv2.setMouseCallback(window_name, rect_mouse_callback, calibration_state)

        # ── Flush initial frames ──
        # The first few frames from a USB camera are often garbage
        # (auto-exposure settling, buffer prefill).  Discard them.
        for _ in range(10):
            cap.read()

        # ── Start threads ──
        running = True

        # Create spin analyzer
        spin_analyzer = SpinAnalyzer()

        # Thread 1: capture (fast — grab+retrieve only)
        cap_thread = threading.Thread(
            target=capture_thread_fn,
            args=(cap,),
            daemon=True,
        )
        cap_thread.start()

        # Thread 2: processing (detect_ball at its own pace)
        proc_thread = threading.Thread(
            target=processing_thread_fn,
            args=(tracker, calibration_state, spin_analyzer),
            daemon=True,
        )
        proc_thread.start()

        time.sleep(0.2)  # let threads spin up

        # ── Main loop: display ──
        display_count = 0
        t0 = time.time()
        display_fps = 0.0
        show_hud = True

        while running:
            # Grab the latest raw frame for display.
            frame = None
            with raw_lock:
                if latest_raw is not None:
                    frame = latest_raw.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            disp = draw_tracking_overlay(
                frame, tracker, calibration_state, spin_analyzer=spin_analyzer, show_hud=show_hud
            )
            disp = draw_calibration_quad(disp, calibration_state)
            disp = draw_dimension_input_overlay(disp, calibration_state)

            display_count += 1
            dt = time.time() - t0
            if dt >= 0.5:
                display_fps = display_count / dt
                display_count = 0
                t0 = time.time()

            cv2.putText(
                disp,
                f"Display FPS: {display_fps:.1f}",
                (10, disp.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if recording:
                cv2.circle(disp, (disp.shape[1] - 24, 24), 8, (0, 0, 255), -1)
                cv2.putText(
                    disp,
                    "REC",
                    (disp.shape[1] - 68, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(window_name, disp)

            if recording and recorder is not None:
                recorder.write(disp)

            key = cv2.waitKey(1) & 0xFF
            if calibration_state.get("input_mode", False):
                handle_dimension_input(key, calibration_state)
                continue

            if key == ord("q") or key == 27:
                print("\nQuit requested...")
                break

            if key == ord("h") or key == ord("H"):
                show_hud = not show_hud

            if key == ord("f") or key == ord("F"):
                if not recording:
                    recorder, recorder_path = create_recording_writer(
                        output_dir=record_dir,
                        frame_size=(disp.shape[1], disp.shape[0]),
                        fps=max(1.0, display_fps if display_fps > 1.0 else float(fps)),
                    )
                    if recorder is None:
                        print("Failed to start recording: could not open video writer.")
                    else:
                        recording = True
                        print(f"Recording started: {recorder_path}")
                else:
                    recording = False
                    if recorder is not None:
                        recorder.release()
                        print(f"Recording saved: {recorder_path}")
                    recorder = None
                    recorder_path = None

            if key == ord("t") or key == ord("T"):
                calibration_state["input_mode"] = True
                calibration_state["active_input_field"] = "width_cm"

            if key == ord("d"):
                if calibration_state["points"] is None:
                    calibration_state["points"] = init_rect_from_frame(frame)
                calibration_state["editing"] = True
                calibration_state["locked"] = False
                calibration_state["px_per_meter_x"] = None
                calibration_state["px_per_meter_y"] = None

            if key in (10, 13) and calibration_state["editing"]:
                px_per_meter_x, px_per_meter_y = compute_pixels_per_meter(
                    calibration_state["points"],
                    calibration_state["table_width_m"],
                    calibration_state["table_height_m"],
                )
                calibration_state["px_per_meter_x"] = px_per_meter_x
                calibration_state["px_per_meter_y"] = px_per_meter_y
                calibration_state["editing"] = False
                calibration_state["locked"] = True
                if px_per_meter_x and px_per_meter_y:
                    print(
                        f"Calibration locked: {px_per_meter_x:.1f} px/m x, "
                        f"{px_per_meter_y:.1f} px/m y"
                    )

            if key == ord("["):
                tracker.motion_thresh = max(1, tracker.motion_thresh - 1)
                print(f"motion_thresh -> {tracker.motion_thresh}")
            elif key == ord("]"):
                tracker.motion_thresh = min(255, tracker.motion_thresh + 1)
                print(f"motion_thresh -> {tracker.motion_thresh}")

            if key == ord("-") or key == ord("_"):
                tracker.threshold_k = max(0.1, tracker.threshold_k - 0.1)
                print(f"threshold_k -> {tracker.threshold_k:.2f}")
            elif key == ord("=") or key == ord("+"):
                tracker.threshold_k = min(5.0, tracker.threshold_k + 0.1)
                print(f"threshold_k -> {tracker.threshold_k:.2f}")

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed...")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        print(f"Fatal error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        running = False

        for thread in [cap_thread, proc_thread]:
            try:
                if thread is not None:
                    thread.join(timeout=1.0)
            except Exception:
                pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            if recorder is not None:
                recorder.release()
                print(f"Recording saved: {recorder_path}")
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("=" * 80)
        print("Session ended - cleanup complete")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live ping pong ball tracker with m/s calibration"
    )
    parser.add_argument("--src", default=0, help="Camera source index or path")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=800, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=120, help="Requested FPS")
    parser.add_argument(
        "--backend", type=str, default="auto",
        help="auto | dshow | msmf | v4l2 | any",
    )
    parser.add_argument(
        "--exposure", type=int, default=80,
        help="Exposure time in 0.1ms units (80=8ms). Must be < 1000/fps for target FPS.",
    )
    parser.add_argument("--k", type=float, default=1.5, help="Brightness gate factor k")
    parser.add_argument("--motion", type=int, default=18, help="Motion threshold")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum contour area")
    parser.add_argument("--max-area", type=int, default=3000, help="Maximum contour area")
    parser.add_argument("--circ", type=float, default=0.65, help="Minimum circularity")
    parser.add_argument("--table-width-m", type=float, default=2.74, help="Real table width in meters")
    parser.add_argument("--table-height-m", type=float, default=1.525, help="Real table height in meters")
    parser.add_argument("--record-dir", default="recordings", help="Directory for saved recordings")

    args = parser.parse_args()

    try:
        src_val = int(args.src)
    except Exception:
        src_val = args.src

    ball_tracker_live(
        source=src_val,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        threshold_k=args.k,
        motion_thresh=args.motion,
        min_area=args.min_area,
        max_area=args.max_area,
        circularity_min=args.circ,
        table_width_m=args.table_width_m,
        table_height_m=args.table_height_m,
        exposure=args.exposure,
        record_dir=args.record_dir,
    )