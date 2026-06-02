import cv2
import numpy as np
import time
import argparse
import threading
import signal
import sys
from collections import deque

# -----------------------------
# Globals
# -----------------------------
running = True
latest_frame = None
frame_lock = threading.Lock()

def signal_handler(sig, frame):
    global running
    print("\n\nForce quit detected! Exiting...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -----------------------------
# Ball Tracker
# -----------------------------
class BallTracker:
    def __init__(self,
                 min_radius=2,
                 max_radius=80,
                 history=64,
                 max_jump=200,
                 max_no_detect=0.5,
                 min_area=20,
                 adaptive_brightness=True,
                 brightness_alpha=0.03,
                 brightness_percentile=80,
                 brightness_scale=1.0):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.history = deque(maxlen=history)
        self.max_jump = max_jump
        self.max_no_detect = max_no_detect
        self.last_detect_time = None
        self.min_area = min_area

        # Brightness thresholding
        self.adaptive_brightness = adaptive_brightness
        self.brightness_alpha = brightness_alpha
        self.brightness_percentile = brightness_percentile
        self.brightness_scale = brightness_scale
        self.last_bright_T = None  # for debug print

        # Tracking state
        self.last_center = None
        self.last_radius = None

        # THREAD SAFETY: lock for all shared tracker state
        self.lock = threading.Lock()

    def detect_ball(self, frame):
        """
        Return (center, radius, mask) where center is (x,y) or None.
        NOTE: This is pure computation on 'frame' except for brightness EMA state.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Determine brightness threshold (adaptive or fixed)
        if self.adaptive_brightness:
            p = np.percentile(gray_blur, self.brightness_percentile)
            T_new = float(p * self.brightness_scale)

            # last_bright_T is shared state -> lock
            with self.lock:
                if self.last_bright_T is None:
                    self.last_bright_T = T_new
                else:
                    self.last_bright_T = (1 - self.brightness_alpha) * self.last_bright_T + self.brightness_alpha * T_new
                T = self.last_bright_T
        else:
            with self.lock:
                self.last_bright_T = 200.0
                T = self.last_bright_T

        _, mask = cv2.threshold(gray_blur, int(T), 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        best_radius = None
        best_score = None

        # last_center is shared state; snapshot it once
        with self.lock:
            last_center_snapshot = self.last_center

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            radius = float(radius)

            if radius < self.min_radius or radius > self.max_radius:
                continue

            score = radius

            if last_center_snapshot is not None:
                dx = x - last_center_snapshot[0]
                dy = y - last_center_snapshot[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_jump:
                    continue
                score = score - 0.01 * dist

            if best_score is None or score > best_score:
                best_score = score
                best_center = (int(x), int(y))
                best_radius = radius

        return best_center, best_radius, mask

    def update_tracking(self, center, radius, timestamp):
        """
        Update track history with new detection (or None).
        MUST be locked because UI thread reads these.
        """
        with self.lock:
            if center is not None:
                self.history.appendleft((center, radius, timestamp))
                self.last_center = center
                self.last_radius = radius
                self.last_detect_time = timestamp
            else:
                if self.last_detect_time is not None:
                    if (timestamp - self.last_detect_time) > self.max_no_detect:
                        self.last_center = None
                        self.last_radius = None
                        self.last_detect_time = None

    def get_predicted_position_from_snapshot(self, history_snapshot):
        """
        Pure function: predict from a provided snapshot.
        """
        if len(history_snapshot) < 2:
            if len(history_snapshot) == 1:
                return history_snapshot[0][0]
            return None

        (c1, _, t1) = history_snapshot[0]
        (c2, _, t2) = history_snapshot[1]
        dt = t1 - t2
        if dt <= 1e-6:
            return c1

        # v = (x1-x2)/dt, (y1-y2)/dt
        # Not used for extrapolation here; return c1.
        return c1

# -----------------------------
# Capture Thread
# -----------------------------
def capture_thread(cap, tracker):
    global running, latest_frame

    print("Capture thread started...")
    frame_count = 0
    start_time = time.time()
    error_count = 0

    while running:
        try:
            if cap.grab():
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    now = time.time()

                    center, radius, _ = tracker.detect_ball(frame)
                    tracker.update_tracking(center, radius, now)

                    with frame_lock:
                        latest_frame = frame  # avoid copy here; UI copies

                    frame_count += 1
                    error_count = 0

                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:
                        capture_fps = frame_count / elapsed
                        with tracker.lock:
                            bt = tracker.last_bright_T
                        bt_str = f"{bt:.1f}" if bt is not None else "N/A"
                        print(f"Capture FPS: {capture_fps:.1f} | Ball detected: {center is not None} | BrightT: {bt_str}")
                        frame_count = 0
                        start_time = time.time()
                else:
                    error_count += 1
            else:
                error_count += 1

            if error_count > 30:
                print("Too many capture errors; stopping capture thread.")
                break

        except Exception as e:
            error_count += 1
            print(f"Error in capture thread: {e}")
            time.sleep(0.01)

    print("Capture thread ended")

# -----------------------------
# Drawing / UI
# -----------------------------
def draw_tracking_overlay(frame, center_snapshot, history_snapshot, pred_snapshot,
                          show_history=True, show_predicted=True):
    if center_snapshot is not None:
        cv2.circle(frame, center_snapshot, 8, (0, 255, 0), -1)

    if show_history:
        for i, item in enumerate(history_snapshot):
            c, r, t = item
            if c is None:
                continue
            thickness = max(1, int(5 - i / 10))
            cv2.circle(frame, c, thickness, (0, 0, 255), -1)

    if show_predicted and pred_snapshot is not None:
        cv2.circle(frame, pred_snapshot, 6, (255, 0, 0), 2)

    return frame

# -----------------------------
# Main Loop
# -----------------------------
def ball_tracker_live(src=0,
                      backend="v4l2",
                      width=1280,
                      height=720,
                      fps=120,
                      exposure=-1,
                      min_radius=2,
                      max_radius=80,
                      min_area=20,
                      adaptive_brightness=True,
                      brightness_alpha=0.03,
                      brightness_percentile=80,
                      brightness_scale=1.0,
                      show_mask=False):
    global running, latest_frame

    tracker = BallTracker(
        min_radius=min_radius,
        max_radius=max_radius,
        min_area=min_area,
        adaptive_brightness=adaptive_brightness,
        brightness_alpha=brightness_alpha,
        brightness_percentile=brightness_percentile,
        brightness_scale=brightness_scale,
    )

    if backend.lower() == "v4l2":
        api_pref = cv2.CAP_V4L2
    elif backend.lower() == "gstreamer":
        api_pref = cv2.CAP_GSTREAMER
    elif backend.lower() == "opencv":
        api_pref = cv2.CAP_ANY
    else:
        api_pref = cv2.CAP_ANY

    cap = cv2.VideoCapture(src, api_pref)
    if not cap.isOpened():
        print("ERROR: Could not open capture device.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

    t = threading.Thread(target=capture_thread, args=(cap, tracker), daemon=True)
    t.start()

    print("Main UI loop started. Press 'q' to quit.")

    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.001)
            continue

        # Snapshot tracker state (FAST), then draw without holding locks
        with tracker.lock:
            center_snapshot = tracker.last_center
            history_snapshot = list(tracker.history)  # <-- snapshot prevents deque mutation error

        pred_snapshot = tracker.get_predicted_position_from_snapshot(history_snapshot)

        frame_disp = draw_tracking_overlay(
            frame,
            center_snapshot=center_snapshot,
            history_snapshot=history_snapshot,
            pred_snapshot=pred_snapshot,
        )

        cv2.imshow("Ball Tracker", frame_disp)

        if show_mask:
            # Mask display: this calls detect_ball again; for debugging only
            _, _, mask = tracker.detect_ball(frame)
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    running = False
    try:
        t.join(timeout=1.0)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 80)
    print("Session ended - cleanup complete")
    print("=" * 80)

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live ball tracker.")
    p.add_argument("--src", default=0, help="Camera index or path. Default 0.")
    p.add_argument("--backend", default="v4l2", choices=["v4l2", "gstreamer", "opencv"], help="OpenCV backend.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("--exposure", type=float, default=-1)
    p.add_argument("--min-radius", type=int, default=2)
    p.add_argument("--max-radius", type=int, default=80)
    p.add_argument("--min-area", type=int, default=20)

    p.add_argument("--no-adaptive-brightness", action="store_true", help="Disable adaptive brightness thresholding.")
    p.add_argument("--brightness-alpha", type=float, default=0.03)
    p.add_argument("--brightness-percentile", type=float, default=80)
    p.add_argument("--brightness-scale", type=float, default=1.0)

    p.add_argument("--show-mask", action="store_true", help="Show threshold mask window.")
    return p.parse_args()

def main():
    args = parse_args()
    src = args.src
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    ball_tracker_live(
        src=src,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        exposure=args.exposure,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_area=args.min_area,
        adaptive_brightness=not args.no_adaptive_brightness,
        brightness_alpha=args.brightness_alpha,
        brightness_percentile=args.brightness_percentile,
        brightness_scale=args.brightness_scale,
        show_mask=args.show_mask,
    )

if __name__ == "__main__":
    main()
