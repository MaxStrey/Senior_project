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
# Ball Tracker (motion + brightness gating + circularity)
# -----------------------------
class BallTracker:
    def __init__(
        self,
        threshold_k=1.5,          # relative brightness threshold: T = mean + k*std
        motion_thresh=18,         # absdiff threshold in grayscale (10–30 typical)
        min_area=50,
        max_area=3000,
        circularity_min=0.65,
        history_len=30,
        lost_timeout=0.10         # seconds until velocity/speed reset if ball not seen
    ):
        self.threshold_k = threshold_k
        self.motion_thresh = motion_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_min = circularity_min
        self.lost_timeout = lost_timeout

        # Tracking history
        self.positions = deque(maxlen=history_len)
        self.times = deque(maxlen=history_len)

        # State
        self.center = None
        self.radius = 0
        self.last_radius = 15

        self.velocity = (0.0, 0.0)  # px/s
        self.speed = 0.0
        self.direction_deg = 0.0

        self.last_seen_time = None

        # For motion
        self.prev_gray = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Debug last thresholds
        self.last_bright_T = None

    def detect_ball(self, frame):
        """
        Robust detection:
          1) CLAHE normalize
          2) frame differencing -> motion mask
          3) relative brightness gate -> bright mask
          4) combine -> candidate mask
          5) contours -> circularity + area -> best candidate
        Returns: (center, radius, debug_mask)
        """
        try:
            # Ensure grayscale
            if frame is None:
                return None, 0, None

            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Local contrast normalization (helps with exposure/lighting shifts)
            gray_eq = self.clahe.apply(gray)

            # Initialize prev frame for motion
            if self.prev_gray is None:
                self.prev_gray = gray_eq
                return None, 0, None

            # Motion mask
            diff = cv2.absdiff(gray_eq, self.prev_gray)
            self.prev_gray = gray_eq

            diff = cv2.GaussianBlur(diff, (7, 7), 0)
            _, motion = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)

            # Brightness gate (relative, not absolute 240)
            mean, std = cv2.meanStdDev(gray_eq)
            mean = float(mean)
            std = float(std)
            T = mean + self.threshold_k * std
            T = max(60.0, min(240.0, T))  # clamp
            self.last_bright_T = T

            _, bright = cv2.threshold(gray_eq, T, 255, cv2.THRESH_BINARY)

            # Combine: moving AND bright-ish
            mask = cv2.bitwise_and(motion, bright)

            # Morph clean
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_center = None
            best_radius = 0
            best_score = -1.0

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area or area > self.max_area:
                    continue

                per = cv2.arcLength(c, True)
                if per <= 0:
                    continue

                circ = 4.0 * np.pi * area / (per * per)
                if circ < self.circularity_min:
                    continue

                (x, y), r = cv2.minEnclosingCircle(c)
                if r <= 0:
                    continue

                M = cv2.moments(c)
                if M["m00"] <= 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Score: prefer circular + larger area (within bounds)
                score = circ * area
                if score > best_score:
                    best_score = score
                    best_center = (cx, cy)
                    best_radius = int(r)

            return best_center, best_radius, mask

        except Exception as e:
            print(f"Error in detect_ball: {e}")
            return None, 0, None

    def update_tracking(self, center, radius, current_time):
        try:
            if center is not None:
                self.positions.append(center)
                self.times.append(current_time)

                self.center = center
                self.radius = radius
                if radius > 0:
                    self.last_radius = radius

                self.last_seen_time = current_time

                # Velocity from last 2 samples
                if len(self.positions) >= 2:
                    (x1, y1) = self.positions[-2]
                    (x2, y2) = self.positions[-1]
                    t1 = self.times[-2]
                    t2 = self.times[-1]

                    dt = t2 - t1
                    if dt > 1e-6:
                        vx = (x2 - x1) / dt
                        vy = (y2 - y1) / dt
                        self.velocity = (vx, vy)
                        self.speed = float(np.hypot(vx, vy))

                        ang = np.degrees(np.arctan2(vy, vx))
                        if ang < 0:
                            ang += 360.0
                        self.direction_deg = float(ang)
            else:
                self.center = None
                self.radius = 0

                # If we've "lost" the ball for long enough, zero velocity
                if self.last_seen_time is not None and (current_time - self.last_seen_time) > self.lost_timeout:
                    self.velocity = (0.0, 0.0)
                    self.speed = 0.0

        except Exception as e:
            print(f"Error in update_tracking: {e}")

# -----------------------------
# Capture thread
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

                    center, radius, mask = tracker.detect_ball(frame)
                    tracker.update_tracking(center, radius, now)

                    with frame_lock:
                        latest_frame = frame.copy()

                    frame_count += 1
                    error_count = 0

                 
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:
                        capture_fps = frame_count / elapsed
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
def draw_tracking_overlay(frame, tracker, center, radius):
    try:
        if frame is None:
            return None

        if len(frame.shape) == 2:
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            disp = frame.copy()

        # Trail
        if len(tracker.positions) > 1:
            for i in range(1, len(tracker.positions)):
                p0 = tracker.positions[i - 1]
                p1 = tracker.positions[i]
                if p0 is None or p1 is None:
                    continue
                alpha = i / len(tracker.positions)
                color_intensity = int(255 * alpha)
                thickness = max(1, int(3 * alpha))
                cv2.line(disp, p0, p1, (0, color_intensity, 255), thickness)

        # Ball annotation
        if center is not None:
            display_radius = radius if radius > 0 else tracker.last_radius
            cv2.circle(disp, center, display_radius, (0, 255, 255), 2)
            cv2.circle(disp, center, 5, (0, 255, 0), -1)
            cv2.circle(disp, center, 8, (0, 255, 0), 2)

            if tracker.speed > 1:
                arrow_length = min(120, max(20, tracker.speed * 0.08))
                ang = np.radians(tracker.direction_deg)
                end = (int(center[0] + arrow_length * np.cos(ang)),
                       int(center[1] + arrow_length * np.sin(ang)))
                cv2.arrowedLine(disp, center, end, (255, 0, 255), 3, tipLength=0.3)
                cv2.putText(disp, f"{tracker.speed:.0f} px/s", (end[0] + 10, end[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Info panel
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 420, 190
        cv2.rectangle(disp, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(disp, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

        if center is not None:
            status_text = "Ball: DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "Ball: NOT FOUND"
            status_color = (0, 0, 255)

        cv2.putText(disp, status_text, (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Show thresholds
        bt = tracker.last_bright_T if tracker.last_bright_T is not None else -1
        cv2.putText(disp, f"Bright gate T: {bt:.1f}  (T = mean + k*std, k={tracker.threshold_k:.2f})",
                    (panel_x + 10, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.putText(disp, f"Motion thresh: {tracker.motion_thresh}",
                    (panel_x + 10, panel_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Speed/dir/vel
        cv2.putText(disp, f"Speed: {tracker.speed:.1f} px/s",
                    (panel_x + 10, panel_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(disp, f"Direction: {tracker.direction_deg:.1f} deg",
                    (panel_x + 10, panel_y + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        vx, vy = tracker.velocity
        cv2.putText(disp, f"Velocity: ({vx:.1f}, {vy:.1f})",
                    (panel_x + 10, panel_y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return disp

    except Exception as e:
        print(f"Error in draw_tracking_overlay: {e}")
        return frame

def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def _ray_segment_intersect(p, v, a, b):
    r = (float(v[0]), float(v[1]))
    s = (float(b[0] - a[0]), float(b[1] - a[1]))
    rxs = _cross2(r, s)
    if abs(rxs) < 1e-9:
        return None

    ap = (float(a[0] - p[0]), float(a[1] - p[1]))
    t = _cross2(ap, s) / rxs
    u = _cross2(ap, r) / rxs

    if t >= 0.0 and 0.0 <= u <= 1.0:
        hit = (p[0] + t * r[0], p[1] + t * r[1])
        return t, hit
    return None

def _reflect_velocity(v, a, b):
    s = (float(b[0] - a[0]), float(b[1] - a[1]))
    n = (s[1], -s[0])
    n_norm = np.hypot(n[0], n[1])
    if n_norm < 1e-9:
        return v
    n_hat = (n[0] / n_norm, n[1] / n_norm)
    dot = v[0] * n_hat[0] + v[1] * n_hat[1]
    return (v[0] - 2.0 * dot * n_hat[0], v[1] - 2.0 * dot * n_hat[1])

def compute_trajectory(points, center, velocity, min_speed=1.0):
    if points is None or center is None:
        return None

    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < min_speed:
        return None

    p = (float(center[0]), float(center[1]))
    v = (float(velocity[0]), float(velocity[1]))

    best = None
    best_edge_idx = None

    for i in range(4):
        a = points[i]
        b = points[(i + 1) % 4]
        hit = _ray_segment_intersect(p, v, a, b)
        if hit is None:
            continue
        t, pt = hit
        if best is None or t < best[0]:
            best = (t, pt, a, b)
            best_edge_idx = i

    if best is None:
        return None

    _, hit_pt, a, b = best
    v_ref = _reflect_velocity(v, a, b)
    return hit_pt, v_ref, best_edge_idx

def draw_rect_and_trajectory(disp, rect_state, tracker):
    if disp is None:
        return disp

    points = rect_state.get("points")
    editing = rect_state.get("editing", False)

    if points is not None:
        # Draw rectangle/quad
        for i in range(4):
            p0 = tuple(int(x) for x in points[i])
            p1 = tuple(int(x) for x in points[(i + 1) % 4])
            cv2.line(disp, p0, p1, (0, 200, 255), 2)

        # Draw nodes
        for i, p in enumerate(points):
            color = (0, 255, 0) if editing else (0, 200, 200)
            cv2.circle(disp, (int(p[0]), int(p[1])), rect_state["node_radius"], color, -1)
            cv2.circle(disp, (int(p[0]), int(p[1])), rect_state["node_radius"] + 2, (0, 0, 0), 2)

    # Trajectory (only when locked or after editing)
    if points is not None and not editing:
        traj = compute_trajectory(points, tracker.center, tracker.velocity)
        if traj is not None:
            hit_pt, v_ref, edge_idx = traj
            start = (int(tracker.center[0]), int(tracker.center[1]))
            hit = (int(hit_pt[0]), int(hit_pt[1]))
            cv2.line(disp, start, hit, (0, 255, 255), 2)

            ref_len = 200
            v_ref_norm = np.hypot(v_ref[0], v_ref[1])
            if v_ref_norm > 1e-6:
                ref_end = (
                    int(hit[0] + ref_len * v_ref[0] / v_ref_norm),
                    int(hit[1] + ref_len * v_ref[1] / v_ref_norm),
                )
                cv2.line(disp, hit, ref_end, (255, 255, 0), 2)
                cv2.circle(disp, hit, 6, (0, 0, 255), -1)

                side_names = ["top", "right", "bottom", "left"]
                side = side_names[edge_idx] if edge_idx is not None else "unknown"
                cv2.putText(
                    disp,
                    f"Hit side: {side}",
                    (hit[0] + 10, hit[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

    # Editing hint
    if editing:
        cv2.putText(
            disp,
            "Edit mode: drag nodes. Press Enter to lock. Press d to re-open later.",
            (10, disp.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

    return disp

def fourcc_to_str(v):
    v = int(v)
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])

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

def rect_mouse_callback(event, x, y, flags, rect_state):
    if not rect_state.get("editing", False):
        return
    points = rect_state.get("points")
    if points is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        best_idx = -1
        best_dist = None
        for i, p in enumerate(points):
            dx = x - p[0]
            dy = y - p[1]
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist is not None and best_dist <= (rect_state["node_radius"] * 2) ** 2:
            rect_state["drag_idx"] = best_idx

    elif event == cv2.EVENT_MOUSEMOVE:
        idx = rect_state.get("drag_idx", -1)
        if idx >= 0:
            rect_state["points"][idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        rect_state["drag_idx"] = -1

# -----------------------------
# Main
# -----------------------------
def ball_tracker_live(
    source=0,
    width=1280,
    height=800,
    fps=100,
    backend="auto",
    threshold_k=1.5,
    motion_thresh=18,
    min_area=50,
    max_area=3000,
    circularity_min=0.65,
):
    global running

    print("=" * 80)
    print("Ping Pong Ball Tracker - Motion-based (robust to lighting)")
    print("=" * 80)

    # Backend selection
    # - Windows: CAP_DSHOW often best
    # - Linux: CAP_V4L2 best
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
    capture_thread_obj = None

    try:
        print(f"DEBUG: source={source!r} type={type(source)} backend={backend_name}")
        cap = cv2.VideoCapture(source, api)

        if not cap.isOpened():
            print(f"❌ Cannot open camera source={source} backend={backend_name}")
            return

        # Request mode (NOTE: drivers may ignore these; we print actual)
        # On Linux, MJPG often required for high-FPS at high res.
        # On Windows, forcing FOURCC sometimes helps, sometimes hurts; you can toggle.
        if not sys.platform.startswith("win"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        print("\nCamera Negotiated Settings (what you actually got):")
        print(f"  Backend  : {backend_name}")
        print(f"  Source   : {source}")
        print(f"  FOURCC   : {actual_fourcc}")
        print(f"  Res      : {int(actual_w)} x {int(actual_h)}")
        print(f"  FPS (rep): {actual_fps:.1f}   (reported; may lie on Windows)")

        print("\nTracking Settings:")
        print(f"  Motion threshold        : {motion_thresh}")
        print(f"  Brightness k (mean+k*std): {threshold_k:.2f}")
        print(f"  Area range              : {min_area}..{max_area}")
        print(f"  Circularity min         : {circularity_min:.2f}")

        print("\nControls:")
        print("  q / ESC : quit")
        print("  [ / ]   : decrease/increase motion threshold")
        print("  - / =   : decrease/increase brightness k")
        print("  d       : edit rectangle (drag nodes)")
        print("  Enter   : lock rectangle")
        print("=" * 80)

        tracker = BallTracker(
            threshold_k=threshold_k,
            motion_thresh=motion_thresh,
            min_area=min_area,
            max_area=max_area,
            circularity_min=circularity_min,
        )

        # Window
        window_name = "Ping Pong Ball Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(actual_w), int(actual_h))

        rect_state = {
            "editing": False,
            "points": None,
            "drag_idx": -1,
            "node_radius": 8,
        }
        cv2.setMouseCallback(window_name, rect_mouse_callback, rect_state)

        # Warm up
        for _ in range(10):
            cap.read()

        # Start capture thread
        running = True
        capture_thread_obj = threading.Thread(target=capture_thread, args=(cap, tracker), daemon=True)
        capture_thread_obj.start()

        time.sleep(0.1)

        display_count = 0
        t0 = time.time()
        display_fps = 0.0

        while running:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            # Read tracker state
            center = tracker.center
            radius = tracker.radius if tracker.radius > 0 else tracker.last_radius

            # Draw
            disp = draw_tracking_overlay(frame, tracker, center, radius)
            disp = draw_rect_and_trajectory(disp, rect_state, tracker)

            # Display FPS estimate
            display_count += 1
            dt = time.time() - t0
            if dt >= 0.5:
                display_fps = display_count / dt
                display_count = 0
                t0 = time.time()

            cv2.putText(disp, f"Display FPS: {display_fps:.1f}",
                        (10, disp.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\nQuit requested...")
                break

            if key == ord("d"):
                if rect_state["points"] is None:
                    rect_state["points"] = init_rect_from_frame(frame)
                rect_state["editing"] = True

            if key in (10, 13) and rect_state["editing"]:
                rect_state["editing"] = False

            # Motion threshold adjust: [ / ]
            if key == ord("["):
                tracker.motion_thresh = max(1, tracker.motion_thresh - 1)
                print(f"motion_thresh -> {tracker.motion_thresh}")
            elif key == ord("]"):
                tracker.motion_thresh = min(255, tracker.motion_thresh + 1)
                print(f"motion_thresh -> {tracker.motion_thresh}")

            # Brightness k adjust: - / =
            if key == ord("-") or key == ord("_"):
                tracker.threshold_k = max(0.1, tracker.threshold_k - 0.1)
                print(f"threshold_k -> {tracker.threshold_k:.2f}")
            elif key == ord("=") or key == ord("+"):
                tracker.threshold_k = min(5.0, tracker.threshold_k + 0.1)
                print(f"threshold_k -> {tracker.threshold_k:.2f}")

            # Window closed?
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed...")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        running = False

        try:
            if capture_thread_obj is not None:
                capture_thread_obj.join(timeout=1.0)
        except Exception:
            pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("=" * 80)
        print("Session ended - cleanup complete")
        print("=" * 80)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ping pong ball tracker (motion-based)")
    parser.add_argument("--src", default=0, help="Camera source index or path (e.g., 0 or /dev/video2)")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=800, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=100, help="Requested FPS")
    parser.add_argument("--backend", type=str, default="auto",
                        help="auto | dshow | msmf | v4l2 | any")
    parser.add_argument("--k", type=float, default=1.5,
                        help="Brightness gate factor k in T = mean + k*std (lower -> more candidates)")
    parser.add_argument("--motion", type=int, default=18,
                        help="Motion threshold for absdiff (lower -> more sensitive)")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum contour area")
    parser.add_argument("--max-area", type=int, default=3000, help="Maximum contour area")
    parser.add_argument("--circ", type=float, default=0.65, help="Minimum circularity")

    args = parser.parse_args()

    # Allow src to be int if possible (common case)
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
    )
