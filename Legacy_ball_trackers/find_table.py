#!/usr/bin/env python3
"""
find_table.py

Goal:
  Look at a live camera feed and automatically outline a "table" region
  using contrast/edges (not a learned model).

Approach (simple + practical):
  1) Convert to grayscale + CLAHE (stabilizes contrast under lighting changes)
  2) Blur
  3) Canny edges (thresholds controlled by trackbars)
  4) Morphological close/dilate to connect edges
  5) Find contours, approximate polygons, pick the best quadrilateral candidate
  6) Draw outline + corner points
  7) Press 's' to save detected corners to table_corners.json (if found)

Notes:
  - This is NOT color segmentation. It's edge/contrast based.
  - It works best when the table boundary forms a strong outline (edges/corners).
  - If your "table" is just a dark sheet on a bed with weak edges, this may fail.
    In that case you either: (a) add border tape, or (b) do color segmentation
    in HSV / Lab using the sheet color, or (c) click corners manually once.
"""

import cv2
import numpy as np
import argparse
import json
import time
import sys
from pathlib import Path

def order_points(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    pts: array-like shape (4,2)
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)          # x+y
    diff = np.diff(pts, axis=1)  # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def polygon_area(pts):
    """Shoelace formula area for 4 points (or N points)."""
    pts = np.array(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def angle_cos(p0, p1, p2):
    """
    Cosine of angle at p1 formed by p0-p1-p2.
    Near 0 => ~90 degrees (good for rectangles).
    """
    d1 = p0 - p1
    d2 = p2 - p1
    denom = (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9)
    return float(np.dot(d1, d2) / denom)

def find_best_quad(edge_img, min_area_frac=0.05, max_area_frac=0.98, debug=False):
    """
    From an edge/binary image, find the best quadrilateral contour candidate.

    Returns:
      best_quad (np.ndarray shape (4,2) float32) or None
      best_score (float) or None
    """
    h, w = edge_img.shape[:2]
    img_area = float(h * w)

    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best_quad = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * img_area or area > max_area_frac * img_area:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        # Approximate polygon; epsilon controlled indirectly by perimeter fraction
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        quad = order_points(quad)

        # Score rectangularity: angles near 90 degrees and large area
        # Use cos(angle) near 0 for right angle; penalize large cos magnitude.
        cosines = []
        for i in range(4):
            p0 = quad[(i - 1) % 4]
            p1 = quad[i]
            p2 = quad[(i + 1) % 4]
            cosines.append(abs(angle_cos(p0, p1, p2)))  # abs(cos) close to 0 is good
        mean_abs_cos = float(np.mean(cosines))

        # Score: prefer large area and right angles.
        # mean_abs_cos in [0..1]. We want small. Convert to (1 - mean_abs_cos).
        rect_score = max(0.0, 1.0 - mean_abs_cos)

        # Normalize area to [0..1] range relative to image
        area_score = float(area / img_area)

        # Combine (weights can be tuned)
        score = (0.7 * rect_score) + (0.3 * area_score)

        if debug:
            print(f"candidate area={area_score:.3f}, rect={rect_score:.3f}, score={score:.3f}")

        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad, best_score if best_quad is not None else (None, None)

def nothing(_):
    pass

def main():
    parser = argparse.ArgumentParser(description="Detect and outline a table-like quadrilateral using contrast/edges.")
    parser.add_argument("--src", default="0", help="Camera source index (e.g. 0) or /dev/video2")
    parser.add_argument("--backend", default="auto", help="auto|v4l2|dshow|msmf|any")
    parser.add_argument("--width", type=int, default=1280, help="Requested width")
    parser.add_argument("--height", type=int, default=800, help="Requested height")
    parser.add_argument("--fps", type=int, default=100, help="Requested FPS")
    parser.add_argument("--save", default="table_corners.json", help="Output JSON filename")
    args = parser.parse_args()

    # Parse src: int if possible
    try:
        src = int(args.src)
    except Exception:
        src = args.src

    # Backend selection
    if args.backend == "auto":
        if sys.platform.startswith("win"):
            api = cv2.CAP_DSHOW
            backend_name = "CAP_DSHOW"
        else:
            api = cv2.CAP_V4L2
            backend_name = "CAP_V4L2"
    else:
        m = {
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "any": cv2.CAP_ANY,
        }
        api = m.get(args.backend.lower(), cv2.CAP_ANY)
        backend_name = args.backend

    cap = cv2.VideoCapture(src, api)
    if not cap.isOpened():
        print(f"❌ Cannot open camera src={args.src} backend={backend_name}")
        sys.exit(1)

    # Request mode (may be ignored by driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # On Linux, MJPG often needed for high FPS at higher res (optional; you can toggle)
    if not sys.platform.startswith("win"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # UI windows
    win = "find_table"
    dbg = "debug_edges"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(dbg, cv2.WINDOW_NORMAL)

    # Trackbars for tuning
    cv2.createTrackbar("Canny low", dbg, 40, 255, nothing)
    cv2.createTrackbar("Canny high", dbg, 120, 255, nothing)
    cv2.createTrackbar("Blur k", dbg, 5, 31, nothing)          # odd
    cv2.createTrackbar("Close iters", dbg, 2, 10, nothing)
    cv2.createTrackbar("Dilate iters", dbg, 1, 10, nothing)
    cv2.createTrackbar("Min area %", dbg, 5, 50, nothing)      # percent of image
    cv2.createTrackbar("Max area %", dbg, 98, 100, nothing)

    # CLAHE (contrast stabilization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    last_quad = None
    last_score = None
    last_save_time = 0.0

    print("=" * 80)
    print("find_table.py")
    print("Controls:")
    print("  s : save detected corners to JSON")
    print("  r : reset (forget last good quad)")
    print("  q / ESC : quit")
    print("=" * 80)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Grayscale + CLAHE to reduce lighting sensitivity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # Read trackbars
        low = cv2.getTrackbarPos("Canny low", dbg)
        high = cv2.getTrackbarPos("Canny high", dbg)
        blur_k = cv2.getTrackbarPos("Blur k", dbg)
        close_iters = cv2.getTrackbarPos("Close iters", dbg)
        dilate_iters = cv2.getTrackbarPos("Dilate iters", dbg)
        min_area_pct = cv2.getTrackbarPos("Min area %", dbg)
        max_area_pct = cv2.getTrackbarPos("Max area %", dbg)

        # Enforce valid parameters
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            blur_k += 1
        high = max(high, low + 1)

        # Pipeline
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, low, high)

        # Connect edges: close then dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if close_iters > 0:
            edges2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        else:
            edges2 = edges.copy()

        if dilate_iters > 0:
            edges2 = cv2.dilate(edges2, kernel, iterations=dilate_iters)

        # Find quad
        h, w = edges2.shape[:2]
        min_area_frac = float(min_area_pct) / 100.0
        max_area_frac = float(max_area_pct) / 100.0

        quad, score = find_best_quad(edges2, min_area_frac=min_area_frac, max_area_frac=max_area_frac)

        # Display
        disp = frame.copy()

        if quad is not None:
            last_quad = quad
            last_score = score

        # If we have a last quad, draw it (even if current frame failed)
        if last_quad is not None:
            q = last_quad.astype(np.int32)
            cv2.polylines(disp, [q], True, (0, 255, 0), 3)

            for i, p in enumerate(q):
                cv2.circle(disp, tuple(p), 6, (0, 255, 255), -1)
                cv2.putText(disp, f"P{i+1}", (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            area = polygon_area(last_quad)
            cv2.putText(disp, f"Quad score: {last_score:.3f}  area(px^2): {area:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "No table quad found (tune edges/iters/min-area or improve border contrast)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show windows
        cv2.imshow(win, disp)
        cv2.imshow(dbg, edges2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            last_quad = None
            last_score = None
            print("Reset: cleared last quad.")
        elif key == ord("s"):
            now = time.time()
            if now - last_save_time < 0.5:
                continue  # debounce
            last_save_time = now

            if last_quad is None:
                print("❌ Cannot save: no quad detected.")
                continue

            out = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": str(args.src),
                "backend": backend_name,
                "requested": {"width": args.width, "height": args.height, "fps": args.fps},
                "corners_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                "corners_px": last_quad.tolist(),
                "score": float(last_score) if last_score is not None else None,
            }

            Path(args.save).write_text(json.dumps(out, indent=2))
            print(f"✅ Saved corners to {args.save}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
