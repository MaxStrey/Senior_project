"""
calibrate_rail.py

Interactive calibration for the paddle rail. Produces calibration.json
with the steps-per-meter conversion and the usable soft limits.

Run:
    python calibrate_rail.py --port /dev/ttyACM0

Procedure:
    1. Jog the paddle to your desired mechanical center.
    2. Type 'z' to zero the step counter there.
    3. Jog to the right a fixed number of steps.
    4. Measure the physical distance with a ruler.
    5. The script computes steps_per_meter and saves it.
    6. Enter the usable left/right limits (mm from center).
"""

import argparse
import json
from pathlib import Path

from paddle_client import PaddleClient


def print_help():
    print()
    print("Commands:")
    print("   j <steps>   jog by <steps> (negative = left)")
    print("   g <steps>   goto absolute <steps>")
    print("   z           zero current position")
    print("   ?           query current position from Arduino")
    print("   h           print this help")
    print("   done        finish jogging, enter calibration measurement")
    print("   quit        exit without saving")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True,
                        help="Serial port of the Arduino (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", default="calibration.json",
                        help="Output path for calibration JSON")
    args = parser.parse_args()

    print(f"Connecting to {args.port} @ {args.baud}...")
    pc = PaddleClient(args.port, args.baud)
    print("Connected.")

    print("=" * 60)
    print("  PADDLE RAIL CALIBRATION")
    print("=" * 60)
    print_help()

    current_pos = 0

    while True:
        try:
            cmd = input("cal> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            pc.close()
            return

        if not cmd:
            continue
        if cmd == "quit":
            pc.close()
            return
        if cmd == "h":
            print_help()
            continue
        if cmd == "?":
            st = pc.query()
            if st is None:
                print("  (no reply from Arduino)")
            else:
                print(f"  pos={st.position_steps}  "
                      f"tgt={st.target_steps}  busy={st.busy}")
                current_pos = st.position_steps
            continue
        if cmd == "z":
            pc.zero_here()
            current_pos = 0
            print("  zeroed.")
            continue
        if cmd == "done":
            break

        parts = cmd.split()
        if len(parts) != 2:
            print("  ? (type 'h' for help)")
            continue
        try:
            val = int(parts[1])
        except ValueError:
            print("  numeric arg required")
            continue

        if parts[0] == "j":
            # Sync from Arduino first in case an earlier move was interrupted
            st = pc.query()
            if st is not None:
                current_pos = st.position_steps
            current_pos += val
            pc.goto_steps(current_pos, dedupe=False)
            print(f"  jogging to {current_pos}")
        elif parts[0] == "g":
            current_pos = val
            pc.goto_steps(val, dedupe=False)
            print(f"  going to {val}")
        else:
            print("  unknown command")

    # ----------------------------------------------------------------
    # Measurement phase
    # ----------------------------------------------------------------
    print()
    print("Now we'll compute steps_per_meter.")
    print("Make sure the paddle is at the reference-zero position you")
    print("want (usually the middle of your usable range).")
    print()
    input("Press ENTER once paddle is at reference zero... ")
    pc.zero_here()
    print("  Zeroed.")
    print()

    try:
        target_steps = int(input("Step count to move RIGHT (e.g. 2000): "))
    except ValueError:
        print("Bad input; exiting.")
        pc.close()
        return

    pc.goto_steps(target_steps, dedupe=False)
    input(f"Moving {target_steps} steps. Press ENTER once motion stops... ")

    try:
        dist_mm = float(input("Measured distance with ruler (mm): "))
    except ValueError:
        print("Bad input; exiting.")
        pc.close()
        return

    if dist_mm <= 0:
        print("Distance must be positive; exiting.")
        pc.close()
        return

    steps_per_meter = target_steps / (dist_mm / 1000.0)
    print()
    print(f"  steps_per_meter = {steps_per_meter:.1f}")
    print(f"                  = {steps_per_meter/1000:.3f} steps/mm")
    print()

    # Soft limits
    try:
        xmin_mm = float(input("Leftmost usable X (mm from center, negative): "))
        xmax_mm = float(input("Rightmost usable X (mm from center, positive): "))
    except ValueError:
        print("Bad limits; defaulting to +/- 150 mm.")
        xmin_mm, xmax_mm = -150.0, 150.0

    # Return to zero for safety
    pc.goto_steps(0, dedupe=False)
    print("  Returning to zero...")

    calib = {
        "steps_per_meter": steps_per_meter,
        "x_min_meters": xmin_mm / 1000.0,
        "x_max_meters": xmax_mm / 1000.0,
    }

    Path(args.out).write_text(json.dumps(calib, indent=2))
    print()
    print(f"Saved calibration to {args.out}")
    print(json.dumps(calib, indent=2))

    pc.close()


if __name__ == "__main__":
    main()
