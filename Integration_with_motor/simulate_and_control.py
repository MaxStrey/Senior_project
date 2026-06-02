"""
simulate_and_control.py

The main demo. Spawns virtual balls in software, predicts where each
will cross the paddle line, commands the real Arduino-driven paddle to
that position, and animates the whole thing with matplotlib.

NO CAMERA REQUIRED.

Run:
    python simulate_and_control.py --port /dev/ttyACM0 --calib calibration.json

Dependencies:
    pip install pyserial numpy matplotlib
"""

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patches

from paddle_client import PaddleClient, RailCalibration, CalibratedPaddle


# ============================================================================
# Ball: minimal state for one virtual ball
# ============================================================================
@dataclass
class Ball:
    x0: float    # launch position (m)
    y0: float
    vx: float    # constant velocity (m/s)
    vy: float
    t0: float    # wall-clock time at launch


# ============================================================================
# BallSimulator
#
# Spawns a ball every `interval_s`, lets it coast in a straight line until
# it crosses the paddle line (y=0), then pauses and respawns.
#
# Two functions provide access to the ball's state:
#   get_true_position(now)   -> exact (x, y) for drawing
#   get_observation(now)     -> (x, y) with Gaussian noise; this is the
#                               "measurement" that the Director consumes.
#                               Simulates what a vision system would produce.
# ============================================================================
class BallSimulator:
    def __init__(
        self,
        table_width_m: float = 0.4,   # lateral extent of ball spawn points
        spawn_y_m: float = 1.5,       # starting distance from paddle line
        paddle_y_m: float = 0.0,      # paddle line y-coordinate
        obs_noise_m: float = 0.005,   # measurement noise stddev (5 mm)
        interval_s: float = 2.5,      # time between ball spawns
        min_flight_s: float = 0.8,    # fastest flight time
        max_flight_s: float = 1.6,    # slowest flight time
        rng_seed: int = 42,
    ):
        self.table_width_m = table_width_m
        self.spawn_y_m = spawn_y_m
        self.paddle_y_m = paddle_y_m
        self.obs_noise_m = obs_noise_m
        self.interval_s = interval_s
        self.min_flight_s = min_flight_s
        self.max_flight_s = max_flight_s

        self.ball: Optional[Ball] = None
        self.next_spawn_t = time.time() + 1.0  # small delay at startup
        self.rng = np.random.default_rng(rng_seed)

    def _spawn(self, now: float) -> None:
        # Random launch x inside the table's half-width
        x0 = self.rng.uniform(-self.table_width_m / 2,
                               self.table_width_m / 2)

        # Random target x on the paddle line, biased to within 70% of the
        # simulator's declared width so most balls are reachable.
        x_target = self.rng.uniform(-self.table_width_m / 2 * 0.7,
                                     self.table_width_m / 2 * 0.7)

        # Random flight time, then solve for constant velocities that get
        # from (x0, spawn_y) to (x_target, paddle_y) in that time.
        flight = self.rng.uniform(self.min_flight_s, self.max_flight_s)
        vx = (x_target - x0) / flight
        vy = -(self.spawn_y_m - self.paddle_y_m) / flight   # vy<0: toward paddle

        self.ball = Ball(x0=x0, y0=self.spawn_y_m, vx=vx, vy=vy, t0=now)

    def step(self, now: float) -> None:
        """Advance simulation to wall-clock time `now`. Spawns/retires balls."""
        if self.ball is None:
            if now >= self.next_spawn_t:
                self._spawn(now)
            return

        pos = self.get_true_position(now)
        if pos is not None and pos[1] <= self.paddle_y_m:
            # Ball crossed the paddle line; retire, schedule the next spawn.
            self.ball = None
            self.next_spawn_t = now + self.interval_s

    def get_true_position(self, now: float) -> Optional[Tuple[float, float]]:
        """Noise-free ground truth. For plotting only."""
        if self.ball is None:
            return None
        dt = now - self.ball.t0
        return (self.ball.x0 + self.ball.vx * dt,
                self.ball.y0 + self.ball.vy * dt)

    def get_observation(self, now: float) -> Optional[Tuple[float, float]]:
        """Noisy measurement. This is what the Director sees."""
        true = self.get_true_position(now)
        if true is None:
            return None
        nx = self.rng.normal(0.0, self.obs_noise_m)
        ny = self.rng.normal(0.0, self.obs_noise_m)
        return (true[0] + nx, true[1] + ny)


# ============================================================================
# Director
#
# Consumes (x, y, t) observations from whatever source (simulator today,
# real camera tomorrow). Maintains a short rolling window, fits a line
# x(t) and a line y(t), extrapolates to find when y crosses the paddle
# line, then commands the paddle to the predicted x at that moment.
#
# Math:
#     Given recent samples {(t_i, x_i, y_i)} fit:
#         x(t) ~ a_x + b_x (t - t_mean)
#         y(t) ~ a_y + b_y (t - t_mean)
#     "Now" state:
#         x_now = a_x + b_x (t_latest - t_mean)
#         y_now = a_y + b_y (t_latest - t_mean)
#     Time to cross paddle_y:
#         tau = (paddle_y - y_now) / b_y         (requires b_y < 0)
#     Predicted intercept:
#         x_pred = x_now + b_x * tau
#
# This is a constant-velocity estimator. It does NOT model gravity,
# drag, spin, or bounce. That's fine for v1 with a side-view camera or
# a top-down view where the ball's image-frame motion is approximately
# linear between bounces.
# ============================================================================
class Director:
    def __init__(
        self,
        paddle: CalibratedPaddle,
        paddle_y_m: float = 0.0,
        history_len: int = 6,
        min_fit_points: int = 3,
        min_flight_time_s: float = 0.02,
        max_flight_time_s: float = 3.0,
        # ──────────────────────────────────────────────────────────────
        # HIERARCHICAL CONTROL TUNING
        # ──────────────────────────────────────────────────────────────
        # Big moves get high speed; small moves get held off entirely
        # until the prediction is committed enough to act on. This
        # eliminates the "shivering paddle" problem where every noisy
        # 5 mm wobble in x_pred triggers a new motor command.
        big_move_threshold_m: float = 0.030,   # >= 30 mm: "big move"
        small_move_dead_band_m: float = 0.010, # < 10 mm: ignore entirely
        ema_alpha: float = 0.20,               # smoothing on x_pred
        # Per-move speed envelope (steps/s). Big moves get fast_speed,
        # small moves get slow_speed, intermediate is linearly
        # interpolated.
        slow_speed_sps: int = 1500,
        fast_speed_sps: int = 6000,
        # Acceleration scales with the move magnitude similarly.
        slow_accel_sps2: int = 12000,
        fast_accel_sps2: int = 50000,
    ):
        self.paddle = paddle
        self.paddle_y_m = paddle_y_m
        self.history_len = history_len
        self.min_fit_points = min_fit_points
        self.min_flight_time_s = min_flight_time_s
        self.max_flight_time_s = max_flight_time_s
        self.big_move_threshold_m = big_move_threshold_m
        self.small_move_dead_band_m = small_move_dead_band_m
        self.ema_alpha = ema_alpha
        self.slow_speed_sps = slow_speed_sps
        self.fast_speed_sps = fast_speed_sps
        self.slow_accel_sps2 = slow_accel_sps2
        self.fast_accel_sps2 = fast_accel_sps2

        # Smoothed prediction (exponential moving average). Each new
        # raw prediction is blended with the previous smoothed value:
        #     x_smooth_new = alpha * x_raw + (1 - alpha) * x_smooth_old
        # alpha=1.0 disables smoothing. alpha~0.3 cuts noise by ~3x at
        # the cost of ~3 frames of lag. This is the right tradeoff: at
        # 60 fps that's 50 ms of lag, which is small compared to flight
        # time but big compared to noise frequency.
        self.x_pred_smooth: Optional[float] = None

        self.ts = deque(maxlen=history_len)
        self.xs = deque(maxlen=history_len)
        self.ys = deque(maxlen=history_len)

        self.last_prediction_x: Optional[float] = None
        self.last_commanded_x: Optional[float] = None

    def clear(self) -> None:
        """Called when the ball disappears (between trajectories).
        Prevents an old trajectory's samples from contaminating the next
        trajectory's line fit, AND clears the smoothing filter so the
        next trajectory starts fresh."""
        self.ts.clear()
        self.xs.clear()
        self.ys.clear()
        self.last_prediction_x = None
        self.last_commanded_x = None
        self.x_pred_smooth = None

    def observe(self, x_m: float, y_m: float, t: float) -> None:
        self.ts.append(t)
        self.xs.append(x_m)
        self.ys.append(y_m)

        if len(self.ts) < self.min_fit_points:
            return

        # Numpy-friendly arrays
        t_arr = np.fromiter(self.ts, dtype=float)
        x_arr = np.fromiter(self.xs, dtype=float)
        y_arr = np.fromiter(self.ys, dtype=float)

        # Mean-center time for numerical stability (t values can be
        # billions of seconds since the epoch).
        t_mean = t_arr.mean()
        t_c = t_arr - t_mean

        # Variance of t. If it's essentially zero, we have no temporal
        # spread in the samples and can't estimate velocity.
        tvar = float(np.dot(t_c, t_c) / t_c.size)
        if tvar < 1e-10:
            return

        # Slopes from the covariance / variance identity for 1-D regression.
        # Equivalent to np.polyfit(t_arr, x_arr, 1)[0] but faster and clearer.
        b_x = float(np.dot(t_c, x_arr) / (t_c.size * tvar))
        b_y = float(np.dot(t_c, y_arr) / (t_c.size * tvar))

        # Intercepts evaluated at t_mean (these equal the mean of x and y).
        a_x = float(x_arr.mean())
        a_y = float(y_arr.mean())

        # Project forward to "now" (the most recent sample).
        t_latest = t_arr[-1]
        x_now = a_x + b_x * (t_latest - t_mean)
        y_now = a_y + b_y * (t_latest - t_mean)

        # We only act if the ball is actually moving toward the paddle
        # line. b_y should be strongly negative. A tiny negative b_y
        # means near-stationary-in-y and the division below would blow up.
        if b_y >= -0.05:   # slower than 5 cm/s toward paddle: ignore
            return

        # Time until the linear extrapolation crosses the paddle line.
        tau = (self.paddle_y_m - y_now) / b_y
        if tau <= self.min_flight_time_s or tau > self.max_flight_time_s:
            return

        # The raw prediction.
        x_pred_raw = x_now + b_x * tau

        # ── EMA smoothing on the prediction ──────────────────────────
        # Without this, x_pred_raw wobbles by ~tau * sigma_v meters per
        # frame purely due to measurement noise. With alpha=0.35 the
        # noise gets cut by ~sqrt(alpha/(2-alpha)) ≈ 0.45, i.e. more
        # than half. Lag introduced is ~(1-alpha)/alpha frames,
        # roughly 2 frames at this alpha, which is fine.
        if self.x_pred_smooth is None:
            self.x_pred_smooth = x_pred_raw
        else:
            self.x_pred_smooth = (
                self.ema_alpha * x_pred_raw
                + (1.0 - self.ema_alpha) * self.x_pred_smooth
            )

        x_pred = self.x_pred_smooth
        self.last_prediction_x = x_pred

        # ── Hierarchical decision: how big is this requested move? ──
        # delta = how far the paddle would have to travel from the
        # current commanded target to the new predicted intercept.
        # Three regimes:
        #
        #   delta < dead_band         : do NOTHING. The wobble is
        #                               noise; chasing it causes the
        #                               jitter you saw.
        #
        #   delta >= big_threshold    : full speed and acceleration.
        #                               This is a real, large
        #                               correction (e.g. ball aimed
        #                               at the other side of the rail
        #                               than where we're sitting).
        #
        #   between                   : linearly interpolate speed
        #                               and accel. Smooth transition
        #                               between gentle and aggressive.
        # ───────────────────────────────────────────────────────────
        if self.last_commanded_x is None:
            delta = float("inf")
        else:
            delta = abs(x_pred - self.last_commanded_x)

        if delta < self.small_move_dead_band_m:
            # Inside dead band: don't even bother. Crucially, we do NOT
            # update last_commanded_x here, so the dead band is measured
            # against the last actually-issued command, not the last
            # noisy prediction. This prevents drift through the
            # dead band by tiny accumulating updates.
            return

        if delta >= self.big_move_threshold_m:
            speed = self.fast_speed_sps
            accel = self.fast_accel_sps2
        else:
            # Linear interpolation between (dead_band, slow) and
            # (big_threshold, fast).
            t = ((delta - self.small_move_dead_band_m)
                 / (self.big_move_threshold_m - self.small_move_dead_band_m))
            speed = int(self.slow_speed_sps
                        + t * (self.fast_speed_sps - self.slow_speed_sps))
            accel = int(self.slow_accel_sps2
                        + t * (self.fast_accel_sps2 - self.slow_accel_sps2))

        # Push speed/accel to the Arduino BEFORE the goto, so the new
        # move uses the new envelope from its very first step.
        self.paddle.client.set_max_speed(speed)
        self.paddle.client.set_acceleration(accel)
        self.paddle.goto_meters(x_pred)
        self.last_commanded_x = x_pred


# ============================================================================
# Animation glue
# ============================================================================
def run_demo(port: str, calib_path: str, duration_s: float, max_speed: int,
             accel: int):
    # ---- Calibration ----
    with open(calib_path) as f:
        cfg = json.load(f)
    calib = RailCalibration(
        steps_per_meter=cfg["steps_per_meter"],
        x_min_meters=cfg["x_min_meters"],
        x_max_meters=cfg["x_max_meters"],
    )

    # ---- Arduino ----
    client = PaddleClient(port)
    client.set_max_speed(max_speed)
    client.set_acceleration(accel)
    paddle = CalibratedPaddle(client, calib)

    # ---- Simulator ----
    # Make the simulator's lateral range match what the paddle can reach,
    # so most balls are physically interceptable.
    reach = min(abs(calib.x_min_meters), abs(calib.x_max_meters))
    sim = BallSimulator(
        table_width_m=2 * reach * 0.9,   # 90% of paddle reach
        spawn_y_m=1.2,
        paddle_y_m=0.0,
        obs_noise_m=0.005,
        interval_s=2.5,
    )

    director = Director(paddle=paddle, paddle_y_m=0.0)

    # ---- Matplotlib ----
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(calib.x_min_meters - 0.05, calib.x_max_meters + 0.05)
    ax.set_ylim(-0.1, sim.spawn_y_m + 0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  (distance from paddle)")
    ax.set_title("Simulated ball + real paddle")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Fixed decorations
    ax.axhline(0.0, color="k", lw=1.5, label="paddle line")
    ax.axvline(calib.x_min_meters, color="gray", ls="--", alpha=0.5,
               label="soft limit")
    ax.axvline(calib.x_max_meters, color="gray", ls="--", alpha=0.5)

    # Updated each frame
    ball_dot, = ax.plot([], [], "o", color="red", ms=12, label="ball")
    trail_line, = ax.plot([], [], "-", color="red", alpha=0.4)
    prediction_line = ax.axvline(0.0, color="orange", ls=":", lw=2,
                                 alpha=0.0, label="predicted intercept")
    paddle_w = 0.08  # 8 cm paddle width in the plot
    paddle_rect = patches.Rectangle((-paddle_w/2, -0.02), paddle_w, 0.04,
                                     color="blue", alpha=0.75, label="paddle")
    ax.add_patch(paddle_rect)
    ax.legend(loc="upper right")

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        verticalalignment="top", family="monospace",
                        fontsize=9)

    trail = deque(maxlen=40)
    t_start = time.time()
    last_query_t = [0.0]         # list so closure can mutate
    paddle_x_m = [0.0]
    prev_ball_active = [False]

    def update(_frame):
        now = time.time()

        # 1) Advance simulator
        sim.step(now)

        # 2) Feed director. Clear its history when a ball just ended.
        ball_active = (sim.ball is not None)
        if prev_ball_active[0] and not ball_active:
            director.clear()
            trail.clear()
        prev_ball_active[0] = ball_active

        if ball_active:
            obs = sim.get_observation(now)
            if obs is not None:
                director.observe(obs[0], obs[1], now)
            true = sim.get_true_position(now)
            if true is not None:
                trail.append(true)

        # 3) Query paddle position at ~20 Hz (not every animation frame)
        if now - last_query_t[0] > 0.05:
            last_query_t[0] = now
            m = paddle.query_meters()
            if m is not None:
                paddle_x_m[0] = m

        # 4) Update artists
        true = sim.get_true_position(now)
        if true is not None:
            ball_dot.set_data([true[0]], [true[1]])
        else:
            ball_dot.set_data([], [])

        if len(trail) > 1:
            tr = np.array(trail)
            trail_line.set_data(tr[:, 0], tr[:, 1])
        else:
            trail_line.set_data([], [])

        if director.last_prediction_x is not None and ball_active:
            prediction_line.set_xdata(
                [director.last_prediction_x, director.last_prediction_x])
            prediction_line.set_alpha(0.7)
        else:
            prediction_line.set_alpha(0.0)

        paddle_rect.set_xy((paddle_x_m[0] - paddle_w/2, -0.02))

        # HUD
        pred_str = (f"{director.last_prediction_x*1000:+.0f}"
                    if director.last_prediction_x is not None else "   -- ")
        info_text.set_text(
            f"paddle    = {paddle_x_m[0]*1000:+7.1f} mm\n"
            f"predicted = {pred_str} mm\n"
            f"ball      = "
            f"{('active' if ball_active else 'waiting')}"
        )

        if now - t_start > duration_s:
            plt.close(fig)

        return (ball_dot, trail_line, prediction_line, paddle_rect, info_text)

    ani = anim.FuncAnimation(
        fig, update,
        interval=16,                # ~60 Hz animation; sim runs at this rate
        blit=False,
        cache_frame_data=False,
    )
    try:
        plt.show()
    finally:
        client.close()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--calib", default="calibration.json")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="Demo length in seconds")
    parser.add_argument("--max-speed", type=int, default=6000,
                        help="Motor max speed in steps/s")
    parser.add_argument("--accel", type=int, default=40000,
                        help="Motor acceleration in steps/s^2")
    args = parser.parse_args()
    run_demo(args.port, args.calib, args.duration, args.max_speed, args.accel)