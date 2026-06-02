# Senior Project: Ball Tracking & Paddle Control System

A computer vision-based system for tracking ping pong ball trajectories and controlling a motorized paddle via Arduino integration. This project combines advanced ball tracking algorithms with real-time motor control to predict and respond to ball motion.

## Project Overview

This repository contains the complete implementation of a senior capstone project that uses camera feeds to:
- Detect and track ping pong balls in real-time
- Analyze ball spin and trajectory
- Control a motorized paddle rail to intercept the ball
- Calibrate and synchronize hardware components

**Status:** Complete senior project with final report and comprehensive documentation

## Repository Structure

### 📁 `Integration_with_motor/`
Main production code combining camera vision with Arduino motor control.

- **`camera_and_control_v2.py`** - Latest primary implementation combining vision and motor control
  - Real-time video capture and processing
  - Ball tracking and prediction
  - Motor commands and feedback loop
  - Video recording capability with ffmpeg
  
- **`camera_and_control.py`** - Previous version (v1) for reference

- **`calibrate_rail.py`** - Interactive calibration utility for the paddle rail
  - Establishes steps-per-meter conversion
  - Sets soft limits for the paddle range
  - Generates `calibration.json` for the system
  - Usage: `python calibrate_rail.py --port /dev/ttyACM0`

- **`paddle_client.py`** - Arduino communication client
  - Serial protocol for paddle control
  - Position queries and movement commands
  - Zero point calibration

- **`simulate_and_control.py`** - Simulation environment for testing control logic without hardware

- **`Testing_rail.jpeg`** - Hardware setup documentation with testing rail image

### 📁 `Legacy_ball_trackers/`
Evolution of ball tracking algorithms through multiple iterations.

- **`ball_tracker_1.0.py`** - Initial basic tracking implementation
- **`ball_tracker_2.0.py`** - Improved tracking with better detection
- **`ball_tracker_3.0.py`** - Advanced tracking with spin analysis
- **`ball_tracker_3_2.py`** - Final refined version with optimizations
- **`find_table.py`** - Table detection utilities
- **`spin_analyser_1_0.py`** - Spin analysis algorithms

### 📁 `Special_recordings/`
Video recordings and test data for validation and analysis.

- **`Legacy_ball_trackers_recordings/`** - Test videos used with legacy trackers

### 📄 Key Files

- **`Final_report.pdf`** - Complete technical report of the project
- **`LICENSE`** - MIT License (open source)
- **`README.md`** - This file
