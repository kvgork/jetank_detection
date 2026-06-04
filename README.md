# jetank_detection

Single-class sock detector for the JeTank robot, using YOLO11n with a ROS 2
lifecycle action server.

## Purpose

Detects socks lying on the floor via the left stereo camera
(`/stereo_camera/left/image_raw`).  The detector is single-class (`sock`) and
is designed to be triggered on-demand by a task planner (the `DetectSocks`
action), with an optional `continuous:=true` mode for live detection during
exploration.

The depth/3D component (stereo query at the bbox centroid) is sketched in
`plans/sock-detection-plan.md §3` but not yet implemented — depth is gated on
P1 training and the grasp task.

---

## 3-Stage TensorRT Integration Plan (see plan §5)

| Stage | Backend | Speed | Status |
|-------|---------|-------|--------|
| 1 | PyTorch via `ultralytics` (pip in pixi env) | ~30–40 ms/frame | **Current** |
| 2 | TensorRT `.engine` via `--system-site-packages` | ~16–30 ms/frame | Pending P4 |
| 3 | Subprocess fallback (system Python + JetPack TRT) | ~16–30 ms/frame | Fallback if Stage 2 fails |

Stage 1 is sufficient for ≥10 Hz on a discrete pick task.

---

## Installing the Backend (Stage 1)

```bash
# Enter the pixi environment
cd /path/to/ros2_ws
pixi shell

# Install ultralytics (PyTorch + YOLO11 support)
pip install ultralytics

# Verify
python3 -c "from ultralytics import YOLO; print('OK')"
```

---

## Running

### Sim vs real models

Detection in Gazebo and on the real robot are different visual domains, so the
pipeline uses **two models** and picks one at runtime from the explicit `sim`
flag (not `use_sim_time`):

- `sim:=true`  → loads `model_path_sim` (trained on Gazebo imagery)
- `sim:=false` → loads `model_path_real` (trained on real camera frames)
- `model_path:=...` overrides the selection for one-off testing.

Two launch entry points wrap the base `detect.launch.py` and pin the flag:

| Entry point | `sim` | Default mode | Use |
|-------------|-------|--------------|-----|
| `detect_sim.launch.py`  | `true`  | continuous (live) | Gazebo / `sim_demo` |
| `detect_real.launch.py` | `false` | on-demand (action) | physical robot |

### 1. Build

```bash
cd /path/to/ros2_ws
pixi run build
# or
pixi run -- colcon build --symlink-install --packages-select jetank_detection
```

### 2. Real robot — on-demand (default)

```bash
# Terminal 1 — launch the real entry point (sim:=false)
source install/setup.bash
ros2 launch jetank_detection detect_real.launch.py model_path_real:=/path/to/sock_real.pt

# Terminal 2 — configure and activate
ros2 lifecycle set /sock_detector configure
ros2 lifecycle set /sock_detector activate

# Terminal 3 — send a detection goal
ros2 action send_goal /detect_socks jetank_detection/action/DetectSocks \
    '{timeout: 5.0, min_confidence: 0.5, n_frames: 10}'
```

### 3. Sim — continuous (live publisher)

```bash
ros2 launch jetank_detection detect_sim.launch.py model_path_sim:=/path/to/sock_sim.pt
ros2 lifecycle set /sock_detector configure
ros2 lifecycle set /sock_detector activate
ros2 topic echo /detections/socks
```

> The base `detect.launch.py` is still available if you want to set `sim` and
> the model paths by hand (e.g. `detect.launch.py sim:=true model_path:=...`).

### 4. Simulation demo (via jetank_ros_main)

```bash
# Sock arena sim + live detection
pixi run detect
# or equivalently:
ros2 launch jetank_ros_main sim_demo.launch.py world:=sock_arena detect:=true slam:=false
```

---

## Capturing a training dataset

The **real** track is captured with the web UI `/capture` button
(`jetank_web_control`, saving to `~/datasets/detection`). The **sim** track uses
the headless `capture_frames` node — start a Gazebo `sock_arena` session, then
record frames at a fixed interval (no human clicking):

```bash
# Terminal 1 — sim with the sock arena
ros2 launch jetank_ros_main sim_demo.launch.py world:=sock_arena slam:=false

# Terminal 2 — capture 600 frames, one every 0.5 s, into the sim dataset dir
ros2 run jetank_detection capture_frames --ros-args \
    -p output_dir:=$HOME/datasets/detection/sim \
    -p domain:=sim -p interval_sec:=0.5 -p max_frames:=600
```

Frames are written as `sock_<domain>_NNNNNN.jpg`; numbering resumes after any
existing frames so multiple runs (different floor textures / lighting) accumulate
into one dataset. `max_frames:=0` captures until Ctrl-C.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_image_topic` | `/stereo_camera/left/image_raw` | image topic to capture |
| `output_dir` | `~/datasets/detection/sim` | where JPEGs are written |
| `domain` | `sim` | filename tag (`sim` / `real`) |
| `interval_sec` | `1.0` | seconds between saved frames |
| `max_frames` | `0` | stop after N frames (0 = unlimited) |
| `jpeg_quality` | `95` | JPEG quality 1–100 |

The full sim data→model loop (domain randomization, auto-labelling, training) is
specified in `jetank_ros_main/plans/sock-sim-autotrain-plan.md`.

---

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections/socks` | `vision_msgs/Detection2DArray` | Detected socks (published when active) |
| `/detections/socks/debug` | `sensor_msgs/Image` | Annotated debug image (when `debug:=true`) |
| `/stereo_camera/left/image_raw` | `sensor_msgs/Image` | Input (configurable via `input_image_topic`) |

## Action

| Action | Type | Description |
|--------|------|-------------|
| `/detect_socks` | `jetank_detection/DetectSocks` | On-demand: collect N frames, return best detection |

### DetectSocks interface

```
# Goal
float32 timeout         # max seconds to wait for frames
float32 min_confidence  # minimum score to accept
int32 n_frames          # number of frames to process
---
# Result
vision_msgs/Detection2DArray best  # best-frame detections
float32 confidence                 # best detection score
bool found                         # true if any sock found
---
# Feedback
int32 frames_processed
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | bool | `false` | Environment selector: `true` loads `model_path_sim`, `false` loads `model_path_real` |
| `model_path` | string | `""` | Explicit `.pt`/`.engine` path; **overrides** the sim/real selection when set |
| `model_path_sim` | string | `""` | Model loaded when `sim:=true` (trained on Gazebo imagery) |
| `model_path_real` | string | `""` | Model loaded when `sim:=false` (trained on real camera frames) |
| `input_image_topic` | string | `/stereo_camera/left/image_raw` | Input camera topic |
| `confidence` | float | `0.5` | Detection confidence threshold |
| `n_frames` | int | `10` | Frames per action goal |
| `continuous` | bool | `false` | Live detection mode |
| `debug` | bool | `true` | Publish debug image |
| `detections_topic` | string | `/detections/socks` | Output detections topic |
| `debug_image_topic` | string | `/detections/socks/debug` | Debug image topic |

---

## NOT YET DONE / Hardware-Gated

The following phases require physical hardware or off-robot compute and are
**NOT** implemented in this package (see `plans/sock-detection-plan.md`):

| Phase | Blocker | Plan reference |
|-------|---------|----------------|
| **P0 — Dataset** | Requires physical robot + floor photos (400–600 images) | §4 |
| **P1 — Training** | Requires P0 dataset; training on x86 GPU / Colab T4 | §4 |
| **P4 — TensorRT engine** | Requires trained `.pt` + on-device `yolo export format=engine`; TRT bound to JetPack | §5 |
| **P5 — Thermal soak** | Requires real Jetson under full Nav2+SLAM+detection load (`jtop`/`tegrastats`) | §6 tests 10–12 |

A model file path (`model_path`) left empty at launch is **intentional** — the
node will start, log a warning, and await lifecycle configuration with a real
model once P1 is complete.

### Do-NOT-touch boundaries

Per `plans/sock-detection-plan.md §7`:
- `jetank_motor_control`, `jetank_navigation`, `jetank_moveit_config`, `jetank_web_control`
  nav code are **read-only** from this package.
- Camera topics in `jetank_perception` are consumed read-only.
- Detection is purely additive: new package + topics/action only.
