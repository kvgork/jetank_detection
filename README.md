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

### 1. Build

```bash
cd /path/to/ros2_ws
pixi run build
# or
pixi run -- colcon build --symlink-install --packages-select jetank_detection
```

### 2. Lifecycle node (on-demand mode, default)

```bash
# Terminal 1 — launch the node
source install/setup.bash
ros2 launch jetank_detection detect.launch.py model_path:=/path/to/sock.pt

# Terminal 2 — configure and activate
ros2 lifecycle set /sock_detector configure
ros2 lifecycle set /sock_detector activate

# Terminal 3 — send a detection goal
ros2 action send_goal /detect_socks jetank_detection/action/DetectSocks \
    '{timeout: 5.0, min_confidence: 0.5, n_frames: 10}'
```

### 3. Continuous (live publisher) mode

```bash
ros2 launch jetank_detection detect.launch.py \
    model_path:=/path/to/sock.pt \
    continuous:=true
ros2 lifecycle set /sock_detector configure
ros2 lifecycle set /sock_detector activate
ros2 topic echo /detections/socks
```

### 4. Simulation demo (via jetank_ros_main)

```bash
# Sock arena sim + live detection
pixi run detect
# or equivalently:
ros2 launch jetank_ros_main sim_demo.launch.py world:=sock_arena detect:=true slam:=false
```

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
| `model_path` | string | `""` | Path to `.pt` or `.engine` model |
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
