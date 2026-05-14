# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Docker Environment

This workspace is designed to run inside a Docker container. The host directory `~/Eric/thinkgrasp_server` is mounted as `/workspace` inside the container.

**Start the container:**
```bash
docker run -it -d \
  --gpus all \
  --ipc host \
  --net host \
  --privileged \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  --name thinkgrasp \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/thinkgrasp_server:/workspace \
  ericmjk/panda_ws:thinkgrasp
```

Available images: `ericmjk/panda_ws:vanilla`, `ericmjk/panda_ws:latest`, `ericmjk/panda_ws:thinkgrasp`

## Workspace Layout

```
/workspace/   (= ~/Eric/thinkgrasp_server on host)
  thinkgrasp/
    ThinkGrasp/           ← Vision-language grasp detection system (forked submodule)
    run.sh                ← sets CUDA env vars; source before running ThinkGrasp
    simulation_main_with_viewer.py   ← simulation entry point with Matplotlib viewer
    sim_main_w_gui.ipynb
    simulation_main_interactive.ipynb
  ClientServerTests/      ← minimal socket test scripts (connectivity smoke tests)
    test_server.py        ← binds PORT 5050, accepts one connection, sends "hello"
    test_client.py        ← connects to a server IP:5050, prints received message
    ClientServerTutorial.md
```

Initialize the submodule with:
```bash
git submodule update --init --recursive
```

Submodule URLs:
- `thinkgrasp/ThinkGrasp` → `https://github.com/eric-mjk/_forked_ThinkGrasp.git`

## Build & Test Commands

All commands run **inside the container** from `/workspace/ros2_ws`.

```bash
# Build
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCHECK_TIDY=ON

# Build a single package
colcon build --packages-select <package_name>

# Run tests
colcon test
colcon test-result

# Run tests for a single package
colcon test --packages-select <package_name>
```

Before running anything, source the workspace: `source /workspace/ros2_ws/install/setup.bash`

> All paths below use `/workspace` (the in-container path).

## Launch Commands

```bash
# Fake hardware (no robot needed — for development/simulation)
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true

# Isaac Sim (start Isaac Sim first and press Play, then run this)
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=false

# Real robot
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=<ROBOT_IP> load_gripper:=true

# Hardware-only bringup (without MoveIt)
ros2 launch franka_bringup franka.launch.py robot_ip:=<ROBOT_IP> use_fake_hardware:=false
```

Controller management (after bringup):
```bash
# List available/active controllers
ros2 control list_controllers

# Spawn and activate a controller
ros2 control load_controller --set-state active joint_velocity_example_controller

# Switch active controller
ros2 control switch_controllers --activate joint_impedance_example_controller \
  --deactivate joint_velocity_example_controller
```

## Linting & Formatting

- **C++ formatting**: Chromium style, C++14, column limit 100 (`src/panda_ros2/.clang-format`)
- **C++ static analysis**: Comprehensive rules in `src/panda_ros2/.clang-tidy` (lower_case variables, CamelCase classes/structs)
- Linting is enforced via `ament_lint_auto` which runs: `ament_clang_format`, `ament_clang_tidy`, `ament_cppcheck`, `ament_copyright`, `ament_flake8`, `ament_pep257`, `ament_xmllint`

## Architecture Overview

### ROS 2 Control Stack (panda_ros2)

This is a **ROS 2 Humble hardware driver and control framework** for Franka Emika Panda robot arms. It bridges `libfranka` (Franka's C++ SDK) with the `ros2_control` ecosystem.

**Control flow:**
```
libfranka (robot hardware)
    ↓
FrankaHardwareInterface (franka_hardware) — plugin loaded by ros2_control
    ↓ read()
Controller Manager — runs the real-time control loop
    ↓ update()
Example Controllers (franka_example_controllers)
    ↓ write()
FrankaHardwareInterface → libfranka → robot
```

The main control node (`franka_control2`) creates a multi-threaded executor with `SCHED_FIFO` real-time scheduling (priority 50). The hardware interface is loaded as a `pluginlib` plugin.

**Package responsibilities** (all under `ros2_ws/src/panda_ros2/`):

| Package | Role |
|---|---|
| `franka_hardware` | `ros2_control` hardware interface plugin; wraps `libfranka`; exposes state/command interfaces; hosts parameter services (stiffness, load, frames, collision behavior) and error recovery |
| `franka_control2` | Main control node binary; sets up `ControllerManager`, real-time executor, period-based `read/update/write` loop |
| `franka_msgs` | All Franka-specific ROS 2 message, service (`SetJointStiffness`, `ErrorRecovery`, etc.), and action (`Grasp`, `Homing`, `Move`) definitions |
| `franka_example_controllers` | 8+ example `ros2_control` controllers covering joint velocity/position/impedance, Cartesian velocity, gravity compensation; dual-arm variants included |
| `franka_robot_state_broadcaster` | Publishes `FrankaState` topic with full robot state (joints, forces, torques, Cartesian info, collision data) |
| `franka_semantic_components` | Adapter layer translating `ros2_control` hardware interfaces into Franka-specific semantic types |
| `franka_gripper` | Action server for Franka Hand gripper (Grasp, Move, Homing actions) |
| `franka_description` | URDF/Xacro robot descriptions; single-arm (`panda_arm.urdf.xacro`) and dual-arm (`dual_panda_arm.urdf.xacro`); `ros2_control` xacro configs |
| `franka_bringup` | Launch files and controller YAML configs for single/dual-arm bringup; MoveIt2 integration |
| `franka_moveit_config` | MoveIt 2 motion planning configuration; supports `use_fake_hardware`, `use_isaac_sim`, and real-robot modes |

**Additional workspace packages** (under `ros2_ws/src/`):

| Package | Role |
|---|---|
| `topic_based_ros2_control` | `ros2_control` hardware interface that bridges to/from ROS topics; used with Isaac Sim (`use_isaac_sim:=true`) so MoveIt communicates with the simulator over joint state/command topics |

**Key design patterns:**
- **Hardware parameters at runtime**: `franka_hardware` hosts ROS 2 parameter services so controllers and users can change robot behavior (stiffness, collision thresholds, TCP frame) without restart.
- **Error recovery**: Service servers in `franka_hardware` expose error recovery without restarting the control node.
- **Dual-arm support**: `FrankaMultiHardwareInterface` and dual-arm example controllers handle synchronized multi-robot configurations.
- **Real-time constraints**: The control loop uses `SCHED_FIFO` scheduling. Avoid allocations or blocking calls in controller `update()` methods.

### pymoveit2

Python client library (`ros2_ws/src/pymoveit2/`) providing async MoveIt 2 interfaces. Key classes: `MoveIt2` (arm planning/execution), `MoveIt2Gripper`, `MoveIt2Servo`. Used to drive the Panda from Python nodes without writing C++ controllers.

### ThinkGrasp

Vision-language grasp detection system (`thinkgrasp/ThinkGrasp/`) — CoRL 2024. Uses LangSAM for segmentation and FGC-GraspNet for 6-DOF grasp pose estimation. Runs in PyBullet simulation or real-world via Flask API.

**CUDA environment** — source `thinkgrasp/run.sh` or set manually before running any ThinkGrasp script:
```bash
source /workspace/thinkgrasp/run.sh
conda activate thinkgrasp
```

**Required env vars for real-robot server**:
```bash
export OPENAI_API_KEY="sk-..."           # GPT-4o is called for scene/object analysis
export THINKGRASP_SHOW_MATPLOTLIB=1      # set to 0 to skip matplotlib visualizations
export THINKGRASP_SHOW_OPEN3D=0          # set to 1 to save Open3D point cloud renders
export THINKGRASP_CHECKPOINT_GRASP_PATH="logs/checkpoint_fgc.tar"
```

**Simulation** (runs in PyBullet, from `/workspace/thinkgrasp/ThinkGrasp/`):
```bash
wandb login
export OPENAI_API_KEY="sk-..."
python simulation_main.py --gui      # or use simulation_main_with_viewer.py for Matplotlib UI
```
Assets required in `ThinkGrasp/assets/`: `unseen_objects_40/` (download from HuggingFace, not Google Drive). Many URDFs were patched to replace missing `textured.obj` — see `thinkgrasp_edits.txt` for the patch log.

**Real-robot Flask server** (port 5000, from `/workspace/thinkgrasp/ThinkGrasp/`):

There are three server entry points:

| Script | Input method | Use case |
|---|---|---|
| `realarm.py` | JSON body with file paths on server disk | original, paths must exist on server |
| `realarm_server_safe.py` | same as above | adds `THINKGRASP_SHOW_*` env var guards |
| `realarm_upload_server.py` | multipart form upload (`image`, `depth`, `text` fields) | client sends files over HTTP; no shared filesystem needed |

All three expose `POST /grasp_pose` → returns `{"xyz": [...], "rot": [[...]], "dep": ...}`.
`realarm_upload_server.py` additionally exposes `GET /health`.

```bash
# Standard start (run from ThinkGrasp/)
python realarm_upload_server.py

# Kill if needed
ps aux | grep realarm
kill -9 <PID>
```

Ray is initialized with `num_gpus=2`; LangSAM actor is allocated 0.8 GPU. If only one GPU is available, edit `realarm_server_safe.py` to `ray.init(num_gpus=1)`.

### Client-Server Architecture (Local ↔ Server)

The real-robot workflow splits across two machines:

```
Local PC (client)                      Server PC (GPU host)
  Camera (RealSense D4xx)  ──────────────▶  ThinkGrasp Flask server (port 5000)
  Robot (Panda via ROS2)                     returns grasp pose (xyz, rot, dep)
  pymoveit2 execution  ◀── grasp pose ──────
```

**Raw socket protocol** (used in `ClientServerTests/` and working examples in `ClientServerTutorial.md`):
```
[4 bytes big-endian uint32: metadata JSON length]
[N bytes: UTF-8 JSON metadata]
[4 bytes: payload length]  (or 8 bytes for rgb+depth: rgb_len, depth_len)
[payload bytes]
```
The server responds with `[4 bytes: result length][result JSON bytes]`.

**Connectivity test**:
```bash
# On server
python3 /workspace/ClientServerTests/test_server.py

# On local PC
# Edit SERVER_IP in test_client.py, then:
python3 /workspace/ClientServerTests/test_client.py
```

**Working real-robot client** pattern (see `ClientServerTutorial.md`): RealSense frames are JPEG-encoded for bandwidth, depth is sent as raw `uint16` bytes, joint state / FK pose are included in JSON metadata. The local client uses `pymoveit2` to execute the returned grasp pose.

## Critical Dependencies — Do Not Corrupt

### libfranka 0.8.0 (source build)

- **Version**: 0.8.0 — **incompatible with the newer apt package** `ros-humble-libfranka` (0.20.4) which is also present on the system
- **Source**: `/opt/libfranka/` (do not delete)
- **Installed to**: `/usr/local/lib/libfranka.so.0.8`, headers at `/usr/local/include/franka`, cmake at `/usr/local/lib/cmake/Franka/`
- **How it wins over the apt version**: `CMAKE_PREFIX_PATH=/usr/local:...` and `LD_LIBRARY_PATH=/usr/local/lib:...` are set so CMake and the linker find the source-built 0.8.0 **before** the apt-installed 0.20.4 at `/opt/ros/humble`

**What would break it**:
- `sudo apt upgrade` or `apt install ros-humble-*` that modifies `ros-humble-libfranka`
- Reordering `CMAKE_PREFIX_PATH` so `/opt/ros/humble` comes before `/usr/local`
- Running `sudo make install` in any other libfranka build directory (overwrites `/usr/local`)
- Deleting `/opt/libfranka/`

**Verify**:
```bash
ldconfig -p | grep franka          # should show /usr/local/lib/libfranka.so.0.8
find /usr/local -name "FrankaConfig.cmake"
```

### librealsense 2.53.1 (source build)

- **Installed to**: `/usr/local/lib`, headers at `/usr/local/include/librealsense2`, cmake at `/usr/local/lib/cmake/realsense2`

**Verify**:
```bash
ldconfig -p | grep realsense
find /usr/local -name "*realsense*Config.cmake"
python3 -c "import pyrealsense2 as rs; print('OK')"
```

### MoveIt 2.5.9 (apt)

- **Install method**: `apt` — `ros-humble-moveit 2.5.9` at `/opt/ros/humble/`
- Do not run `sudo apt remove ros-humble-moveit*` or manually downgrade

**Verify**:
```bash
dpkg -l | grep ros-humble-moveit   # should show 2.5.9
```

### Required environment variables

Must be set before building or running (check `~/.bashrc`):
```bash
export CMAKE_PREFIX_PATH=/usr/local:/opt/openrobots/lib/cmake:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

For ThinkGrasp:
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Pinocchio (robotpkg)

- **Install location**: `/opt/openrobots`
- Verify: `ls /opt/openrobots/lib/cmake`
