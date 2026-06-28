# thinkgrasp_server

GPU server workspace for ThinkGrasp — a vision-language grasp detection system (CoRL 2024). Uses GPT-4o for scene analysis, LangSAM for segmentation, and FGC-GraspNet for 6-DOF grasp pose estimation.

In the real-robot setup this machine acts as the **server**: it receives camera frames from a local PC over the network, runs inference, and returns a grasp pose.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/eric-mjk/thinkgrasp_server.git ~/Eric/thinkgrasp_server
cd ~/Eric/thinkgrasp_server
git submodule update --init --recursive
```

### 2. Start the container
```bash
docker run -it -d \
  --gpus all \
  --ipc host \
  --net host \
  --privileged \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  --name eric_thinkgrasp \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/thinkgrasp_server:/workspace \
  ericmjk/panda_ws:thinkgrasp
```

### 3. Set up environment (inside container)
```bash
source /workspace/thinkgrasp/run.sh   # sets CUDA_HOME, PATH, LD_LIBRARY_PATH
conda activate thinkgrasp
```

## Running

### Simulation (PyBullet)

Assets must be present in `thinkgrasp/ThinkGrasp/assets/unseen_objects_40/` — download from the [ThinkGrasp HuggingFace dataset](https://huggingface.co/datasets/thinkgrasp/thinkgrasp) (not Google Drive).

```bash
cd /workspace/thinkgrasp/ThinkGrasp
wandb login
export OPENAI_API_KEY="sk-..."
python simulation_main.py --gui
```

Use `simulation_main_with_viewer.py` for the Matplotlib-based interactive viewer, or the notebooks (`sim_main_w_gui.ipynb`, `simulation_main_interactive.ipynb`) for step-by-step runs.

### Real-robot Flask server (port 5000)

```bash
cd /workspace/thinkgrasp/ThinkGrasp
export OPENAI_API_KEY="sk-..."
export THINKGRASP_SHOW_MATPLOTLIB=1   # set to 0 to skip visualizations
export THINKGRASP_SHOW_OPEN3D=0
python realarm_upload_server.py
```

The server exposes:
- `POST /grasp_pose` — accepts multipart form upload with `image` (RGB), `depth`, and `text` (instruction) fields; returns `{"xyz": [...], "rot": [[...]], "dep": ...}`
- `GET /health`

Kill the server:
```bash
ps aux | grep realarm
kill -9 <PID>
```

> **Alternative entry points**: `realarm_server_safe.py` accepts JSON with file paths on server disk instead of uploads. `realarm.py` is the original version without display guards.

## Client-Server Architecture

```
Local PC                              Server PC (this machine)
  RealSense camera  ─── HTTP POST ──▶  ThinkGrasp Flask server :5000
  Panda robot (ROS2)  ◀─ grasp pose ─  (GPT-4o + LangSAM + FGC-GraspNet)
  pymoveit2 execution
```

The local PC captures RGB+depth from a RealSense camera, sends them to this server via `POST /grasp_pose`, then executes the returned grasp pose through `pymoveit2`.

See `ClientServerTests/` for minimal socket connectivity tests and `ClientServerTutorial.md` for a working full-stack example including camera streaming and robot execution.

## Docker Images

- [`ericmjk/panda_ws`](https://hub.docker.com/repository/docker/ericmjk/panda_ws/general) — tags: `vanilla`, `latest`, `thinkgrasp`
- [`ericmjk/panda_thinkgrasp`](https://hub.docker.com/repository/docker/ericmjk/panda_thinkgrasp/general)
