# mono_depth_viewer

Qt viewer for **monocular depth estimation** on image topics recorded in ROS 2 bags
(sqlite3 `.db3`). Runs transformer models (Depth Anything V1, DPT-Hybrid MiDaS) from Hugging Face
and shows the camera image next to a colourised depth map.

## Install (ROS 2 Humble)

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/divagarn/mono_depth_ros2.git mono_depth_viewer
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
pip install transformers==4.40.2           # plus torch/torchvision; see requirements.txt
colcon build --symlink-install --packages-select mono_depth_viewer
source install/setup.bash
```

If you keep torch/transformers in a virtualenv (created with `--system-site-packages`) instead of the system Python,
tell the launch file which interpreter to use, once per shell or per launch:

```bash
export MONO_DEPTH_PYTHON=~/venvs/depth/bin/python        # or:  ... viewer.launch.py python:=~/venvs/depth/bin/python
```

## Run with `ros2 launch`

```bash
# live, headless: subscribe to the camera topic, publish /mono_depth/points
ros2 launch mono_depth_viewer viewer.launch.py auto_level:=true hfov:=100
# without a camera, play a bag in another terminal:  ros2 bag play <bag> --topics /camera/image_raw --loop

# debug GUI with a bag file
ros2 launch mono_depth_viewer viewer.launch.py debug:=true bag:=/path/to/bag pointcloud:=true auto_level:=true play:=true
```

Launch arguments (all optional; `ros2 launch mono_depth_viewer viewer.launch.py --show-args` lists them):
`python`, `debug`, `bag`, `play`, `pointcloud`, `image_topic`, `model`, `input_size`, `pc_topic`, `pc_frame`, `pc_axes`,
`pc_stamp`, `pc_stride`, `hfov`, `near`, `far`, `pitch`, `cam_height`, `auto_level`.

## Use

```bash
ros2 launch mono_depth_viewer viewer.launch.py debug:=true bag:=/path/to/bag_folder
```

Pick the topic (default `/camera/image_raw`), model, input size and colormap in the toolbar.
You can play, scrub, save a frame (PNG + raw `.npy`) or export a side-by-side MP4.
The first run downloads model weights from Hugging Face.

## Point cloud output

The node has two modes:

* **Live (default, no window):** subscribes to a camera topic, runs the depth model and publishes a
  coloured `sensor_msgs/PointCloud2`. Works with a real camera or with `ros2 bag play`.
* **`--debug` (GUI):** opens the viewer to open a bag, play/scrub it and see the depth image. Add `--pointcloud`
  to publish clouds from the GUI as well.

```bash
# live, headless (e.g. with `ros2 bag play <bag> --topics /camera/image_raw --loop` in another terminal)
ros2 run mono_depth_viewer viewer --image-topic /camera/image_raw --auto-level --hfov 100

# debug GUI with a bag
ros2 run mono_depth_viewer viewer --debug /path/to/bag --pointcloud --auto-level --play --hfov 100

# same through launch
ros2 launch mono_depth_viewer viewer.launch.py image_topic:=/camera/image_raw auto_level:=true hfov:=100
ros2 launch mono_depth_viewer viewer.launch.py debug:=true bag:=/path/to/bag pointcloud:=true
```

| Option | Default | Meaning |
|---|---|---|
| `--debug` | off | show the GUI instead of running headless |
| `--image-topic` | `/camera/image_raw` | live input (`Image` or `CompressedImage`) |
| `--model` | `small` | `small`, `base`, `large` or `dpt` |
| `--input-size` | `364` | network input short side in pixels; smaller = faster |
| `--pointcloud` | off | (`--debug` only) also publish; always on in live mode |
| `--pc-topic` | `/mono_depth/points` | topic name |
| `--pc-axes` | `ros` | `ros`: x forward, y left, z up (REP-103). `optical`: x right, y down, z forward |
| `--pc-frame` | `camera_link` (`camera_optical_frame` for `optical`) | `frame_id` (`image` = use each image's own header frame) |
| `--auto-level` | off | fit the ground plane every frame, level it and put the ground at z = 0 |
| `--pitch` | `0` | manual alternative: camera tilt-down angle in degrees |
| `--cam-height` | `0` | lift the cloud by this many metres (use with `--pitch`) |
| `--pc-stamp` | `image` (live) / `now` (`--debug`) | header stamp: the image's own (`image`) or wall clock (`now`) |
| `--pc-stride` | `4` | use every Nth pixel (bigger = fewer points, faster) |
| `--hfov` | `90` | camera horizontal field of view in degrees (pinhole model) |
| `--near` / `--far` | `0.5` / `20` | metres assigned to the closest / farthest part of the scene |
| `--play` | off | (`--debug`) start playback as soon as the model is ready |

View it in RViz with *Fixed Frame* = `camera_link` (the default `--pc-frame`) and a PointCloud2 display on the topic.
With the default `ros` axes the ground is horizontal only if the camera is level; for a camera that looks
down, add `--auto-level` (or `--pitch <deg>`).

**The scale is an assumption.** The network predicts relative depth, so `--near`/`--far` set the
metric range; distances are plausible but not measured. Lens distortion is not corrected (pinhole model),
so wide-angle or fisheye cameras give curved geometry. Clouds are published at the rate frames are
processed (about 1 Hz on CPU); in live mode only the newest camera frame is processed, older ones are dropped.

## Performance

Measured speeds, the tested hardware and what 30 FPS needs are in [PERFORMANCE.md](PERFORMANCE.md). Short version:
on a laptop CPU the default settings give about 0.75 Hz and the fast settings about 2 Hz; 30 Hz needs a GPU.
`benchmarks/` has the scripts to measure your own hardware.

## Notes

- Output is **relative** inverse depth (larger = closer), not metric distance.
- On CPU the Small model runs at roughly 1 FPS at 364 px; a CUDA GPU is much faster.
- Only `.db3` bags are supported (no `.mcap`).
- Model licenses: Depth Anything V1 Small is Apache-2.0; Base/Large are CC-BY-NC-4.0
  (non-commercial); DPT-Hybrid is MIT. This package's code is Apache-2.0.

## Test

```bash
colcon test --packages-select mono_depth_viewer && colcon test-result --verbose
```
