# mono_depth_viewer

Qt viewer for **monocular depth estimation** on image topics recorded in ROS 2 bags
(sqlite3 `.db3`). Runs transformer models (Depth Anything V1, DPT-Hybrid MiDaS) from Hugging Face
and shows the camera image next to a colourised depth map.

## Install (ROS 2 Humble)

```bash
cd ~/ros2_ws/src
git clone <this-repo-url> mono_depth_viewer
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
pip install -r src/mono_depth_viewer/requirements.txt
colcon build --packages-select mono_depth_viewer
source install/setup.bash
```

## Use

```bash
ros2 run mono_depth_viewer viewer /path/to/bag_folder
# or
ros2 launch mono_depth_viewer viewer.launch.py bag:=/path/to/bag_folder
```

Pick the topic (default `/camera/image_raw`), model, input size and colormap in the toolbar.
You can play, scrub, save a frame (PNG + raw `.npy`) or export a side-by-side MP4.
The first run downloads model weights from Hugging Face.

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
