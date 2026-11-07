^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package mono_depth_viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2026-11-07)
------------------
* Live mode (default): subscribe to an Image or CompressedImage topic (bag playback or a real camera) and publish a coloured sensor_msgs/PointCloud2 on /mono_depth/points. No window is opened.
* Breaking: the Qt GUI now needs the --debug flag (``viewer --debug <bag>``); a bag path without --debug is an error.
* Point cloud options: ROS (x forward, y left, z up) or optical axes, --auto-level ground fitting, manual --pitch/--cam-height, --hfov, --near/--far, --pc-stride, --pc-frame, --pc-stamp.
* Launch file with arguments for all options; python:= / MONO_DEPTH_PYTHON selects a virtualenv interpreter.
* Faster point cloud creation (about 127 ms to 7 ms per cloud at stride 4).
* Clean shutdown on Ctrl+C.
* Add PERFORMANCE.md (measured rates, tested hardware, 30 FPS requirements) and benchmarks/ scripts.
* Contributors: Divagar

0.1.1 (2026-10-23)
------------------
* Initial release.
* Qt viewer that reads sensor_msgs/Image and CompressedImage topics from ROS 2 sqlite3 (.db3) bags.
* Monocular depth estimation with transformer models (Depth Anything V1 Small/Base/Large, DPT-Hybrid MiDaS).
* Playback, frame scrubbing, frame skipping, colormap and input size selection.
* Save frames (PNG + raw .npy) and export side-by-side MP4 videos.
* Contributors: Divagar
