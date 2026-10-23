^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package mono_depth_viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.1 (2026-10-23)
------------------
* Initial release.
* Qt viewer that reads sensor_msgs/Image and CompressedImage topics from ROS 2 sqlite3 (.db3) bags.
* Monocular depth estimation with transformer models (Depth Anything V1 Small/Base/Large, DPT-Hybrid MiDaS).
* Playback, frame scrubbing, frame skipping, colormap and input size selection.
* Save frames (PNG + raw .npy) and export side-by-side MP4 videos.
* Contributors: Divagar
