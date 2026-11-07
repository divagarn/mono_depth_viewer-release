#!/usr/bin/env python3
"""Entry point for `ros2 run mono_depth_viewer viewer`.

Default (live, headless):  subscribe to --image-topic and publish a point cloud; no window is opened.
With --debug:              open the Qt GUI (depth view, bag playback) instead.
"""
import argparse
import sys

from mono_depth_viewer.models import MODEL_ALIASES


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="viewer", description="Monocular (single camera) depth -> PointCloud2, live or from a bag in the GUI.")
    p.add_argument("bag", nargs="?", default="", help="rosbag2 folder or .db3 file (only with --debug)")
    p.add_argument("--debug", action="store_true", help="show the GUI (depth image, playback controls) instead of running headless")
    p.add_argument("--play", action="store_true", help="(--debug) start playing as soon as the model is ready")
    p.add_argument("--image-topic", default="/camera/image_raw", help="live mode input, Image or CompressedImage (default: %(default)s)")
    p.add_argument("--model", choices=MODEL_ALIASES, default="small", help="depth model (default: %(default)s)")
    p.add_argument("--input-size", type=int, default=364, help="short-side resolution fed to the network; smaller = faster (default: %(default)s)")
    g = p.add_argument_group("point cloud output (sensor_msgs/PointCloud2)")
    g.add_argument("--pointcloud", action="store_true", help="(--debug) also publish the cloud; always on in live mode")
    g.add_argument("--pc-topic", default="/mono_depth/points", help="topic name (default: %(default)s)")
    g.add_argument("--pc-frame", default=None,
                   help="frame_id, or 'image' for the image header's own (default: camera_link, or camera_optical_frame with --pc-axes optical)")
    g.add_argument("--pc-axes", choices=("ros", "optical"), default="ros",
                   help="ros: x fwd, y left, z up (default); optical: x right, y down, z fwd")
    g.add_argument("--auto-level", action="store_true",
                   help="fit the ground plane each frame, level it and put the ground at z=0 (ros axes)")
    g.add_argument("--pitch", type=float, default=0.0, help="camera tilt-down angle in degrees to level the cloud (ros axes)")
    g.add_argument("--cam-height", type=float, default=0.0, help="lift the cloud by this many metres (use with --pitch)")
    g.add_argument("--pc-stamp", choices=("now", "image", "bag"), default=None,
                   help="header stamp: wall clock, or the image's own stamp (default: image in live mode, now in --debug)")
    g.add_argument("--pc-stride", type=int, default=4, help="use every Nth pixel (default: %(default)s)")
    g.add_argument("--hfov", type=float, default=90.0, help="camera horizontal field of view, degrees (default: %(default)s)")
    g.add_argument("--near", type=float, default=0.5, help="metres for the closest scene point (default: %(default)s)")
    g.add_argument("--far", type=float, default=20.0, help="metres for the farthest scene point (default: %(default)s)")
    return p, *p.parse_known_args(argv)   # unknown args (Qt, --ros-args) are passed through


def main():
    parser, args, rest = parse_args(sys.argv[1:])
    if args.debug:
        args.pc_stamp = args.pc_stamp or "now"
        from mono_depth_viewer import viewer      # imports Qt only when the GUI is requested
        viewer.run(args, rest)
        return
    if args.bag:
        parser.error("a bag file can only be opened in the GUI: add --debug (or play it with `ros2 bag play` "
                     "and keep the live mode)")
    from mono_depth_viewer.live import run_live
    try:
        run_live(args)
    except KeyboardInterrupt:              # Ctrl+C while loading the model / waiting for the topic
        pass


if __name__ == "__main__":
    main()
