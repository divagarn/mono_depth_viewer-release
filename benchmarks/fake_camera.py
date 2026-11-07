#!/usr/bin/env python3
"""Publish frames from a bag at a fixed rate (default 30 Hz) so the node's maximum throughput can be measured
without `ros2 bag play` timing effects.  Frames are loaded into memory first and cycled.

    python3 benchmarks/fake_camera.py --bag /path/to/bag --rate 30
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from mono_depth_viewer.bag_reader import BagReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--topic", default="/camera/image_raw")
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--frames", type=int, default=20)
    a = p.parse_args()
    reader = BagReader(a.bag)
    reader.select_topic(a.topic)
    frames = [reader.frame(int(k)) for k in np.linspace(0, len(reader) - 1, a.frames).astype(int)]
    rclpy.init()
    node = Node("fake_camera")
    pub = node.create_publisher(Image, a.topic, 5)
    msgs = []
    for f in frames:
        m = Image()
        m.height, m.width, m.encoding, m.step = f.shape[0], f.shape[1], "rgb8", f.shape[1] * 3
        m.header.frame_id = "camera"
        m.data = __import__("array").array("B", f.tobytes())
        msgs.append(m)
    node.get_logger().info(f"Publishing {len(msgs)} cycled frames on {a.topic} at {a.rate} Hz")
    period, nxt, i = 1.0 / a.rate, time.perf_counter(), 0
    try:
        while rclpy.ok():
            m = msgs[i % len(msgs)]
            m.header.stamp = node.get_clock().now().to_msg()
            pub.publish(m)
            i += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
