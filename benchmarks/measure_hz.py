#!/usr/bin/env python3
"""Measure the real publish rate and bandwidth of a topic while the node runs.

    python3 benchmarks/measure_hz.py /mono_depth/points --seconds 40
"""
import argparse
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("topic")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--skip", type=float, default=0, help="ignore the first N seconds (start-up)")
    a = p.parse_args()
    rclpy.init()
    node = Node("measure_hz")
    from rosidl_runtime_py.utilities import get_message
    stamps, sizes = [], []

    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < 30:               # wait for the topic and learn its type
        types = dict(node.get_topic_names_and_types()).get(a.topic)
        if types:
            break
        rclpy.spin_once(node, timeout_sec=0.3)
    else:
        raise SystemExit(f"topic {a.topic} not found")
    msg_cls = get_message(types[0])
    start = time.time()

    def cb(msg):
        if time.time() - start >= a.skip:
            stamps.append(time.time())
            sizes.append(len(msg.data))

    node.create_subscription(msg_cls, a.topic, cb, qos_profile_sensor_data)
    end = start + a.skip + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    gaps = [b - a_ for a_, b in zip(stamps, stamps[1:])]
    if len(stamps) < 2:
        print(f"{a.topic}: {len(stamps)} messages in {a.seconds}s")
    else:
        hz = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        print(f"{a.topic}: {hz:.2f} Hz over {len(stamps)} msgs | gap median {statistics.median(gaps)*1000:.0f} ms, "
              f"worst {max(gaps)*1000:.0f} ms | {statistics.mean(sizes)/1e6:.2f} MB/msg, {hz*statistics.mean(sizes)/1e6:.1f} MB/s")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
