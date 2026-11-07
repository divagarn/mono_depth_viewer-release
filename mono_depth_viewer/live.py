"""Headless live mode: subscribe to a camera topic, estimate depth, publish a point cloud."""
import signal
import threading
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data

from mono_depth_viewer.bag_reader import IMAGE_TYPES, image_to_rgb
from mono_depth_viewer.depth_model import MODEL_ALIASES, DepthEstimator
from mono_depth_viewer.pointcloud import PointCloudPublisher


def _wait_for_topic(node, topic):
    """Block until `topic` is advertised; returns its message class."""
    log, last = node.get_logger(), 0.0
    while rclpy.ok():
        for name, types in node.get_topic_names_and_types():
            if name == topic:
                for t in types:
                    if t in IMAGE_TYPES:
                        return IMAGE_TYPES[t]
        if time.time() - last > 5:
            log.info(f"Waiting for image topic {topic} ...")
            last = time.time()
        rclpy.spin_once(node, timeout_sec=0.5)
    return None


def run_live(args):
    pcp = PointCloudPublisher(
        args.pc_topic, args.pc_frame, args.hfov, args.near, args.far, args.pc_stride, args.pc_stamp or "image",
        axes=args.pc_axes, pitch=args.pitch, cam_height=args.cam_height, auto_level=args.auto_level)
    node = pcp.node
    log = node.get_logger()
    try:
        log.info(f"Loading model '{args.model}' (first run downloads weights) ...")
        est = DepthEstimator(MODEL_ALIASES[args.model])
        log.info(f"Model ready on {est.device}")
        msg_cls = _wait_for_topic(node, args.image_topic)
        if msg_cls is None:
            return

        cond, latest, stop = threading.Condition(), [None], threading.Event()

        def on_image(msg):                      # keep only the newest frame; inference may be slower than the camera
            with cond:
                latest[0] = msg
                cond.notify()

        node.create_subscription(msg_cls, args.image_topic, on_image, qos_profile_sensor_data)
        log.info(f"Subscribed to {args.image_topic}; publishing live point clouds (Ctrl+C to stop)")

        def work():
            count, t0 = 0, time.time()
            while not stop.is_set():
                with cond:
                    while latest[0] is None and not stop.is_set():
                        cond.wait(0.5)
                    msg, latest[0] = latest[0], None
                if msg is None:
                    continue
                try:
                    rgb = image_to_rgb(msg)
                    depth = est.predict(rgb, args.input_size)
                    stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                    pcp.publish(rgb, depth, stamp, msg.header.frame_id)
                except Exception as e:  # a bad frame should not kill the node
                    log.warning(f"Frame skipped: {type(e).__name__}: {e}")
                    continue
                count += 1
                if time.time() - t0 >= 5:
                    log.info(f"{count / (time.time() - t0):.2f} clouds/s")
                    count, t0 = 0, time.time()

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
        thread.join(timeout=10)            # let the current inference finish before tearing torch down
    finally:
        try:
            pcp.shutdown()
        except Exception:
            pass
