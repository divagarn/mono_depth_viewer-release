"""Publish a sensor_msgs/PointCloud2 built from a monocular (relative) depth map.

A single-camera depth network gives *relative* inverse depth, so metric scale is an assumption:
each frame's inverse depth is normalised to [0, 1] (smoothed over frames) and mapped linearly in
inverse-depth between `far` (0) and `near` (1) metres.  Intrinsics come from a pinhole model with the
given horizontal field of view.

Output axes (`axes`):
  "ros"     x forward, y left, z up (REP-103) - the ground looks horizontal in RViz with Z-up.  Default.
  "optical" x right, y down, z forward (camera optical frame).
In "ros" mode the cloud can also be levelled for a camera that is tilted down: `pitch` (degrees, manual) or
`auto_level` (fit the ground plane in the lower image each frame, rotate it to horizontal and put the ground
at z = 0).  `cam_height` lifts the cloud by that many metres (use with `pitch`).
"""
import array

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
MIN_GROUND_POINTS = 30      # a plane needs only a few points; keeps auto-level working at large --pc-stride
FIELDS = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
          for n, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]


class PointCloudPublisher:
    def __init__(self, topic="/mono_depth/points", frame_id=None, hfov=90.0, near=0.5, far=20.0,
                 stride=4, stamp="now", smooth=0.8, axes="ros", pitch=0.0, cam_height=0.0, auto_level=False):
        if not 0 < near < far:
            raise ValueError("need 0 < near < far")
        if axes not in ("ros", "optical"):
            raise ValueError("axes must be 'ros' or 'optical'")
        if axes == "optical" and (pitch or cam_height or auto_level):
            raise ValueError("pitch / cam_height / auto_level need axes='ros'")
        if not 1 <= hfov < 180:
            raise ValueError("hfov must be in [1, 180) degrees")
        self.axes, self.pitch, self.cam_height, self.auto_level = axes, np.radians(pitch), cam_height, auto_level
        self.frame_id = frame_id or ("camera_link" if axes == "ros" else "camera_optical_frame")
        self.hfov, self.near, self.far = hfov, near, far
        self._rng = np.random.default_rng(0)
        self._gn = self._gd = None             # smoothed ground normal / plane offset (camera "ros" axes)
        self._warned = False
        self.stride, self.stamp_mode, self.smooth = max(1, int(stride)), stamp, smooth
        self._lo = self._hi = None
        self._rays = {}
        if not rclpy.ok():
            # no rclpy signal handlers: Ctrl+C / SIGTERM keep their normal meaning (the live node handles them itself)
            rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
        self.node = Node("mono_depth_viewer")
        self.pub = self.node.create_publisher(PointCloud2, topic, 5)
        self.node.get_logger().info(
            f"Publishing PointCloud2 on {self.pub.topic_name} frame={self.frame_id} axes={axes} "
            f"(hfov={hfov}°, near={near} m, far={far} m, stride={self.stride}"
            f"{', auto-level' if auto_level else f', pitch={pitch}°' if pitch else ''})")

    def _ray_grid(self, h, w):
        key = (h, w)
        if key not in self._rays:
            s = self.stride
            v, u = np.mgrid[s // 2:h:s, s // 2:w:s].astype(np.float32)
            f = (w / 2) / np.tan(np.radians(self.hfov) / 2)
            self._rays[key] = ((u - (w - 1) / 2) / f, (v - (h - 1) / 2) / f)
        return self._rays[key]

    def depth_metres(self, sample):
        """Relative inverse depth (larger = closer) -> (metres, normalised 0..1)."""
        lo, hi = np.percentile(sample, 1), np.percentile(sample, 99)
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            self._lo = self.smooth * self._lo + (1 - self.smooth) * lo
            self._hi = self.smooth * self._hi + (1 - self.smooth) * hi
        s = np.clip((sample - self._lo) / max(self._hi - self._lo, 1e-6), 0, 1)
        return 1.0 / (1.0 / self.far + s * (1.0 / self.near - 1.0 / self.far)), s

    def _fit_ground(self, P):
        """RANSAC plane on candidate ground points P (N,3, ros axes). Returns (up_normal, d) with n.p = d."""
        a, b, c = (P[i] for i in self._rng.integers(0, len(P), (3, 150)))
        nv = np.cross(b - a, c - a)
        nn = np.linalg.norm(nv, axis=1)
        ok = nn > 1e-6
        nv = nv[ok] / nn[ok, None]
        nv *= np.where(nv[:, 2:3] < 0, -1.0, 1.0)            # point up
        d = (a[ok] * nv).sum(1)
        dist = np.abs(P @ nv.T - d)
        k = int((dist < 0.15).sum(0).argmax())
        q = P[dist[:, k] < 0.15]
        c0 = q.mean(0)
        n = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        n = n if n[2] > 0 else -n
        return n, float(n @ c0)

    def _level(self, P, ground):
        """Rotate P so the smoothed ground plane is horizontal and shift it to z = 0."""
        if len(ground) < MIN_GROUND_POINTS and self._gn is None and not self._warned:
            self._warned = True
            self.node.get_logger().warning(
                "auto-level: too few ground points in the lower-centre of the image; cloud is not levelled yet "
                "(use --pitch instead)")
        if len(ground) >= MIN_GROUND_POINTS:
            n, d = self._fit_ground(ground)
            if self._gn is None:
                self._gn, self._gd = n, d
            else:
                self._gn = self.smooth * self._gn + (1 - self.smooth) * n
                self._gn /= np.linalg.norm(self._gn)
                self._gd = self.smooth * self._gd + (1 - self.smooth) * d
        if self._gn is None:
            return P
        v, cosang = np.cross(self._gn, [0, 0, 1.0]), float(self._gn[2])
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + cosang)
        out = P @ R.T
        out[:, 2] -= self._gd                                  # ground plane (z' = d < 0) -> z = 0
        return out

    def publish(self, rgb, depth, stamp_ns=None, frame_id=None):
        """Publish one cloud; returns the number of points."""
        h, w = depth.shape
        s = self.stride
        o = s // 2
        rx, ry = self._ray_grid(h, w)
        z, norm = self.depth_metres(depth[o::s, o::s])
        valid = norm > 0                       # pixels beyond `far` (sky etc.) are dropped
        X, Y, Z = rx * z, ry * z, z            # camera optical axes
        col = rgb[o::s, o::s][valid].astype(np.uint32)
        if self.axes == "ros":
            P = np.stack([Z[valid], -X[valid], -Y[valid]], 1)   # x forward, y left, z up
            if self.auto_level:
                hh, ww = z.shape
                cand = np.zeros_like(valid)
                cand[int(hh * .55):int(hh * .9), int(ww * .2):int(ww * .8)] = True
                cand &= valid & (z > 1.5)          # lower-centre image, skipping the vehicle body
                P = self._level(P, np.stack([Z[cand], -X[cand], -Y[cand]], 1))
            else:
                if self.pitch:
                    c, sn = np.cos(self.pitch), np.sin(self.pitch)   # camera pitched down by `pitch`
                    P = np.stack([P[:, 0] * c + P[:, 2] * sn, P[:, 1], -P[:, 0] * sn + P[:, 2] * c], 1)
                P[:, 2] += self.cam_height
        else:
            P = np.stack([X[valid], Y[valid], Z[valid]], 1)
        pts = np.empty(len(P), DTYPE)
        pts["x"], pts["y"], pts["z"] = P[:, 0], P[:, 1], P[:, 2]
        pts["rgb"] = (col[:, 0] << 16) | (col[:, 1] << 8) | col[:, 2]

        msg = PointCloud2()
        if self.stamp_mode in ("bag", "image") and stamp_ns is not None:
            msg.header.stamp = Time(sec=int(stamp_ns // 10**9), nanosec=int(stamp_ns % 10**9))
        else:
            msg.header.stamp = self.node.get_clock().now().to_msg()
        # "image" -> reuse each image's own header.frame_id (only useful if the bag's camera frame is stable)
        msg.header.frame_id = (frame_id or self.frame_id) if self.frame_id == "image" else self.frame_id
        msg.height, msg.width, msg.fields = 1, len(pts), FIELDS
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step, msg.row_step = DTYPE.itemsize, DTYPE.itemsize * len(pts)
        msg.data = array.array("B", pts.tobytes())   # rclpy's fast path; a plain bytes value is validated byte by byte in Python
        self.pub.publish(msg)
        return len(pts)

    def reset_range(self):
        self._lo = self._hi = None
        self._gn = self._gd = None

    def shutdown(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
