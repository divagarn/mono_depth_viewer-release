"""Read image frames from a ROS 2 sqlite3 (.db3) bag without needing a running ROS graph."""
import os
import sqlite3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

IMAGE_TYPES = {"sensor_msgs/msg/Image": Image, "sensor_msgs/msg/CompressedImage": CompressedImage}


def _find_db3(path):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".db3"))
        if not files:
            raise FileNotFoundError(f"No .db3 file in {path} (mcap bags are not supported)")
        return [os.path.join(path, f) for f in files]
    if path.endswith(".db3"):
        return [path]
    raise ValueError("Select a rosbag2 folder or a .db3 file")


def image_to_rgb(msg):
    """Convert sensor_msgs/Image or CompressedImage to an HxWx3 uint8 RGB array."""
    if isinstance(msg, CompressedImage):
        bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    raw = np.frombuffer(bytes(msg.data), np.uint8)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if "a" in enc else 3
        img = raw.reshape(h, msg.step)[:, : w * ch].reshape(h, w, ch)
        code = {"rgb8": None, "bgr8": cv2.COLOR_BGR2RGB, "rgba8": cv2.COLOR_RGBA2RGB, "bgra8": cv2.COLOR_BGRA2RGB}[enc]
        return img if code is None else cv2.cvtColor(img, code)
    if enc in ("mono8", "8uc1"):
        img = raw.reshape(h, msg.step)[:, :w]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if enc.startswith("bayer_"):
        img = raw.reshape(h, msg.step)[:, :w]
        code = {"bayer_rggb8": cv2.COLOR_BAYER_BG2RGB, "bayer_bggr8": cv2.COLOR_BAYER_RG2RGB,
                "bayer_gbrg8": cv2.COLOR_BAYER_GR2RGB, "bayer_grbg8": cv2.COLOR_BAYER_GB2RGB}[enc]
        return cv2.cvtColor(img, code)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class BagReader:
    def __init__(self, path):
        self.path = path
        self.dbs = [sqlite3.connect(f"file:{f}?mode=ro", uri=True) for f in _find_db3(path)]
        self.topics = {}  # name -> (type, class)
        for db in self.dbs:
            for name, typ in db.execute("SELECT name, type FROM topics"):
                if typ in IMAGE_TYPES:
                    self.topics[name] = (typ, IMAGE_TYPES[typ])
        self._index = []  # [(db_idx, message_id, timestamp_ns)]

    def select_topic(self, topic):
        """Index all messages of `topic`; returns the frame count."""
        self.topic = topic
        self.cls = self.topics[topic][1]
        self._index = []
        for i, db in enumerate(self.dbs):
            q = ("SELECT m.id, m.timestamp FROM messages m JOIN topics t ON t.id = m.topic_id "
                 "WHERE t.name = ? ORDER BY m.timestamp")
            self._index += [(i, mid, ts) for mid, ts in db.execute(q, (topic,))]
        self._index.sort(key=lambda r: r[2])
        return len(self._index)

    def __len__(self):
        return len(self._index)

    def timestamp(self, k):
        return self._index[k][2] * 1e-9

    def frame(self, k):
        i, mid, _ = self._index[k]
        (data,) = self.dbs[i].execute("SELECT data FROM messages WHERE id = ?", (mid,)).fetchone()
        return image_to_rgb(deserialize_message(data, self.cls))

    def close(self):
        for db in self.dbs:
            db.close()
