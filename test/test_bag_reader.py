import numpy as np
from sensor_msgs.msg import Image

from mono_depth_viewer.bag_reader import image_to_rgb


def _msg(encoding, channels, data):
    m = Image()
    m.height, m.width, m.encoding = data.shape[0], data.shape[1], encoding
    m.step = m.width * channels
    m.data = data.tobytes()
    return m


def test_rgba8_drops_alpha():
    img = np.random.randint(0, 255, (4, 6, 4), np.uint8)
    out = image_to_rgb(_msg("rgba8", 4, img))
    assert out.shape == (4, 6, 3) and (out == img[..., :3]).all()


def test_bgr8_is_swapped():
    img = np.random.randint(0, 255, (4, 6, 3), np.uint8)
    out = image_to_rgb(_msg("bgr8", 3, img))
    assert (out == img[..., ::-1]).all()


def test_mono8_gives_three_channels():
    img = np.random.randint(0, 255, (4, 6), np.uint8)
    assert image_to_rgb(_msg("mono8", 1, img)).shape == (4, 6, 3)
