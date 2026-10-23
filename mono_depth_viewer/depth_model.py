"""Monocular depth estimation with transformer models (Hugging Face)."""
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "Depth Anything V1 Small (fast)": ("LiheYoung/depth-anything-small-hf", 14),
    "Depth Anything V1 Base": ("LiheYoung/depth-anything-base-hf", 14),
    "Depth Anything V1 Large (best, slow)": ("LiheYoung/depth-anything-large-hf", 14),
    "DPT Hybrid MiDaS": ("Intel/dpt-hybrid-midas", 32),
}
COLORMAPS = {"Turbo": cv2.COLORMAP_TURBO, "Inferno": cv2.COLORMAP_INFERNO,
             "Magma": cv2.COLORMAP_MAGMA, "Gray": None}


class DepthEstimator:
    def __init__(self, model_name, device=None):
        repo, self.multiple = MODELS[model_name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoImageProcessor.from_pretrained(repo)
        self.mean = np.array(proc.image_mean, np.float32)
        self.std = np.array(proc.image_std, np.float32)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).to(self.device).eval()
        self._lo = self._hi = None  # smoothed normalisation range

    @torch.inference_mode()
    def predict(self, rgb, input_size=518):
        """Return relative inverse depth (larger = closer) at the input image resolution."""
        h, w = rgb.shape[:2]
        scale = input_size / min(h, w)
        m = self.multiple
        nh, nw = max(m, round(h * scale / m) * m), max(m, round(w * scale / m) * m)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - self.mean) / self.std).transpose(2, 0, 1))[None].to(self.device)
        d = self.model(pixel_values=x).predicted_depth[None]
        d = torch.nn.functional.interpolate(d, (h, w), mode="bicubic", align_corners=False)
        return d[0, 0].float().cpu().numpy()

    def colorize(self, depth, cmap="Turbo", smooth=0.8):
        """Normalise to 0-255 using an EMA'd range (limits flicker between frames) and apply a colormap."""
        lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
        if self._lo is None or smooth <= 0:
            self._lo, self._hi = lo, hi
        else:
            self._lo = smooth * self._lo + (1 - smooth) * lo
            self._hi = smooth * self._hi + (1 - smooth) * hi
        g = np.clip((depth - self._lo) / max(self._hi - self._lo, 1e-6), 0, 1)
        g = (g * 255).astype(np.uint8)
        code = COLORMAPS[cmap]
        if code is None:
            return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(cv2.applyColorMap(g, code), cv2.COLOR_BGR2RGB)

    def reset_range(self):
        self._lo = self._hi = None
