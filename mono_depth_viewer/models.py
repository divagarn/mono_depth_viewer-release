"""Names of the supported depth models (kept free of heavy imports so --help stays fast)."""

MODELS = {
    "Depth Anything V1 Small (fast)": ("LiheYoung/depth-anything-small-hf", 14),
    "Depth Anything V1 Base": ("LiheYoung/depth-anything-base-hf", 14),
    "Depth Anything V1 Large (best, slow)": ("LiheYoung/depth-anything-large-hf", 14),
    "DPT Hybrid MiDaS": ("Intel/dpt-hybrid-midas", 32),
}
MODEL_ALIASES = {"small": "Depth Anything V1 Small (fast)", "base": "Depth Anything V1 Base",
                 "large": "Depth Anything V1 Large (best, slow)", "dpt": "DPT Hybrid MiDaS"}
