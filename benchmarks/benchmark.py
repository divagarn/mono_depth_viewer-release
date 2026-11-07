#!/usr/bin/env python3
"""Measure depth-model latency and the point-cloud pipeline on this machine.

    source /opt/ros/humble/setup.bash
    python3 benchmarks/benchmark.py --bag /path/to/bag --model small --sizes 252,364,518

Prints the machine, per-input-size model latency, and the per-stage pipeline cost (bag read/decode, depth model,
point cloud build + publish).  Use measure_hz.py to measure the real topic rates of a running node.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import time

import numpy as np
import torch

from mono_depth_viewer.bag_reader import BagReader
from mono_depth_viewer.depth_model import MODEL_ALIASES, DepthEstimator
from mono_depth_viewer.pointcloud import PointCloudPublisher


def cpu_name():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor() or "unknown"


def stats(ms):
    return {"median_ms": round(statistics.median(ms), 1), "worst_ms": round(max(ms), 1), "n": len(ms)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True)
    p.add_argument("--topic", default="/camera/image_raw")
    p.add_argument("--model", choices=MODEL_ALIASES, default="small")
    p.add_argument("--sizes", default="252,364,518", help="network input short-side sizes to test")
    p.add_argument("--strides", default="4,8", help="point cloud strides to test")
    p.add_argument("--pipeline-size", type=int, default=364, help="input size used for the pipeline breakdown")
    p.add_argument("--frames", type=int, default=12, help="frames sampled evenly across the bag")
    p.add_argument("--json", help="also write the results to this file")
    a = p.parse_args()

    reader = BagReader(a.bag)
    reader.select_topic(a.topic)
    idx = np.linspace(0, len(reader) - 1, a.frames).astype(int)
    res = {"cpu": cpu_name(), "threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "model": a.model, "frames": a.frames}

    t = []
    frames = []
    for k in idx:
        t0 = time.perf_counter()
        frames.append(reader.frame(int(k)))
        t.append((time.perf_counter() - t0) * 1000)
    res["image"] = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
    res["bag_read_decode"] = stats(t)

    t0 = time.perf_counter()
    est = DepthEstimator(MODEL_ALIASES[a.model])
    res["model_load_s"] = round(time.perf_counter() - t0, 1)
    sync = torch.cuda.synchronize if res["cuda"] else (lambda: None)

    res["predict"] = {}
    depths = {}
    for size in map(int, a.sizes.split(",")):
        est.predict(frames[0], size)
        est.predict(frames[0], size)                       # warm-up
        t = []
        for f in frames:
            t0 = time.perf_counter()
            d = est.predict(f, size)
            sync()
            t.append((time.perf_counter() - t0) * 1000)
        depths[size] = [est.predict(f, size) for f in frames] if size == a.pipeline_size else None
        res["predict"][size] = {**stats(t), "fps": round(1000 / statistics.median(t), 2)}
        print(f"  {a.model} @ {size}: {res['predict'][size]}", flush=True)

    res["pipeline"] = {}
    ds = depths[a.pipeline_size] if a.pipeline_size in depths and depths[a.pipeline_size] else None
    if ds is None:
        ds = [est.predict(f, a.pipeline_size) for f in frames]
    for stride in map(int, a.strides.split(",")):
        pub = PointCloudPublisher(topic=f"/bench_{stride}", hfov=100, stride=stride, auto_level=True)
        for f, d in list(zip(frames, ds))[:3]:
            pub.publish(f, d)                              # settle the smoothing / warm up
        t, npts = [], 0
        for f, d in zip(frames, ds):
            t0 = time.perf_counter()
            npts = pub.publish(f, d)
            t.append((time.perf_counter() - t0) * 1000)
        res["pipeline"][stride] = {**stats(t), "points": npts, "bytes": npts * 16}
        print(f"  cloud build+publish stride {stride}: {res['pipeline'][stride]}", flush=True)
        pub.shutdown()

    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
