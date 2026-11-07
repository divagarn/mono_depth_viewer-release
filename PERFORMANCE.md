# Performance and hardware requirements

Measured on **2026-11-07**. Everything marked *measured* was run on the machine below with the scripts in
[`benchmarks/`](benchmarks/). Everything marked *projection* is **not** measured by us and is labelled as such.

## Summary

| | Result | Status |
|---|---|---|
| Current rate, default settings (`small`, input 364, stride 4, auto-level) | **0.72–0.76 Hz** (about 1.4 s per cloud) on CPU | measured |
| Current rate, fast settings (`small`, input 252, stride 8, auto-level) | **2.0 Hz** (about 0.5 s per cloud) on CPU | measured |
| Target: 30 FPS / 30 Hz | **not reached** on the tested hardware (needs about 15× to 40× more speed) | measured gap |
| Hardware that should reach 30 Hz | an NVIDIA GPU class of V100 / A100 / RTX 4090 with the `small` or `base` model | **projection, untested** |

On CPU the depth network is the bottleneck: it is 90% or more of the time per frame. The point-cloud step is not
(7–10 ms at stride 4).

## Tested hardware and software

| | |
|---|---|
| CPU | Intel Core Ultra 5 125U (12 cores / 14 threads, max 4.3 GHz, 12 MiB L3), 15 W-class laptop chip |
| RAM | 30 GiB |
| GPU | **none used**. Intel integrated graphics only; `torch.cuda.is_available()` is `False` |
| OS / kernel | Ubuntu 22.04.5 LTS, Linux 6.8.0-138-generic |
| ROS | ROS 2 Humble |
| Python / torch | Python 3.10.12, PyTorch 1.12.1+cu113 (CPU execution, 12 threads), transformers 4.40.2, NumPy 1.26.4, OpenCV 4.11.0 |
| Power state | **on battery** (`Discharging`), CPU governor `powersave` with energy preference `balance_performance` |
| Input data | `new_delhi_bag_3`: `/camera/image_raw`, 960×540 `rgba8`, recorded at 3.56 Hz (627 frames, 176 s) |

The laptop was on battery and the CPU slows down under sustained load, so repeated runs get slower. Results on AC power
or on a desktop CPU may be better. Rows that varied between runs are given as ranges.

## Measured results

### Depth model alone (one frame, CPU)

Median time over 12 frames sampled across the bag; the range is the lowest and highest of 3 runs (2 for `base`).

| Model | Input size | Time per frame | FPS |
|---|---|---|---|
| `small` (Depth Anything V1 Small, 24.8 M parameters) | 252 | 398–526 ms | 1.9–2.5 |
| `small` | 364 (default) | 932–1512 ms | 0.66–1.07 |
| `small` | 518 | 4275–4361 ms | 0.23 |
| `base` (97.5 M parameters) | 252 | 1078–1511 ms | 0.66–0.93 |
| `base` | 364 | 3898–3899 ms | 0.26 |
| `large`, `dpt` | | not measured | |

Model load time: about 1 s (weights already downloaded). Weights on disk: about 100 MB (`small`), 370 MB (`base`).

### Rest of the pipeline (per frame, CPU, single thread)

| Step | Time |
|---|---|
| Read and decode one image (bag read + conversion) | 3–5 ms |
| Build and publish the point cloud, stride 8 (7,810 points, 0.125 MB) | 3–5 ms |
| Build and publish the point cloud, stride 4 (31,481 points, 0.50 MB) | 7–10 ms |
| Build and publish the point cloud, stride 2 (125,912 points, 2.0 MB) | 19–25 ms |

This includes the auto-level ground fit (about 1 ms). The steps run one after the other in a single worker thread, so
the highest possible rate is `1 / (decode + depth model + cloud)`.

### The running node, topic rates (measured)

The node was fed a constant 30 Hz image stream (`benchmarks/fake_camera.py`, 960×540 frames from the bag), so
it always has a new frame waiting. The output rate is therefore the node's maximum throughput.
Settings: `--hfov 100 --auto-level`.

| Settings | Output rate | Gap between clouds | Cloud size |
|---|---|---|---|
| `small`, input 364, stride 4 | **0.72 Hz** (node log: 0.70–0.99, mean 0.76) | median 1379 ms, worst 1456 ms | 0.51 MB |
| `small`, input 252, stride 8 | **2.03 Hz** (node log: 1.94–2.08, mean 2.01) | median 493 ms, worst 564 ms | 0.13 MB |

- Only the newest camera frame is processed and older ones are dropped, so a slow node lowers the output rate but never
  falls behind or builds a queue.
- Resources while running (`small`): about 530–630 MB RAM. CPU use averaged about 620% (about 6 of 14 threads) at the
  default settings, because PyTorch spreads the work over all cores.
- Each published cloud carries the timestamp of the image it came from.

## What is needed for 30 FPS

A 30 Hz output leaves **33.3 ms per frame** for the whole pipeline.

**Time budget (measured parts):** decode 3–5 ms, plus cloud build 7–10 ms at stride 4 (3–5 ms at stride 8). That
leaves about **20 ms (stride 4) to 25 ms (stride 8) for the depth model**, including its own resize, normalise and
copy steps in `predict()`. Those steps were not timed on a GPU.

**Gap on the tested hardware:** the fastest measured setting (`small`, input 252) takes about 0.4–0.5 s per frame, so it
would need to be roughly 16–25× faster to fit in 20–25 ms, and the default setting about 37–75× faster. No CPU
gets that, so **30 FPS needs a GPU**.

### Projection (not measured by us)

The authors of Depth Anything V1 publish these GPU inference times
(<https://github.com/LiheYoung/Depth-Anything>, README). Their V100 and A100 numbers **exclude** pre- and
post-processing; their RTX 4090 numbers use **TensorRT** and include it. The page does not state the input resolution.

| Model | V100 | A100 | RTX 4090 (TensorRT) |
|---|---|---|---|
| `small` | 12 ms | 8 ms | 3 ms |
| `base` | 13 ms | 9 ms | 6 ms |
| `large` | 20 ms | 13 ms | 12 ms |

Reading these against the 20–25 ms budget above: `small` and `base` on a V100 / A100 / RTX 4090 class GPU look
feasible, `large` does not leave room on a V100. Our own pre/post-processing on the GPU, the host copy of the depth map
and a real camera driver are extra, and we have **not** run any of this. Treat these as expectations, not results.

### Requirements list

| For 30 Hz | Requirement |
|---|---|
| GPU | NVIDIA with CUDA and a PyTorch build that supports it (the PyTorch here is run on CPU). V100 / A100 / RTX 4090 class or better is the projection for `small`/`base`. Weights in 32-bit floats are about 0.1 GB (`small`), 0.4 GB (`base`) and 1.3 GB (`large`), plus activations and the CUDA context; 2–4 GB or more of GPU memory is a reasonable guess (projection) |
| CPU | One fast core for the serial decode and cloud steps (7–10 ms at stride 4 on the tested CPU). On slower CPUs use `--pc-stride 8` (3–5 ms) |
| RAM | 4 GB free is enough; measured 0.5–1.2 GB |
| Camera | publishes at 30 Hz or more |
| Input bandwidth | 960×540 `rgb8` at 30 Hz = 46.7 MB/s (`rgba8`: 62.2 MB/s) |
| Output bandwidth | stride 4: 15.1 MB/s, stride 8: 3.7 MB/s, stride 2: 60 MB/s |
| Other platforms (Jetson Orin, etc.) | **not tested**; measure with the scripts below |

Ways to go faster that we have **not** tried: half-precision, TensorRT, smaller input size, a larger `--pc-stride`, and
running inference and cloud building in separate threads.

## Measure your own hardware

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH          # run from the repository root

# 1) model latency and per-step pipeline cost (no ROS graph needed)
python3 benchmarks/benchmark.py --bag /path/to/bag --model small --sizes 252,364,518 --strides 2,4,8

# 2) real topic rate of the running node, fed at 30 Hz
python3 -m mono_depth_viewer.main --hfov 100 --auto-level &                  # the node
python3 benchmarks/fake_camera.py --bag /path/to/bag --rate 30 &              # 30 Hz camera stand-in
python3 benchmarks/measure_hz.py /mono_depth/points --seconds 40 --skip 15    # Hz, gap, size, bandwidth
```

Close other programs first and note the power state. Run it several times, because laptop CPUs slow down under load.
Add your results to the table below.

## Results log

| Date | Hardware | Model / input / stride | Model time | Output rate | Notes |
|---|---|---|---|---|---|
| 2026-11-07 | Intel Core Ultra 5 125U, 30 GiB, CPU only, battery | `small` / 364 / 4 | 0.93–1.5 s | 0.72–0.76 Hz | torch 1.12.1 |
| 2026-11-07 | Intel Core Ultra 5 125U, 30 GiB, CPU only, battery | `small` / 252 / 8 | 0.40–0.53 s | 2.0 Hz | torch 1.12.1 |
| | | | | | |
