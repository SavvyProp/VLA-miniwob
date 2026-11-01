# TinyVLA runtime installation

Run these commands from the repository root on the machine holding the collected
examples. The model runtime lives in `integrations/tinyvla/.venv`, with a separate
`uv.lock`; collection keeps using the root environment.

```bash
# NVIDIA GPU installation and import/device check; no model weights downloaded.
bash scripts/setup_tinyvla.sh cuda

# Alternatively, install a CPU runtime for tests and small diagnostics.
bash scripts/setup_tinyvla.sh cpu
```

Choose one backend per environment. `uv` removes the previous backend when you
switch. Python 3.11 or 3.12 is required; `uv` selects an appropriate interpreter.
The CUDA build uses PyTorch 2.6.0 and CUDA 12.4. It requires a compatible NVIDIA
GPU and driver. PyTorch includes its CUDA runtime; no local CUDA compiler,
FlashAttention, DeepSpeed, or bitsandbytes is needed. CPU operation is supported
for diagnostics but full backbone training is intended for a GPU.

Use the same selected backend for every training and evaluation command:

```bash
uv run --project integrations/tinyvla --frozen --extra cuda vla-benchmark --help
```

Replace `cuda` with `cpu` for the diagnostic environment. Passing `--project`
selects the environment without changing the working directory, so paths such
as `exports/train` and `runs/...` remain relative to the repository root.
Both environments include the benchmark and its browser dependencies. Browser
evaluation also needs the Chrome/ChromeDriver installation used for collection.

The source is installed directly from TinyVLA commit
`94f441827b45e4f76316ef6a0ae443736dc93a5d`, reusing its native backbone and
diffusion head. Upstream calls its root package `act`; this is the package that
actually contains `policy_heads`. The `llava-pythia` dependency metadata is
overridden in the integration project to replace its unrelated UI, robotics,
and distributed-training requirements with the explicit dependencies in this
runtime. Its model source is unchanged. The Transformers version follows
upstream's native model API. Hugging Face Hub is pinned alongside Diffusers to
retain the `cached_download` API used by this Diffusers release.

Inspect imports and device availability again without downloading weights:

```bash
uv run --project integrations/tinyvla --frozen --extra cuda \
  python scripts/check_tinyvla.py --require-cuda
```

The first real training invocation downloads the pinned
`lesjie/Llava-Pythia-700M` backbone from Hugging Face. Reserve disk space for the
Python runtime, backbone cache, optimizer state, and training checkpoints.
The import check does not download or validate pretrained weights.

Sources: [TinyVLA source](https://github.com/liyaxuanliyaxuan/TinyVLA/tree/94f441827b45e4f76316ef6a0ae443736dc93a5d),
[PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/), and
[uv dependency metadata](https://docs.astral.sh/uv/reference/settings/#dependency-metadata).
