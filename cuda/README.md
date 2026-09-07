# CUDA milestone

`attention_forward.cu` is a correctness-first custom CUDA forward kernel. It computes causal attention with an online max/sum update and does not materialize a score matrix. One thread computes one output channel and recomputes the score row; this is intentionally simple and will be slower than a tiled production kernel.

Build from the repository root with the CUDA toolkit:

```powershell
New-Item -ItemType Directory -Force build\cuda | Out-Null
nvcc -O3 -arch=sm_86 cuda\attention_forward.cu -o build\cuda\attention_forward.exe
build\cuda\attention_forward.exe 128
build\cuda\attention_forward.exe 256
```

The RTX 3060 uses compute capability 8.6. The executable prints a JSON record with maximum CPU-reference error and CUDA-event kernel timing. This is a forward-only reference kernel. It has no backward pass, shared-memory tiling, warp specialization, mixed precision, or PyTorch/ForgeML tensor binding, so it must not be called FlashAttention.
