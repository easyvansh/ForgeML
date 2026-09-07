# CUDA milestone

`attention_forward.cu` is a correctness-first custom CUDA forward kernel. It computes causal attention with an online max/sum update and does not materialize a score matrix. One thread computes one output channel and recomputes the score row; this is intentionally simple and will be slower than a tiled production kernel.

Build from PowerShell with the CUDA toolkit and Visual Studio Build Tools:

```powershell
.
scripts\build_cuda.ps1
.\build\cuda\attention_forward.exe 128
.\build\cuda\attention_forward.exe 256
```

Run the checked-in context sweep and save JSONL evidence:

```powershell
.\scripts\run_cuda_forward_sweep.ps1
```

The sweep covers contexts 128, 256, and 512 with 50 timed launches per context. Its output is `results/cuda_attention_forward.jsonl`.

The script loads the Visual Studio 2019 Build Tools environment before calling `nvcc`. Calling `nvcc` directly from an ordinary PowerShell session fails with `Cannot find compiler 'cl.exe' in PATH`, even when CUDA is installed. If your Build Tools are installed elsewhere, update `$vsDev` in the script. The executable commands intentionally use `.\`; PowerShell does not run a program from the current directory when given `build\...` without that prefix.

The RTX 3060 uses compute capability 8.6. The executable prints a JSON record with maximum CPU-reference error and CUDA-event kernel timing. This is a forward-only reference kernel. It has no backward pass, shared-memory tiling, warp specialization, mixed precision, or PyTorch/ForgeML tensor binding, so it must not be called FlashAttention.

The shared-memory tiled reference is built and run with:

```powershell
.\scripts\build_tiled_cuda.ps1
.\build\cuda\attention_tiled.exe 128
```

It stages K/V tiles in shared memory for a correctness-first comparison. It is not yet a warp-specialized production kernel.
