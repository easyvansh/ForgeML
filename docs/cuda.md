# GPU backend implementation plan

## Observed environment

On 2026-09-07, `nvidia-smi` reported an RTX 3060-family GPU, 6144 MiB VRAM, WDDM mode, driver 560.94 and driver CUDA compatibility 12.6. `nvcc` was not available on PATH. GPU compute and toolkit installations have not been verified. PyTorch was not importable in the active Python environment.

## Boundary to implement

Introduce explicit device storage, dtype and ownership instead of embedding device assumptions in Tensor. Tensor operations dispatch to CPU or CUDA implementations; VJPs remain an explicit contract. Device transfers must be visible and never silently occur inside benchmark timings.

Use C++ bindings with error-checked allocations, launches, shape checks, strides and a documented stream policy. Start with contiguous float32 tensors. Specify ownership of outputs and saved backward state. Avoid a caching allocator until correctness is established, then measure allocation overhead before adding one.

## Ordered milestones

1. Build a small extension and device-buffer round trip; test allocation failure and cleanup.
2. Vector operations and reductions, including odd sizes and empty inputs where supported.
3. Stable softmax and LayerNorm forward/backward, checked against float64 CPU values.
4. Library GEMM integration for a credible training baseline. An educational tiled GEMM is a separate benchmark; do not let it dominate and obscure attention results.
5. Materialized causal attention forward/backward with clear workspace accounting.
6. Online-softmax tiled forward and recomputation-based backward; save normalization statistics.
7. Tune tile size, warp layout, register pressure and shared-memory use from measured evidence.

No CUDA source stubs are presented as implemented kernels in this release.

## Profiling protocol

Capture GPU identity, driver/toolkit versions, dtype, shapes, warmups, clock/power behavior and background load. Use CUDA events for device timing and synchronized wall-clock measurements for end-to-end throughput. Capture peak allocated and reserved bytes separately if the allocator distinguishes them. Record OOMs and measurement errors.

Use Nsight tools where supported to inspect launch overhead, occupancy, register usage, shared-memory utilization and memory traffic. Counter availability depends on hardware and permissions. Never substitute an analytic score-matrix size for measured HBM traffic. RTX hardware memory should be described as device/global memory rather than assuming every GPU uses HBM.

The acceptance gate is a correct full training step, followed by profiling. A fast forward-only kernel is not sufficient for a training-runtime claim.

## Windows build troubleshooting

`nvcc` is the CUDA compiler, but on Windows it delegates host C++ compilation to Microsoft `cl.exe`. If `nvcc` says `Cannot find compiler 'cl.exe' in PATH`, open a **Developer Command Prompt for VS** or run the repository helper:

```powershell
.\scripts\build_cuda.ps1
```

The helper currently targets the detected Visual Studio 2019 Build Tools installation at `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`. If the executable was not created, do not run it; PowerShell will correctly report that the path does not exist. After a successful build, invoke it with `.\build\cuda\attention_forward.exe 128`.

If compilation succeeds but execution fails at `cudaMalloc`, compare `nvcc -V` with `nvidia-smi`. This machine currently reports toolkit 13.3 and driver CUDA compatibility 12.6. Install a compatible driver/toolkit pair: update the NVIDIA driver for CUDA 13.3, or install a CUDA 12.6 toolkit and compile with that `nvcc`. A successful compile does not prove that the runtime can launch on the installed driver.
