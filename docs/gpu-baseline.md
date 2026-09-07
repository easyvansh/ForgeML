# GPU baseline protocol

`benchmarks/torch_attention.py` is the first GPU milestone. It records device, CUDA/PyTorch versions, CUDA-event latency, peak allocated bytes, and reserved bytes for several context lengths. It is a PyTorch framework baseline and exists to establish a measurement protocol before a custom kernel is introduced.

Run from the repository root inside `.venv`:

```powershell
.\.venv\Scripts\python.exe benchmarks/torch_attention.py --seq 128 256 512 1024 --repeats 20 --output results/torch_attention.json
```

The current script does not claim that its second implementation is a custom materialized kernel. The custom CUDA milestone begins only when ForgeML owns the forward and backward kernels and passes numerical parity at the same shapes. The benchmark must then randomize method order, warm up both methods, synchronize correctly, record OOMs, and report the exact backend selected.

The measured output is visualized in `docs/figures/attention_memory.svg` and `docs/figures/attention_latency.svg`. Read those figures with the raw JSON metadata and the limitations above.
