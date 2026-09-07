# ForgeML figure manifest

| Asset | Source | Meaning | Status |
|---|---|---|---|
| `architecture.mmd` | Runtime components | End-to-end system and research path | Design diagram |
| `attention_algorithm.md` | `forge/attention.py`, `cuda/attention_forward.cu` | Online-softmax recurrence | Algorithm explanation |
| `attention_memory.svg` | `results/torch_attention.json` | PyTorch GPU allocation baseline | Measured baseline |
| `attention_latency.svg` | `results/torch_attention.json` | PyTorch GPU CUDA-event latency baseline | Measured baseline |

The SVG charts are static, reviewable artifacts. Update them whenever the raw benchmark changes. They are baseline evidence, not custom ForgeML-kernel results.
