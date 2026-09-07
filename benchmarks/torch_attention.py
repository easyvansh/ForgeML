"""CUDA reference benchmark used before custom kernels exist.

This intentionally benchmarks PyTorch SDPA and a materialized reference. It is
baseline evidence, not a ForgeML CUDA kernel or a FlashAttention claim.
"""
import argparse, json, platform, time
from pathlib import Path
import torch

def run_context(seq, batch, heads, head_dim, repeats):
    device='cuda'; dtype=torch.float16
    q=torch.randn(batch,heads,seq,head_dim,device=device,dtype=dtype)
    k=torch.randn_like(q); v=torch.randn_like(q)
    causal=True; rows=[]
    def materialized():
        scores=q@k.transpose(-1,-2)/(head_dim**0.5)
        mask=torch.triu(torch.ones(seq,seq,device=device,dtype=torch.bool),1)
        scores=scores.masked_fill(mask,-torch.inf)
        return scores.softmax(-1)@v
    implementations=[('sdpa',lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)),('materialized',materialized)]
    for name,fn in implementations:
        for _ in range(3): fn(); torch.cuda.synchronize()
        samples=[]
        for _ in range(repeats):
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize(); samples.append(start.elapsed_time(end)/1000)
        rows.append({'implementation':name,'context':seq,'median_seconds':float(torch.tensor(samples).median()),'samples_seconds':samples,
                     'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()})
        torch.cuda.reset_peak_memory_stats()
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/torch_attention.json'); ap.add_argument('--seq',nargs='+',type=int,default=[128,256,512,1024]); ap.add_argument('--repeats',type=int,default=20)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable')
    payload={'system':{'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),'dtype':'float16'},'results':[]}
    for seq in args.seq: payload['results'].extend(run_context(seq,1,4,64,args.repeats))
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
