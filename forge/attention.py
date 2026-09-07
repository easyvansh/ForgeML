"""Forward-only streaming reference; CPU NumPy, not a CUDA implementation."""
import numpy as np

def dense_attention(q,k,v):
    scores=q@k.swapaxes(-1,-2)/np.sqrt(q.shape[-1])
    t=q.shape[-2]
    scores=np.where(np.tril(np.ones((t,t),dtype=bool)),scores,-np.inf)
    p=np.exp(scores-scores.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
    return p@v

def streaming_attention(q,k,v,tile=32):
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4: raise ValueError('Expected equal [B,H,T,D] arrays')
    if tile<1: raise ValueError('tile must be positive')
    b,h,t,d=q.shape
    out=np.empty_like(q)
    for start in range(0,t,tile):
        end=min(start+tile,t); qi=q[:,:,start:end,:]
        m=np.full((b,h,end-start,1),-np.inf); l=np.zeros_like(m); acc=np.zeros_like(qi)
        for j in range(0,end,tile):
            je=min(j+tile,t)
            s=qi@k[:,:,j:je,:].swapaxes(-1,-2)/np.sqrt(d)
            s=np.where(np.arange(j,je)[None,:]<=np.arange(start,end)[:,None],s,-np.inf)
            new_m=np.maximum(m,s.max(-1,keepdims=True))
            alpha=np.exp(m-new_m); p=np.exp(s-new_m)
            acc=alpha*acc+p@v[:,:,j:je,:]
            l=alpha*l+p.sum(-1,keepdims=True); m=new_m
        out[:,:,start:end,:]=acc/l
    return out
