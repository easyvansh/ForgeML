import numpy as np
from .tensor import Tensor

class Module:
    training=True
    def named_parameters(self):
        seen=set()
        def walk(obj,path):
            if isinstance(obj,Tensor) and obj.requires_grad:
                if id(obj) not in seen:
                    seen.add(id(obj)); yield path,obj
            elif isinstance(obj,Module):
                for key,value in vars(obj).items(): yield from walk(value,f'{path}.{key}' if path else key)
            elif isinstance(obj,(list,tuple)):
                for i,value in enumerate(obj): yield from walk(value,f'{path}.{i}')
        yield from walk(self,'')
    def parameters(self): return [p for _,p in self.named_parameters()]
    def zero_grad(self):
        for p in self.parameters(): p.grad=None
    def train(self,mode=True):
        self.training=mode
        for value in vars(self).values():
            if isinstance(value,Module): value.train(mode)
            elif isinstance(value,(list,tuple)):
                for child in value:
                    if isinstance(child,Module): child.train(mode)
        return self
    def eval(self): return self.train(False)
    def __call__(self,*args): return self.forward(*args)

class Linear(Module):
    def __init__(self,ins,outs,rng):
        self.weight=Tensor(rng.normal(0,0.02,(ins,outs)),True)
        self.bias=Tensor(np.zeros(outs),True)
    def forward(self,x): return x@self.weight+self.bias

class Embedding(Module):
    def __init__(self,count,width,rng): self.weight=Tensor(rng.normal(0,0.02,(count,width)),True)
    def forward(self,ids): return self.weight[ids]

class LayerNorm(Module):
    def __init__(self,width,eps=1e-5):
        self.weight=Tensor(np.ones(width),True); self.bias=Tensor(np.zeros(width),True); self.eps=eps
    def forward(self,x):
        z=x-x.mean(-1,True)
        return z/( (z*z).mean(-1,True)+self.eps)**0.5*self.weight+self.bias

class Dropout(Module):
    def __init__(self,p,rng):
        if not 0<=p<1: raise ValueError('dropout must be in [0,1)')
        self.p,self.rng=p,rng
    def forward(self,x):
        if not self.training or self.p==0: return x
        return x*(self.rng.random(x.shape)>=self.p)/(1-self.p)

class Attention(Module):
    def __init__(self,width,heads,rng):
        if width%heads: raise ValueError('width must be divisible by heads')
        self.heads=heads
        self.q=Linear(width,width,rng); self.k=Linear(width,width,rng)
        self.v=Linear(width,width,rng); self.out=Linear(width,width,rng)
    def forward(self,x):
        b,t,c=x.shape; h=self.heads; d=c//h
        q,k,v=[layer(x).reshape(b,t,h,d).transpose(0,2,1,3) for layer in (self.q,self.k,self.v)]
        scores=(q@k.transpose(0,1,3,2))/np.sqrt(d)
        mask=np.where(np.triu(np.ones((t,t)),1),-np.inf,0)
        y=(scores+mask).softmax()@v
        return self.out(y.transpose(0,2,1,3).reshape(b,t,c))

class Block(Module):
    def __init__(self,width,heads,rng):
        self.n1=LayerNorm(width); self.n2=LayerNorm(width)
        self.attn=Attention(width,heads,rng)
        self.up=Linear(width,4*width,rng); self.down=Linear(4*width,width,rng)
    def forward(self,x):
        x=x+self.attn(self.n1(x))
        return x+self.down(self.up(self.n2(x)).gelu())

class Transformer(Module):
    def __init__(self,vocab=8,context=16,width=16,heads=2,layers=1,seed=42):
        rng=np.random.default_rng(seed)
        self.context=context
        self.token=Embedding(vocab,width,rng); self.position=Embedding(context,width,rng)
        self.blocks=[Block(width,heads,rng) for _ in range(layers)]
        self.norm=LayerNorm(width); self.head=Linear(width,vocab,rng)
    def forward(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if ids.ndim!=2 or not 0<ids.shape[1]<=self.context: raise ValueError('Expected [batch,time] within context')
        x=self.token(ids)+self.position(np.arange(ids.shape[1]))
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x))
