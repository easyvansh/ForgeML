import numpy as np

class Optimizer:
    def __init__(self,params,lr):
        self.params=list(params); self.lr=lr; self.t=0
    def zero_grad(self):
        for p in self.params: p.grad=None
    def diagnostics(self):
        return {'parameter_norm':float(np.sqrt(sum(np.sum(p.data**2) for p in self.params))),
                'gradient_norm':float(np.sqrt(sum(np.sum(p.grad**2) for p in self.params if p.grad is not None)))}

class SGD(Optimizer):
    def __init__(self,params,lr=0.01,momentum=0.9,weight_decay=0):
        super().__init__(params,lr); self.momentum=momentum; self.weight_decay=weight_decay
        self.velocity=[np.zeros_like(p.data) for p in self.params]
    def step(self):
        for p,v in zip(self.params,self.velocity):
            if p.grad is None: continue
            v *= self.momentum; v += p.grad+self.weight_decay*p.data
            p.data -= self.lr*v
        self.t+=1

class Adam(Optimizer):
    decoupled=False
    def __init__(self,params,lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0):
        super().__init__(params,lr)
        self.betas,self.eps,self.weight_decay=betas,eps,weight_decay
        self.m=[np.zeros_like(p.data) for p in self.params]; self.v=[np.zeros_like(p.data) for p in self.params]
        self.steps=[0]*len(self.params)
    def step(self):
        b1,b2=self.betas
        for i,(p,m,v) in enumerate(zip(self.params,self.m,self.v)):
            if p.grad is None: continue
            self.steps[i]+=1; t=self.steps[i]
            g=p.grad.copy()
            if self.decoupled: p.data *= 1-self.lr*self.weight_decay
            else: g += self.weight_decay*p.data
            m *= b1; m += (1-b1)*g
            v *= b2; v += (1-b2)*g*g
            p.data -= self.lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+self.eps)
        self.t+=1

class AdamW(Adam):
    decoupled=True
