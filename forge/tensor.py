"""NumPy storage, independently implemented reverse-mode vector-Jacobian products."""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np

_record = ContextVar('record', default=True)

@contextmanager
def no_grad():
    token = _record.set(False)
    try:
        yield
    finally:
        _record.reset(token)

def unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = ()
        self.vjp = lambda g: ()

    @property
    def shape(self): return self.data.shape

    @staticmethod
    def wrap(x): return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def op(data, parents, vjp):
        out = Tensor(data, _record.get() and any(p.requires_grad for p in parents))
        if out.requires_grad:
            out.parents, out.vjp = parents, vjp
        return out

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise ValueError('Tensor does not require gradients')
        if gradient is None:
            if self.data.size != 1: raise ValueError('Non-scalar output needs a seed gradient')
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=self.data.dtype)
        if gradient.shape != self.shape: raise ValueError('Seed gradient shape mismatch')
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node.parents)
        grads = {self: gradient}
        for node in reversed(order):
            g = grads.get(node)
            if g is None: continue
            if not node.parents:
                node.grad = g.copy() if node.grad is None else node.grad + g
            for parent, value in zip(node.parents, node.vjp(g)):
                if parent.requires_grad:
                    grads[parent] = grads.get(parent, 0) + value

    def __add__(self, other):
        b = Tensor.wrap(other)
        return Tensor.op(self.data+b.data, (self,b), lambda g: (unbroadcast(g,self.shape),unbroadcast(g,b.shape)))
    __radd__ = __add__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + -Tensor.wrap(other)
    def __rsub__(self, other): return Tensor.wrap(other) + -self
    def __mul__(self, other):
        b = Tensor.wrap(other)
        return Tensor.op(self.data*b.data,(self,b),lambda g:(unbroadcast(g*b.data,self.shape),unbroadcast(g*self.data,b.shape)))
    __rmul__ = __mul__
    def __pow__(self, power):
        return Tensor.op(self.data**power,(self,),lambda g:(g*power*self.data**(power-1),))
    def __truediv__(self, other): return self * Tensor.wrap(other)**-1
    def __rtruediv__(self, other): return Tensor.wrap(other) * self**-1
    def __matmul__(self, other):
        b = Tensor.wrap(other)
        if self.data.ndim < 2 or b.data.ndim < 2:
            raise ValueError('matmul requires rank >= 2; reshape vectors explicitly')
        return Tensor.op(self.data@b.data,(self,b),lambda g:(unbroadcast(g@b.data.swapaxes(-1,-2),self.shape),unbroadcast(self.data.swapaxes(-1,-2)@g,b.shape)))
    def sum(self, axis=None, keepdims=False):
        axes = tuple(range(self.data.ndim)) if axis is None else ((axis,) if isinstance(axis,int) else tuple(axis))
        axes = tuple(a % self.data.ndim for a in axes)
        def backward(g):
            if not keepdims:
                for a in sorted(axes): g=np.expand_dims(g,a)
            return (np.broadcast_to(g,self.shape),)
        return Tensor.op(self.data.sum(axis=axis,keepdims=keepdims),(self,),backward)
    def mean(self, axis=None, keepdims=False):
        axes = tuple(range(self.data.ndim)) if axis is None else ((axis,) if isinstance(axis,int) else axis)
        return self.sum(axis,keepdims)/np.prod([self.shape[a] for a in axes])
    def reshape(self,*shape):
        return Tensor.op(self.data.reshape(*shape),(self,),lambda g:(g.reshape(self.shape),))
    def transpose(self,*axes):
        axes = axes or tuple(reversed(range(self.data.ndim)))
        return Tensor.op(self.data.transpose(axes),(self,),lambda g:(g.transpose(np.argsort(axes)),))
    def __getitem__(self,index):
        def backward(g):
            result=np.zeros_like(self.data)
            np.add.at(result,index,g)
            return (result,)
        return Tensor.op(self.data[index],(self,),backward)
    def exp(self):
        out=np.exp(self.data)
        return Tensor.op(out,(self,),lambda g:(g*out,))
    def log(self): return Tensor.op(np.log(self.data),(self,),lambda g:(g/self.data,))
    def tanh(self):
        out=np.tanh(self.data)
        return Tensor.op(out,(self,),lambda g:(g*(1-out*out),))
    def maximum(self,other):
        b=Tensor.wrap(other)
        left=(self.data>b.data)+0.5*(self.data==b.data)
        return Tensor.op(np.maximum(self.data,b.data),(self,b),lambda g:(unbroadcast(g*left,self.shape),unbroadcast(g*(1-left),b.shape)))
    def relu(self): return self.maximum(0)
    def gelu(self): return 0.5*self*(1+(np.sqrt(2/np.pi)*(self+0.044715*self**3)).tanh())
    def softmax(self,axis=-1):
        z=self-Tensor(self.data.max(axis=axis,keepdims=True))
        e=z.exp()
        return e/e.sum(axis=axis,keepdims=True)

def cross_entropy(logits, targets):
    targets=np.asarray(targets,dtype=np.int64).reshape(-1)
    x=logits.reshape(-1,logits.shape[-1])
    if targets.size != x.shape[0] or np.any(targets<0) or np.any(targets>=x.shape[1]):
        raise ValueError('Invalid targets')
    z=x-Tensor(x.data.max(axis=-1,keepdims=True))
    return (z.exp().sum(axis=-1).log()-z[np.arange(len(targets)),targets]).mean()
