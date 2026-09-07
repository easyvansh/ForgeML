import unittest
import numpy as np
from forge.tensor import Tensor
from forge.optim import Adam,AdamW,SGD
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'Optional PyTorch dependency is not installed')
class TorchParity(unittest.TestCase):
    def test_batched_matmul(self):
        rng=np.random.default_rng(42); a=rng.normal(size=(2,3,4)); b=rng.normal(size=(4,2))
        x,y=Tensor(a,True),Tensor(b,True); z=(x@y).softmax(); (z*z).sum().backward()
        tx=torch.tensor(a,requires_grad=True); ty=torch.tensor(b,requires_grad=True)
        tz=(tx@ty).softmax(-1); (tz*tz).sum().backward()
        np.testing.assert_allclose(x.grad,tx.grad.numpy(),atol=1e-10)
        np.testing.assert_allclose(y.grad,ty.grad.numpy(),atol=1e-10)
    def test_optimizers(self):
        for cls,ref,extra in [(SGD,torch.optim.SGD,{'momentum':.9}),(Adam,torch.optim.Adam,{}),(AdamW,torch.optim.AdamW,{})]:
            p=Tensor([1.,2.],True); q=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True)
            a=cls([p],lr=.01,weight_decay=.1,**extra); b=ref([q],lr=.01,weight_decay=.1,**extra)
            for i in range(5):
                g=np.array([.2+i*.03,-.5]); p.grad=g; q.grad=torch.tensor(g); a.step(); b.step()
            np.testing.assert_allclose(p.data,q.detach().numpy(),atol=1e-12)
