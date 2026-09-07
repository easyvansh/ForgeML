import unittest
import numpy as np
from forge.tensor import Tensor,cross_entropy,no_grad
from forge.gradcheck import gradcheck
from forge.nn import Transformer,LayerNorm,Dropout
from forge.optim import SGD,Adam,AdamW
from forge.attention import dense_attention,streaming_attention

class RuntimeTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(7)
    def test_elementwise_gradients(self):
        for fn in [lambda x:(x*x+x/2-x**2/4).mean(),lambda x:x.exp().log().sum(),lambda x:x.tanh().sum(),lambda x:x.gelu().sum(),lambda x:x.relu().sum(),lambda x:x.maximum(0.3).sum()]:
            with self.subTest(fn=fn): gradcheck(fn,[self.rng.normal(size=(2,3))])
    def test_broadcast(self): gradcheck(lambda a,b:((a*b+a)/(b*b+1)).sum(),[self.rng.normal(size=(2,3,4)),self.rng.normal(size=(1,4))])
    def test_matmul(self): gradcheck(lambda a,b:(a@b).sum(),[self.rng.normal(size=(2,2,3)),self.rng.normal(size=(3,2))])
    def test_reshape_transpose(self): gradcheck(lambda x:x.reshape(3,2).transpose(1,0).sum(),[self.rng.normal(size=(2,3))])
    def test_reductions(self): gradcheck(lambda x:(x.mean((0,2))*x.sum((0,2))).sum(),[self.rng.normal(size=(2,3,2))])
    def test_repeated_index(self): gradcheck(lambda x:x[np.array([1,1,0])].sum(),[self.rng.normal(size=(3,2))])
    def test_softmax(self): gradcheck(lambda x:(x.softmax()*np.arange(3)).sum(),[self.rng.normal(size=(2,3))])
    def test_cross_entropy(self): gradcheck(lambda x:cross_entropy(x,np.array([0,2])),[self.rng.normal(size=(2,3))])
    def test_extreme_logits(self):
        x=Tensor([[1000,-1000]],True); loss=cross_entropy(x,[1]); loss.backward()
        self.assertAlmostEqual(float(loss.data),2000); np.testing.assert_allclose(x.grad,[[1,-1]])
    def test_accumulation(self):
        x=Tensor([2.],True); y=(x*x+x).sum(); y.backward(); y.backward()
        np.testing.assert_allclose(x.grad,[10])
    def test_no_grad(self):
        x=Tensor([1.],True)
        with no_grad(): self.assertFalse((x*x).requires_grad)
        self.assertTrue((x*x).requires_grad)
    def test_seed_validation(self):
        with self.assertRaises(ValueError): Tensor([1,2],True).backward()
    def test_layernorm(self):
        layer=LayerNorm(3)
        gradcheck(lambda x: (layer(x)*np.arange(3)).sum(),[self.rng.normal(size=(2,3))])
    def test_dropout_eval(self):
        d=Dropout(.5,self.rng); d.eval(); x=Tensor(np.ones(100)); np.testing.assert_equal(d(x).data,x.data)
        d.train(); self.assertTrue(np.any(d(x).data==0))
    def test_streaming(self):
        for t in [1,7,17,33]:
            for tile in [1,8,32]:
                q,k,v=[self.rng.normal(size=(2,2,t,4)) for _ in range(3)]
                np.testing.assert_allclose(streaming_attention(q,k,v,tile),dense_attention(q,k,v),atol=1e-12)
    def test_causality(self):
        m=Transformer(); x=np.array([[0,1,2,3]]); y=x.copy(); y[0,3]=7
        np.testing.assert_allclose(m(x).data[:,:3],m(y).data[:,:3],atol=1e-12)
    def test_full_model_parameter_gradient(self):
        m=Transformer(vocab=3,context=2,width=4,heads=2,layers=1); ids=np.array([[0,1]])
        loss=cross_entropy(m(ids),[[1,2]]); loss.backward()
        for _,p in m.named_parameters():
            idx=tuple(0 for _ in p.shape); old=p.data[idx]; eps=1e-6
            p.data[idx]=old+eps; a=float(cross_entropy(m(ids),[[1,2]]).data)
            p.data[idx]=old-eps; b=float(cross_entropy(m(ids),[[1,2]]).data); p.data[idx]=old
            self.assertAlmostEqual(p.grad[idx],(a-b)/(2*eps),places=5)
    def test_optimizer_first_update(self):
        for cls in [Adam,AdamW]:
            p=Tensor([1.,-2.],True); p.grad=np.array([.2,-.4]); o=cls([p],lr=.01)
            o.step(); np.testing.assert_allclose(p.data,[.99,-1.99],atol=1e-8)
    def test_adamw_decay(self):
        p=Tensor([2.],True); p.grad=np.zeros(1); AdamW([p],lr=.1,weight_decay=.2).step()
        np.testing.assert_allclose(p.data,[1.96])
    def test_sgd(self):
        p=Tensor([1.],True); p.grad=np.array([2.]); o=SGD([p],lr=.1,momentum=.9); o.step(); o.step()
        np.testing.assert_allclose(p.data,[.42])
    def test_overfit(self):
        m=Transformer(vocab=4,context=4,width=8,heads=2,layers=1); o=AdamW(m.parameters(),lr=.02)
        ids=np.array([[0,1,2,3]]); targets=np.array([[1,2,3,0]])
        first=float(cross_entropy(m(ids),targets).data)
        for _ in range(60):
            m.zero_grad(); loss=cross_entropy(m(ids),targets); loss.backward(); o.step()
        self.assertLess(float(cross_entropy(m(ids),targets).data),first*.2)

if __name__=='__main__': unittest.main()
