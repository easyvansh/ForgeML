import numpy as np

def gradcheck(fn,arrays,eps=1e-6,atol=1e-6,rtol=1e-4):
    from .tensor import Tensor
    xs=[Tensor(np.array(a,copy=True),True) for a in arrays]
    fn(*xs).backward()
    errors=[]
    for x in xs:
        numerical=np.zeros_like(x.data)
        for idx in np.ndindex(x.shape):
            value=x.data[idx]
            x.data[idx]=value+eps; plus=float(fn(*xs).data)
            x.data[idx]=value-eps; minus=float(fn(*xs).data)
            x.data[idx]=value
            numerical[idx]=(plus-minus)/(2*eps)
        np.testing.assert_allclose(x.grad,numerical,atol=atol,rtol=rtol)
        errors.append(float(np.max(np.abs(x.grad-numerical))))
    return max(errors,default=0)
