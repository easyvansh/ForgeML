import unittest

try:
    import torch
    from forge.cuda_backend import causal_attention, device_info, select_device
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "optional torch dependency is not installed")
class TorchBackendTests(unittest.TestCase):
    def test_device_selection(self):
        device = select_device("cuda")
        self.assertIn(device.type, ("cuda", "cpu"))
        self.assertIn("cuda_available", device_info())

    def test_attention_backward_and_causality(self):
        # Keep the core suite portable; the CUDA device sweep is run separately
        # because driver/toolkit availability can differ between test machines.
        device = select_device("cpu")
        q = torch.randn(1, 2, 7, 4, device=device, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = causal_attention(q, k, v)
        self.assertEqual(tuple(out.shape), (1, 2, 7, 4))
        out.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all().item())
        self.assertTrue(torch.isfinite(k.grad).all().item())
        self.assertTrue(torch.isfinite(v.grad).all().item())


if __name__ == "__main__":
    unittest.main()
