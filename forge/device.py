"""Device-aware tensor helpers used by the optional PyTorch backend."""
from dataclasses import dataclass

from .cuda_backend import require_torch, select_device


@dataclass
class DeviceTensor:
    """Small ownership wrapper that makes placement explicit at API boundaries."""
    value: object

    @property
    def device(self):
        return self.value.device

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return tuple(self.value.shape)

    def to(self, device=None, *, dtype=None, non_blocking=False):
        target = select_device(device) if device is not None else self.device
        return DeviceTensor(self.value.to(target, dtype=dtype, non_blocking=non_blocking))

    def detach(self):
        return DeviceTensor(self.value.detach())


def as_device_tensor(value, device=None, *, dtype=None):
    t = require_torch()
    target = select_device(device)
    return DeviceTensor(t.as_tensor(value, device=target, dtype=dtype))
