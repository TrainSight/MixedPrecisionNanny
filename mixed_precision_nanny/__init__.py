"""
MixedPrecisionNanny — 混合精度训练保姆

混合精度训练（FP16/BF16）问题诊断与监控工具。

快速开始::

    from mixed_precision_nanny import MixedPrecisionNanny

    nanny = MixedPrecisionNanny(model, trace_interval=100)

    for step, (x, y) in enumerate(dataloader):
        with nanny.step(step):
            loss = criterion(model(x), y)
            loss.backward()
        optimizer.step()

    nanny.close()
"""

from .nanny import MixedPrecisionNanny
from .analyzer.numerical_checker import Alert, AlertConfig, TensorStats

__version__ = "0.1.0"
__all__ = [
    "MixedPrecisionNanny",
    "Alert",
    "AlertConfig",
    "TensorStats",
]
