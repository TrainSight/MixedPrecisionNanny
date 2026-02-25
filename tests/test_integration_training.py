"""
tests/test_integration_training.py — 端到端训练集成测试

每个测试都包含一段完整的训练代码（forward + backward + optimizer），
通过制造典型的混合精度问题，验证 MixedPrecisionNanny 能够正确检测。

测试场景：
  1. 基线正常训练                — 不产生任何告警
  2. FP16 数值上溢（overflow）   — 激活值 > 0.9 * FP16_MAX → OVERFLOW
  3. FP16 数值下溢（underflow）  — 激活值 < FP16_MIN_NORMAL（非零）→ UNDERFLOW
  4. NaN 前向传播               — log(负数) 产生 NaN → NAN
  5. Inf 前向传播               — 1/0 产生 Inf → INF
  6. 梯度爆炸                  — 大权重在反向传播中放大梯度 → GRAD_EXPLOSION
  7. 梯度消失                  — 极小权重导致梯度趋近于零 → GRAD_VANISH
  8. 大数吃小数（dim imbalance） — 大维度 reduce 导致精度丢失 + overflow → OVERFLOW
                               — 同时通过 FP32 vs FP16 对比数值展示精度丢失

每个测试的结构：
  - 创建有问题的模型
  - 用 nanny.step() 包裹训练步骤
  - 查询 metrics.db 断言期望告警类型已被记录
"""
from __future__ import annotations

import sqlite3

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim

from analyzer.numerical_checker import AlertConfig, FP16_MAX, FP16_MIN_NORMAL
from nanny import MixedPrecisionNanny
from tests.conftest import db_count, db_query


# ─────────────────────────────────────────────────────────────────────────────
# 公用工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_nanny(model, tmp_path, alert_config=None, **kwargs):
    """在 tmp_path 下创建 MixedPrecisionNanny（verbose=False 避免污染 pytest 输出）。"""
    return MixedPrecisionNanny(
        model,
        output_dir=str(tmp_path / "nanny_logs"),
        trace_interval=1,          # 每步都 trace，方便测试
        alert_config=alert_config,
        verbose=False,
        **kwargs,
    )


def _run_step(nanny, model, x, step=0, run_backward=True):
    """
    执行一次完整的训练 step（零梯度 → forward → backward → 返回输出）。
    若 backward 因 NaN/Inf 数值问题失败，安全跳过。
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    with nanny.step(step):
        out = model(x)
        if run_backward:
            # nansum 跳过 NaN；若 loss 仍非有限则跳过 backward
            loss = out.float().nansum()
            if loss.isfinite():
                loss.backward()
    return out


def _db_path(nanny):
    return str(nanny._output_dir) + "/metrics.db"


# ─────────────────────────────────────────────────────────────────────────────
# 辅助操作模块（封装为 nn.Module 使 hook 可以捕获其输出）
# ─────────────────────────────────────────────────────────────────────────────

class _NaNLogOp(nn.Module):
    """对输入取 log(负数)，产生 NaN 输出。"""
    def forward(self, x):
        neg_vals = x - x.detach().abs().max() - 1.0
        return torch.log(neg_vals)


class _InfInjectOp(nn.Module):
    """向第一个样本的所有 channel 注入 Inf。"""
    def forward(self, x):
        inf_mask = torch.zeros_like(x)
        inf_mask[0, :] = float('inf')
        return x + inf_mask


class _UnderflowScaleOp(nn.Module):
    """将输入乘以 1e-8，使值下溢至 FP16_MIN_NORMAL 以下。"""
    def forward(self, x):
        return x * 1e-8


class _ReduceSumOp(nn.Module):
    """在 dim=1 上做 sum，累积大量正值以触发 FP16 溢出。"""
    def forward(self, x):
        return x.sum(dim=1)


class _NegLogOp(nn.Module):
    """对 -|x|-1 取 log，产生 NaN 输出。"""
    def forward(self, x):
        return torch.log(-x.abs() - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 问题模型库
# ─────────────────────────────────────────────────────────────────────────────

class NormalMLP(nn.Module):
    """正常的小型 MLP，无数值问题。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        return self.net(x)


class FP16OverflowModel(nn.Module):
    """
    FP16 上溢：大权重使输出值超过 FP16_MAX * 0.9。
    Linear(16, 16) with weight=4500 → output = 16 * 4500 = 72000 > 65504
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(4500.0)

    def forward(self, x):
        return self.fc(x.abs() + 1.0)  # abs+1 确保所有输入为正


class FP16UnderflowModel(nn.Module):
    """
    FP16 下溢：将输出缩放到远低于 FP16_MIN_NORMAL（6.1e-5）。
    所有非零激活值都进入下溢区域。
    _UnderflowScaleOp 封装为 module，使 hook 可以捕获下溢后的输出。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.scaler = _UnderflowScaleOp()

    def forward(self, x):
        return self.scaler(self.fc(x))


class NaNProducingModel(nn.Module):
    """
    NaN 产生：log(负数) → NaN。
    _NaNLogOp 封装为 module，使 hook 可以捕获包含 NaN 的输出。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.nan_op = _NaNLogOp()

    def forward(self, x):
        return self.nan_op(self.fc(x))


class InfProducingModel(nn.Module):
    """
    Inf 产生：向激活值中注入 ±Inf。
    _InfInjectOp 封装为 module，使 hook 可以捕获含 Inf 的输出。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.inf_op = _InfInjectOp()

    def forward(self, x):
        return self.inf_op(self.fc(x))


class GradExplosionModel(nn.Module):
    """
    梯度爆炸：fc2 的大权重在反向传播时放大梯度。
    grad_fc1_output = grad_fc2_output @ fc2.weight
                    = ones(B,16) @ (1400 * ones(16,16))
                    ≈ B * 16 * 1400 / batch → 每元素 ≈ 22400 > 1e4
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(0.5)
            # 每个 fc2 列元素之和 = 16 * 1400 = 22400 > grad_explosion_threshold(1e4)
            self.fc2.weight.fill_(1400.0)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class GradVanishModel(nn.Module):
    """
    梯度消失：fc2 的极小权重导致梯度缩减到接近零。
    grad_fc1_output = ones(B,16) @ (1e-12 * ones(16,16))
                    ≈ 16 * 1e-12 = 1.6e-11 < 1e-8 (GRAD_VANISH threshold)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc2.weight.fill_(1e-12)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class DimImbalanceReduceModel(nn.Module):
    """
    大数吃小数（维度不平衡）：对 seq_len 维度做 sum 累积，
    导致结果值远超 FP16_MAX。

    单个激活值 = 16 inputs * weight (合理范围内)
    sum over seq_len=256: 256 * (16 * w) ≫ FP16_MAX

    _ReduceSumOp 封装为 module，使 hook 可以捕获溢出后的 reduce 结果。
    """
    def __init__(self, seq_len: int = 256, dim: int = 16, weight: float = 20.0):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            # 每个输出元素 ≈ dim * weight；sum over seq_len → seq_len * dim * weight
            # 256 * 16 * 20 = 81920 > FP16_MAX * 0.9 = 58953 → OVERFLOW
            self.fc.weight.fill_(weight)
        self.seq_len = seq_len
        self.reducer = _ReduceSumOp()

    def forward(self, x):
        # x: [batch, seq_len, dim]
        out = self.fc(x.abs() + 0.1)   # abs+0.1 确保全正，无符号抵消
        # sum over seq_len：累积大量正值，超出 FP16 范围
        return self.reducer(out)        # [batch, dim]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 基线：正常训练不产生告警
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalTraining:

    def test_no_alerts_in_normal_training(self, tmp_path):
        """正常 MLP 训练数个步骤后，不应产生任何告警。"""
        torch.manual_seed(42)
        model = NormalMLP()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        nanny = _make_nanny(model, tmp_path)

        for step in range(5):
            optimizer.zero_grad()
            x = torch.randn(4, 16)
            _run_step(nanny, model, x, step=step)
            optimizer.step()

        nanny.flush()
        db = _db_path(nanny)

        # 无任何告警
        assert db_count(db, "alerts") == 0, \
            "正常训练不应产生告警"
        # 但应有 stats 记录（每步都 trace）
        assert db_count(db, "layer_stats") > 0
        nanny.close()

    def test_stats_recorded_each_step(self, tmp_path):
        """每个 trace step 的各层统计数据应写入 DB。"""
        torch.manual_seed(0)
        model = NormalMLP()
        nanny = _make_nanny(model, tmp_path)

        for step in range(3):
            _run_step(nanny, model, torch.randn(2, 16), step=step)

        nanny.flush()
        db = _db_path(nanny)

        # 3 步 × 3 层 = 9 条 forward stats（最少）
        fwd_count = db_count(db, "layer_stats", "phase='forward'")
        assert fwd_count >= 9, f"expected ≥9 forward stats, got {fwd_count}"

        # 各层名称应在 DB 中
        names = {r[0] for r in db_query(db, "SELECT DISTINCT layer_name FROM layer_stats")}
        assert len(names) >= 3
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FP16 上溢（Overflow / Saturation）
# ─────────────────────────────────────────────────────────────────────────────

class TestFP16Overflow:

    def test_overflow_detected_as_overflow_alert(self, tmp_path):
        """
        模型输出值 ~72000 > FP16_MAX * 0.9（~58953）。
        应触发 OVERFLOW ERROR 告警。
        """
        torch.manual_seed(0)
        model = FP16OverflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.ones(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        overflow_count = db_count(db, "alerts", "alert_type='OVERFLOW'")
        assert overflow_count > 0, \
            "大权重导致的激活值上溢应触发 OVERFLOW 告警"
        nanny.close()

    def test_overflow_alert_severity_is_error_or_warning(self, tmp_path):
        """OVERFLOW 告警的 severity 应为 ERROR 或 WARNING（不应静默）。"""
        torch.manual_seed(0)
        model = FP16OverflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.ones(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT severity FROM alerts WHERE alert_type='OVERFLOW'")
        assert len(rows) > 0
        severities = {r[0] for r in rows}
        assert severities <= {"ERROR", "WARNING"}, \
            f"unexpected severity values: {severities}"
        nanny.close()

    def test_overflow_layer_name_recorded(self, tmp_path):
        """OVERFLOW 告警应记录产生问题的层名称。"""
        torch.manual_seed(0)
        model = FP16OverflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.ones(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT layer_name FROM alerts WHERE alert_type='OVERFLOW'")
        assert len(rows) > 0
        assert all(r[0] is not None and r[0] != "" for r in rows)
        nanny.close()

    def test_fp16_saturation_metric_in_stats(self, tmp_path):
        """layer_stats 中的 fp16_sat 字段应反映溢出比例（接近 1.0）。"""
        torch.manual_seed(0)
        model = FP16OverflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.ones(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT fp16_sat FROM layer_stats WHERE phase='forward'")
        # 产生溢出的层 fp16_sat 应大于告警阈值 0.01
        high_sat = [r[0] for r in rows if r[0] is not None and r[0] > 0.01]
        assert len(high_sat) > 0, \
            "应至少有一层的 fp16_sat 超过告警阈值"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. FP16 下溢（Underflow）
# ─────────────────────────────────────────────────────────────────────────────

class TestFP16Underflow:

    def test_underflow_detected(self, tmp_path):
        """
        将激活值缩减到 ~1e-8 << FP16_MIN_NORMAL（6.1e-5）。
        应触发 UNDERFLOW WARNING 告警。
        """
        torch.manual_seed(0)
        model = FP16UnderflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        underflow_count = db_count(db, "alerts", "alert_type='UNDERFLOW'")
        assert underflow_count > 0, \
            "激活值极小应触发 UNDERFLOW 告警"
        nanny.close()

    def test_underflow_severity_is_warning(self, tmp_path):
        """UNDERFLOW 告警的 severity 应为 WARNING（可接受但值得注意）。"""
        torch.manual_seed(0)
        model = FP16UnderflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT severity FROM alerts WHERE alert_type='UNDERFLOW'")
        assert len(rows) > 0
        for sev, in rows:
            assert sev == "WARNING", f"UNDERFLOW severity should be WARNING, got {sev}"
        nanny.close()

    def test_fp16_underflow_metric_in_stats(self, tmp_path):
        """layer_stats 中的 fp16_udf 字段应反映下溢比例（接近 1.0）。"""
        torch.manual_seed(0)
        model = FP16UnderflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT fp16_udf FROM layer_stats WHERE phase='forward'")
        high_udf = [r[0] for r in rows if r[0] is not None and r[0] > 0.5]
        assert len(high_udf) > 0, \
            "FP16 下溢层的 fp16_udf 应接近 1.0"
        nanny.close()

    def test_underflow_distinct_from_overflow(self, tmp_path):
        """下溢和上溢应该分别独立触发，不相互干扰。"""
        torch.manual_seed(0)
        model = FP16UnderflowModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        assert db_count(db, "alerts", "alert_type='UNDERFLOW'") > 0
        # 下溢场景不应触发 OVERFLOW（值太小，不是太大）
        assert db_count(db, "alerts", "alert_type='OVERFLOW'") == 0
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. NaN 传播
# ─────────────────────────────────────────────────────────────────────────────

class TestNaNPropagation:

    def test_nan_in_forward_detected(self, tmp_path):
        """
        log(负数) 产生 NaN，Nanny 应立即检测到并触发 NAN ERROR 告警。
        """
        torch.manual_seed(0)
        model = NaNProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        nan_count = db_count(db, "alerts", "alert_type='NAN'")
        assert nan_count > 0, \
            "log(负数) 产生的 NaN 应触发 NAN 告警"
        nanny.close()

    def test_nan_alert_severity_is_error(self, tmp_path):
        """NAN 告警的 severity 必须是 ERROR（非 WARNING）。"""
        torch.manual_seed(0)
        model = NaNProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT severity FROM alerts WHERE alert_type='NAN'")
        assert len(rows) > 0
        for sev, in rows:
            assert sev == "ERROR", f"NAN severity must be ERROR, got {sev}"
        nanny.close()

    def test_nan_triggers_dense_sampling(self, tmp_path):
        """
        检测到 NAN ERROR 后，Sampler 应切换到触发模式，
        后续步骤也会被 trace（即使不是周期步）。
        """
        torch.manual_seed(0)
        model = NaNProducingModel()
        # 使用大 interval，使非触发 step 不被 trace
        nanny = MixedPrecisionNanny(
            model,
            output_dir=str(tmp_path / "nanny_logs"),
            trace_interval=1000,
            verbose=False,
        )

        # step=0：周期 trace，触发 NAN → trigger_dense(50)
        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)
        assert db_count(db, "alerts", "step=0 AND alert_type='NAN'") > 0

        # step=1~3：非周期，但在触发模式内，也应写入 stats
        for s in range(1, 4):
            _run_step(nanny, model, torch.randn(4, 16), step=s, run_backward=False)
        nanny.flush()

        for s in range(1, 4):
            cnt = db_count(db, "layer_stats", f"step={s}")
            assert cnt > 0, \
                f"NAN 触发密集采样后，step={s} 也应有 stats 记录"
        nanny.close()

    def test_nan_count_in_stats(self, tmp_path):
        """layer_stats 中的 nan_count 字段应 > 0。"""
        torch.manual_seed(0)
        model = NaNProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT nan_count FROM layer_stats WHERE phase='forward'")
        nonzero_nan = [r[0] for r in rows if r[0] and r[0] > 0]
        assert len(nonzero_nan) > 0, \
            "产生 NaN 的层 stats 中 nan_count 应 > 0"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inf 传播
# ─────────────────────────────────────────────────────────────────────────────

class TestInfDetection:

    def test_inf_in_forward_detected(self, tmp_path):
        """
        向激活值注入 Inf，Nanny 应检测到并触发 INF ERROR 告警。
        """
        torch.manual_seed(0)
        model = InfProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        inf_count = db_count(db, "alerts", "alert_type='INF'")
        assert inf_count > 0, \
            "Inf 激活值应触发 INF 告警"
        nanny.close()

    def test_inf_alert_severity_is_error(self, tmp_path):
        """INF 告警的 severity 必须是 ERROR。"""
        torch.manual_seed(0)
        model = InfProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT severity FROM alerts WHERE alert_type='INF'")
        assert len(rows) > 0
        for sev, in rows:
            assert sev == "ERROR", f"INF severity must be ERROR, got {sev}"
        nanny.close()

    def test_inf_count_in_stats(self, tmp_path):
        """layer_stats 的 inf_count 字段应 > 0。"""
        torch.manual_seed(0)
        model = InfProducingModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(4, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT inf_count FROM layer_stats WHERE phase='forward'")
        nonzero_inf = [r[0] for r in rows if r[0] and r[0] > 0]
        assert len(nonzero_inf) > 0, \
            "产生 Inf 的层 stats 中 inf_count 应 > 0"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. 梯度爆炸
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientExplosion:

    def test_grad_explosion_detected_in_backward(self, tmp_path):
        """
        fc2 权重 = 1400，反向传播到 fc1 的梯度：
          grad_fc1_output[i] = Σ_j grad_fc2_output[j] * W2[j,i]
                             = 16 * 1400 = 22400 > 1e4 (GRAD_EXPLOSION 阈值)
        应触发 GRAD_EXPLOSION ERROR 告警。
        """
        torch.manual_seed(0)
        model = GradExplosionModel()
        nanny = _make_nanny(model, tmp_path)

        x = torch.randn(2, 16)
        _run_step(nanny, model, x, step=0, run_backward=True)
        nanny.flush()
        db = _db_path(nanny)

        explosion_count = db_count(db, "alerts", "alert_type='GRAD_EXPLOSION'")
        assert explosion_count > 0, \
            "大权重导致的梯度爆炸应触发 GRAD_EXPLOSION 告警"
        nanny.close()

    def test_grad_explosion_alert_severity_is_error(self, tmp_path):
        torch.manual_seed(0)
        model = GradExplosionModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(
            db, "SELECT severity FROM alerts WHERE alert_type='GRAD_EXPLOSION'"
        )
        assert len(rows) > 0
        for sev, in rows:
            assert sev == "ERROR"
        nanny.close()

    def test_grad_explosion_only_in_backward_phase(self, tmp_path):
        """GRAD_EXPLOSION 告警只应出现在 backward 阶段，不在 forward。"""
        torch.manual_seed(0)
        model = GradExplosionModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        fwd_explosion = db_count(
            db, "alerts",
            "alert_type='GRAD_EXPLOSION' AND phase='forward'"
        )
        assert fwd_explosion == 0, \
            "GRAD_EXPLOSION 不应在 forward 阶段触发"
        nanny.close()

    def test_grad_explosion_max_val_in_alert(self, tmp_path):
        """告警记录的 value 字段应等于实际梯度最大值（> 1e4）。"""
        torch.manual_seed(0)
        model = GradExplosionModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(
            db, "SELECT value FROM alerts WHERE alert_type='GRAD_EXPLOSION'"
        )
        assert len(rows) > 0
        for val, in rows:
            assert val > 1e4, \
                f"GRAD_EXPLOSION 告警的 value 应 > 1e4，实际 = {val:.2e}"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. 梯度消失
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientVanish:

    def test_grad_vanish_detected_in_backward(self, tmp_path):
        """
        fc2 权重 = 1e-12，反向传播到 fc1 的梯度：
          grad_fc1_output[i] = 16 * 1e-12 = 1.6e-11 < 1e-8 (GRAD_VANISH 阈值)
        且梯度非零，应触发 GRAD_VANISH WARNING 告警。
        """
        torch.manual_seed(0)
        model = GradVanishModel()
        nanny = _make_nanny(model, tmp_path)

        x = torch.randn(2, 16)
        _run_step(nanny, model, x, step=0, run_backward=True)
        nanny.flush()
        db = _db_path(nanny)

        vanish_count = db_count(db, "alerts", "alert_type='GRAD_VANISH'")
        assert vanish_count > 0, \
            "极小权重导致的梯度消失应触发 GRAD_VANISH 告警"
        nanny.close()

    def test_grad_vanish_severity_is_warning(self, tmp_path):
        """GRAD_VANISH 的 severity 应为 WARNING（比 ERROR 轻）。"""
        torch.manual_seed(0)
        model = GradVanishModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.randn(2, 16), step=0)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT severity FROM alerts WHERE alert_type='GRAD_VANISH'")
        assert len(rows) > 0
        for sev, in rows:
            assert sev == "WARNING", f"GRAD_VANISH severity should be WARNING, got {sev}"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. 大数吃小数（维度不平衡 / Dimensional Imbalance）
# ─────────────────────────────────────────────────────────────────────────────

class TestLargeNumberEatsSmallNumber:

    def test_fp16_precision_loss_numerical_demonstration(self):
        """
        数值精度丢失演示（不涉及 Nanny）：

        对混合了大数（100.0）和小数（0.01）的向量做 sum：
          FP32 精确结果 = n_large * 100 + n_small * 0.01
          FP16 结果：在 ~51200 这个量级，FP16 的最小精度间隔 ≈ 32，
                    远大于 0.01，小数值完全被"吃掉"。

        这说明在 FP16 下，大维度 reduce 操作会丢失小数值的贡献，
        即"大数吃小数"现象。
        """
        n = 1024
        large_val = 100.0   # 512 * 100 = 51200 (在 FP16 范围内，但精度受限)
        small_val = 0.01    # FP16 在 51200 处的精度间隔 ≈ 32，远大于 0.01

        x = torch.cat([
            torch.full((n // 2,), large_val),
            torch.full((n // 2,), small_val),
        ])

        sum_fp32 = x.sum().item()
        sum_fp16 = x.half().sum().float().item()

        # 预期的小数贡献
        expected_small = (n // 2) * small_val   # = 5.12

        # FP16 sum 丢失了（或严重低估了）小数贡献
        actual_small_in_fp16 = abs(sum_fp16 - (n // 2) * large_val)
        relative_error = abs(sum_fp16 - sum_fp32) / abs(sum_fp32)

        # FP32 下小数值有贡献
        assert abs(sum_fp32 - (n // 2) * large_val) > expected_small * 0.99, \
            "FP32 应该保留小数贡献"

        # FP16 下存在明显精度丢失（relative error > 1e-5）
        assert relative_error > 1e-5, \
            f"FP16 应存在精度丢失，relative_error={relative_error:.2e}"

        # FP16 丢失的小数贡献比 FP32 少（精度损失）
        assert actual_small_in_fp16 < expected_small * 0.1 or relative_error > 1e-4, \
            "FP16 在大数主导下应丢失小数的精度"

    def test_dim_imbalance_overflow_detected_by_nanny(self, tmp_path):
        """
        大维度 reduce sum 导致累积溢出：
          每个激活值 = 16 inputs * 20 weight = 320
          sum over seq_len=256: 256 * 320 = 81920 > FP16_MAX * 0.9 = 58953
        Nanny 应检测到 OVERFLOW 告警。
        """
        torch.manual_seed(0)
        model = DimImbalanceReduceModel(seq_len=256, dim=16, weight=20.0)
        nanny = _make_nanny(model, tmp_path)

        # 输入：[batch=2, seq_len=256, dim=16]，全 1 值
        x = torch.ones(2, 256, 16)
        _run_step(nanny, model, x, step=0, run_backward=True)
        nanny.flush()
        db = _db_path(nanny)

        overflow_count = db_count(db, "alerts", "alert_type='OVERFLOW'")
        assert overflow_count > 0, \
            "大维度 reduce 导致的激活值溢出应触发 OVERFLOW 告警"
        nanny.close()

    def test_dim_imbalance_fp16_saturation_metric(self, tmp_path):
        """
        大数吃小数场景下，layer_stats 中的 fp16_sat 应接近 1.0。
        """
        torch.manual_seed(0)
        model = DimImbalanceReduceModel(seq_len=256, dim=16, weight=20.0)
        nanny = _make_nanny(model, tmp_path)

        x = torch.ones(2, 256, 16)
        _run_step(nanny, model, x, step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        rows = db_query(db, "SELECT fp16_sat FROM layer_stats WHERE phase='forward'")
        high_sat = [r[0] for r in rows if r[0] is not None and r[0] > 0.5]
        assert len(high_sat) > 0, \
            "大维度 reduce 后的输出层 fp16_sat 应 > 0.5"
        nanny.close()

    def test_large_number_eats_small_in_mixed_scale_input(self, tmp_path):
        """
        混合尺度输入（大值 + 小值）经过 large reduce 后：
          大数主导累加结果，小数贡献被精度截断。
        Nanny 在此场景下应捕获到 OVERFLOW（结果值超出 FP16 范围）。
        """
        torch.manual_seed(0)
        seq_len = 512
        dim = 16
        model = DimImbalanceReduceModel(seq_len=seq_len, dim=dim, weight=15.0)
        nanny = _make_nanny(model, tmp_path)

        # 构造混合尺度输入：一半为大值，一半为极小值
        x_large = torch.ones(1, seq_len // 2, dim) * 2.0    # 大值：2.0
        x_small = torch.ones(1, seq_len // 2, dim) * 1e-6   # 小值：1e-6
        x = torch.cat([x_large, x_small], dim=1)             # [1, 512, 16]

        _run_step(nanny, model, x, step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        # 大值部分：256 * 16 * 15 * 2.0 = 122880 >> FP16_MAX → OVERFLOW
        overflow_count = db_count(db, "alerts", "alert_type='OVERFLOW'")
        assert overflow_count > 0, \
            "混合尺度输入中的大数累积应触发 OVERFLOW"
        nanny.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9. 综合场景：多种问题同时出现
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedIssues:

    def test_overflow_and_nan_in_same_training(self, tmp_path):
        """
        两层网络：第一层产生溢出值，第二层对溢出值取 log → NaN。
        Nanny 应同时检测到 OVERFLOW 和 NAN。
        """
        torch.manual_seed(0)

        class OverflowThenNaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 16, bias=False)
                self.fc2 = nn.Linear(16, 16, bias=False)
                self.nan_op = _NegLogOp()   # 封装为 module 让 hook 捕获 NaN 输出
                with torch.no_grad():
                    self.fc1.weight.fill_(4500.0)   # fc1 产生溢出值 ~72000
                    self.fc2.weight.fill_(0.001)

            def forward(self, x):
                out1 = self.fc1(x.abs() + 1.0)   # ~72000 → OVERFLOW (hook 捕获)
                out2 = self.fc2(out1 * 1e-8)
                return self.nan_op(out2)          # log(-|out2|-1) = NaN (hook 捕获)

        model = OverflowThenNaNModel()
        nanny = _make_nanny(model, tmp_path)

        _run_step(nanny, model, torch.ones(2, 16), step=0, run_backward=False)
        nanny.flush()
        db = _db_path(nanny)

        alert_types = {
            r[0] for r in db_query(db, "SELECT DISTINCT alert_type FROM alerts")
        }
        assert "OVERFLOW" in alert_types, "应检测到 OVERFLOW"
        assert "NAN" in alert_types, "应检测到 NAN"
        nanny.close()

    def test_multiple_steps_progressive_degradation(self, tmp_path):
        """
        模拟渐进式精度退化：
          - 随着训练，权重被更新到越来越大的值（通过手动增大权重模拟）
          - 前期步骤无告警，后期步骤产生 OVERFLOW
        验证 Nanny 能在正确的步骤发出告警。
        """
        torch.manual_seed(0)

        class DegradingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16, bias=False)
                with torch.no_grad():
                    self.fc.weight.fill_(1.0)

            def forward(self, x):
                return self.fc(x)

        model = DegradingModel()
        nanny = _make_nanny(model, tmp_path)

        # step 0-2：正常权重（1.0），无溢出
        for s in range(3):
            _run_step(nanny, model, torch.randn(2, 16), step=s)
        nanny.flush()
        db = _db_path(nanny)
        assert db_count(db, "alerts", "step < 3") == 0, \
            "正常权重阶段不应有告警"

        # 模拟权重爆炸（手动修改权重）
        with torch.no_grad():
            model.fc.weight.fill_(5000.0)

        # step 3：权重变大后，应触发 OVERFLOW
        _run_step(nanny, model, torch.ones(2, 16), step=3)
        nanny.flush()
        assert db_count(db, "alerts", "step=3 AND alert_type='OVERFLOW'") > 0, \
            "权重更新后应检测到 OVERFLOW"
        nanny.close()
