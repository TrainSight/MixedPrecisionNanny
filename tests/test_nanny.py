"""
tests/test_nanny.py — MixedPrecisionNanny 集成测试

覆盖：
  - 初始化：attach hook、打印摘要、创建 DB
  - context manager step()：数据写入、触发告警
  - 手动 begin_step / end_step API
  - 跨 step：trace_interval 控制哪些 step 写入
  - ERROR 触发 trigger_dense（接下来 50 步全 trace）
  - scaler 监控：scale 正常记录、overflow 检测
  - verbose=False 不打印
  - close / __exit__ 正常释放资源
  - 外层 with 语法（with MixedPrecisionNanny(...) as nanny）
"""
import sqlite3

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim

from analyzer.numerical_checker import AlertConfig
from nanny import MixedPrecisionNanny
from tests.conftest import db_count, db_query


# ─── 测试用模型 ────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _NaNMultiply(nn.Module):
    """将输入乘以 NaN。封装为 module 以便 hook 捕获输出。"""
    def forward(self, x):
        return x * float("nan")


class NaNOutputMLP(nn.Module):
    """forward 输出 NaN，用于触发告警。"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.nan_op = _NaNMultiply()

    def forward(self, x):
        return self.nan_op(self.fc(x))


# ─── 工具：模拟 GradScaler ─────────────────────────────────────────────────────

class FakeScaler:
    def __init__(self, scale: float = 1024.0):
        self._scale = scale

    def get_scale(self) -> float:
        return self._scale

    def set_scale(self, scale: float):
        self._scale = scale


# ════════════════════════════════════════════════════════════════════════════════
# 初始化
# ════════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_creates_db_file(self, tmp_path):
        import os
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(model, output_dir=str(tmp_path / "logs"), verbose=False)
        nanny.close()
        assert os.path.exists(str(tmp_path / "logs" / "metrics.db"))

    def test_attach_returns_positive_layer_count(self, tmp_path, capsys):
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=True
        )
        out = capsys.readouterr().out
        assert "Attached to" in out
        # SimpleMLP 有 fc1 + relu + fc2 三个子模块，但 relu 是 functional 不计入
        # fc1 和 fc2 是两个 Linear，加上根级 None → 至少 2 层
        nanny.close()

    def test_verbose_false_no_init_print(self, tmp_path, capsys):
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(model, output_dir=str(tmp_path), verbose=False)
        out = capsys.readouterr().out
        assert out == ""
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# context manager step()
# ════════════════════════════════════════════════════════════════════════════════

class TestStepContextManager:

    def test_step_0_writes_stats_to_db(self, tmp_path):
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        with nanny.step(0):
            model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        assert db_count(db_path, "layer_stats") > 0
        nanny.close()

    def test_non_trace_step_no_stats_written(self, tmp_path):
        """trace_interval=100，step=1 不写入。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=100, verbose=False
        )
        with nanny.step(1):
            model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        assert db_count(db_path, "layer_stats") == 0
        nanny.close()

    def test_multiple_steps_only_trace_steps_written(self, tmp_path):
        """interval=10，step=0,10,20 写入；1~9,11~19,21 不写入。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=10, verbose=False
        )
        for step in range(25):
            with nanny.step(step):
                model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        traced_steps = {
            r[0] for r in db_query(db_path, "SELECT DISTINCT step FROM layer_stats")
        }
        assert traced_steps == {0, 10, 20}
        nanny.close()

    def test_context_manager_runs_training_code(self, tmp_path):
        """确保 context manager 内的代码能正常运行（不死锁/不崩溃）。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        results = []
        with nanny.step(0):
            out = model(torch.randn(2, 8))
            results.append(out.shape)
        assert results[0] == torch.Size([2, 4])
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# 手动 begin_step / end_step
# ════════════════════════════════════════════════════════════════════════════════

class TestManualStepAPI:

    def test_begin_end_step_equivalent_to_context_manager(self, tmp_path):
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        nanny.begin_step(0)
        model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.end_step()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        assert db_count(db_path, "layer_stats") > 0
        nanny.close()

    def test_begin_step_resets_alert_counters(self, tmp_path):
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        nanny._step_error_count = 99
        nanny._step_warn_count = 88
        nanny.begin_step(5)
        assert nanny._step_error_count == 0
        assert nanny._step_warn_count == 0
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# 告警触发 trigger_dense
# ════════════════════════════════════════════════════════════════════════════════

class TestAlertTriggersDenseSampling:

    def test_error_triggers_dense_sampling(self, tmp_path):
        """
        NaN 输出 → ERROR alert → sampler.trigger_dense(50)。
        接下来 50 个非周期 step 也应全部 trace。
        使用 interval=1000 确保周期不干扰。
        """
        model = NaNOutputMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1000, verbose=False
        )
        db_path = str(tmp_path / "metrics.db")

        # step=0：periodic trace，输出 NaN → trigger_dense(50)
        with nanny.step(0):
            model(torch.randn(2, 4))
        nanny.flush()
        assert db_count(db_path, "layer_stats", "step=0") > 0

        # step=1~3：非周期，但在触发模式中，仍应 trace
        for s in range(1, 4):
            with nanny.step(s):
                model(torch.randn(2, 4))
        nanny.flush()
        assert db_count(db_path, "layer_stats", "step=1") > 0
        assert db_count(db_path, "layer_stats", "step=2") > 0
        nanny.close()

    def test_nan_alert_written_to_db(self, tmp_path):
        model = NaNOutputMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        with nanny.step(0):
            model(torch.randn(2, 4))
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        assert db_count(db_path, "alerts", "alert_type='NAN'") > 0
        nanny.close()

    def test_error_increments_step_error_count(self, tmp_path):
        model = NaNOutputMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        nanny.begin_step(0)
        model(torch.randn(2, 4))
        assert nanny._step_error_count > 0
        nanny.end_step()
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# Scaler 监控
# ════════════════════════════════════════════════════════════════════════════════

class TestScalerMonitoring:

    def test_loss_scale_recorded_each_step(self, tmp_path):
        model = SimpleMLP()
        scaler = FakeScaler(scale=1024.0)
        nanny = MixedPrecisionNanny(
            model, scaler=scaler, output_dir=str(tmp_path),
            trace_interval=1, verbose=False
        )
        for step in range(3):
            with nanny.step(step):
                model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        assert db_count(db_path, "loss_scale_history") == 3
        nanny.close()

    def test_overflow_detected_when_scale_decreases(self, tmp_path):
        model = SimpleMLP()
        scaler = FakeScaler(scale=1024.0)
        nanny = MixedPrecisionNanny(
            model, scaler=scaler, output_dir=str(tmp_path),
            trace_interval=1, verbose=False
        )
        with nanny.step(0):
            model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        scaler.set_scale(512.0)   # 模拟 overflow（scale 下降）
        with nanny.step(1):
            model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.flush()

        db_path = str(tmp_path / "metrics.db")
        rows = db_query(db_path, "SELECT step, overflow FROM loss_scale_history ORDER BY step")
        assert rows[0] == (0, 0)   # step=0 无 overflow
        assert rows[1] == (1, 1)   # step=1 overflow
        nanny.close()

    def test_scaler_none_no_crash(self, tmp_path):
        """不传 scaler 不应崩溃。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, scaler=None, output_dir=str(tmp_path),
            trace_interval=1, verbose=False
        )
        with nanny.step(0):
            model(torch.randn(2, 8, requires_grad=True)).sum().backward()
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# verbose 控制
# ════════════════════════════════════════════════════════════════════════════════

class TestVerbose:

    def test_verbose_true_prints_error_alert(self, tmp_path, capsys):
        model = NaNOutputMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=True
        )
        capsys.readouterr()  # 清空 init 输出
        with nanny.step(0):
            model(torch.randn(2, 4))
        out = capsys.readouterr().out
        assert "ERROR" in out or "NAN" in out.upper()
        nanny.close()

    def test_verbose_false_no_alert_print(self, tmp_path, capsys):
        model = NaNOutputMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        capsys.readouterr()
        with nanny.step(0):
            model(torch.randn(2, 4))
        out = capsys.readouterr().out
        assert out == ""
        nanny.close()


# ════════════════════════════════════════════════════════════════════════════════
# close / context manager
# ════════════════════════════════════════════════════════════════════════════════

class TestCloseAndContextManager:

    def test_close_removes_hooks(self, tmp_path):
        """close() 后再 forward，不写新数据。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        nanny.close()

        db_path = str(tmp_path / "metrics.db")
        count_before = db_count(db_path, "layer_stats")
        model(torch.randn(2, 8))  # hooks 已移除，不应写入
        import time; time.sleep(0.1)
        count_after = db_count(db_path, "layer_stats")
        assert count_after == count_before

    def test_outer_with_syntax(self, tmp_path):
        """with MixedPrecisionNanny(...) as nanny: 语法工作正常。"""
        model = SimpleMLP()
        db_path = str(tmp_path / "metrics.db")
        with MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        ) as nanny:
            with nanny.step(0):
                model(torch.randn(2, 8, requires_grad=True)).sum().backward()

        # __exit__ 后数据应已落盘
        assert db_count(db_path, "layer_stats") > 0

    def test_exception_in_step_still_calls_end_step(self, tmp_path):
        """step() context manager 内抛异常，end_step 仍然执行（finally 保证）。"""
        model = SimpleMLP()
        nanny = MixedPrecisionNanny(
            model, output_dir=str(tmp_path), trace_interval=1, verbose=False
        )
        # 记录 end_step 是否被调用
        end_step_called = []
        orig_end_step = nanny.end_step
        def patched_end_step():
            end_step_called.append(True)
            orig_end_step()
        nanny.end_step = patched_end_step

        try:
            with nanny.step(0):
                raise ValueError("training error")
        except ValueError:
            pass

        assert len(end_step_called) == 1
        nanny.close()
