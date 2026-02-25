"""
tests/test_sampler.py — Sampler 单元测试

覆盖：
  - 周期模式（PERIODIC）：正确步数触发
  - 触发模式（TRIGGERED）：trigger_dense(N) 精确 trace N 步
  - 两种模式的交互
  - 边界情况：interval=1、step=0、interval 无效
  - 属性与 repr
"""
import pytest
from tracer.sampler import Sampler


# ─── 辅助：模拟一次完整 step（advance → should_trace → consume） ────────────────

def do_step(sampler: Sampler, step: int) -> bool:
    """执行一次完整的 training step 并返回是否被 trace。"""
    sampler.advance(step)
    traced = sampler.should_trace()
    sampler.consume()
    return traced


# ════════════════════════════════════════════════════════════════════════════════
# 周期模式
# ════════════════════════════════════════════════════════════════════════════════

class TestPeriodicMode:

    def test_traces_at_multiples_of_interval(self):
        s = Sampler(trace_interval=10)
        for step in range(50):
            traced = do_step(s, step)
            expected = (step % 10 == 0)
            assert traced == expected, f"step={step}: expected trace={expected}, got {traced}"

    @pytest.mark.parametrize("interval", [1, 5, 50, 100])
    def test_various_intervals(self, interval):
        s = Sampler(trace_interval=interval)
        traces = [do_step(s, i) for i in range(interval * 3)]
        # 应该恰好在 0, interval, 2*interval 处 trace
        assert traces[0] is True
        assert traces[interval] is True
        assert traces[interval * 2] is True
        # 间隔内不 trace
        if interval > 1:
            assert traces[1] is False

    def test_step_zero_always_traced(self):
        """step=0 是任何 interval 的倍数，必须 trace。"""
        for interval in [1, 7, 100, 999]:
            s = Sampler(trace_interval=interval)
            s.advance(0)
            assert s.should_trace(), f"step=0 should always trace (interval={interval})"
            s.consume()

    def test_interval_1_traces_every_step(self):
        s = Sampler(trace_interval=1)
        for step in range(10):
            s.advance(step)
            assert s.should_trace(), f"step={step} should trace with interval=1"
            s.consume()

    def test_non_multiples_not_traced(self):
        s = Sampler(trace_interval=100)
        for step in [1, 2, 50, 99]:
            s.advance(step)
            assert not s.should_trace(), f"step={step} should NOT trace"
            s.consume()

    def test_invalid_interval_raises_value_error(self):
        with pytest.raises(ValueError, match="trace_interval"):
            Sampler(trace_interval=0)

    def test_invalid_negative_interval_raises(self):
        with pytest.raises(ValueError):
            Sampler(trace_interval=-5)


# ════════════════════════════════════════════════════════════════════════════════
# 触发模式
# ════════════════════════════════════════════════════════════════════════════════

class TestTriggeredMode:

    def test_trigger_dense_traces_exactly_n_steps(self):
        """
        trigger_dense(N) 后，紧接着的 N 个 step 全部 trace，第 N+1 个恢复周期模式。
        使用 interval=1000 确保周期模式不干扰。
        """
        s = Sampler(trace_interval=1000)
        # 在 step=3 触发（该 step 本身不是周期步，也不在触发模式）
        s.advance(3)
        assert not s.should_trace()
        s.trigger_dense(5)
        s.consume()

        # 接下来精确 5 步全部 trace
        for step in range(4, 9):
            s.advance(step)
            assert s.should_trace(), f"step={step} should be in triggered mode"
            s.consume()

        # 第 6 步恢复周期模式（1000 的倍数）
        s.advance(9)
        assert not s.should_trace()

    @pytest.mark.parametrize("n", [1, 3, 10, 50])
    def test_trigger_precise_count(self, n):
        """trigger_dense(N) 精确 trace N 步，不多不少。"""
        s = Sampler(trace_interval=9999)
        s.advance(1)
        s.trigger_dense(n)
        s.consume()

        traced_count = 0
        for step in range(2, 2 + n + 5):
            s.advance(step)
            if s.should_trace():
                traced_count += 1
            s.consume()

        assert traced_count == n, f"trigger_dense({n}) should trace exactly {n} steps, got {traced_count}"

    def test_trigger_does_not_shorten_existing_count(self):
        """已有较大剩余时，较小的 trigger_dense 不缩短。"""
        s = Sampler(trace_interval=9999)
        s.advance(0)
        s.trigger_dense(10)
        # consume() 本 step 已触发，跳过消耗 → remaining 仍为 10
        s.consume()

        s.advance(1)  # 新 step，_triggered_this_step 重置
        assert s._triggered_remaining == 10
        s.trigger_dense(3)  # max(10, 3) = 10，不变
        assert s._triggered_remaining == 10

    def test_trigger_extends_if_longer(self):
        """新的 trigger_dense 比剩余更大时，应该延长。"""
        s = Sampler(trace_interval=9999)
        s.advance(0)
        s.trigger_dense(3)
        # consume() 本 step 已触发，跳过消耗 → remaining 仍为 3
        s.consume()

        s.advance(1)  # remaining=3，_triggered_this_step=False
        assert s._triggered_remaining == 3
        s.trigger_dense(20)  # max(3, 20) = 20
        assert s._triggered_remaining == 20

    def test_returns_to_periodic_after_trigger_expires(self):
        """触发计数耗尽后，自动回到周期模式。"""
        s = Sampler(trace_interval=10)
        s.advance(1)
        s.trigger_dense(3)
        s.consume()

        # 在触发模式期间
        for step in range(2, 5):
            s.advance(step)
            assert s.should_trace()
            s.consume()

        # 回到周期模式：step=5 不是 10 的倍数
        s.advance(5)
        assert not s.should_trace()

        # step=10 恢复周期 trace
        s.advance(10)
        assert s.should_trace()

    def test_trigger_during_triggered_mode_resets_correctly(self):
        """在触发模式期间再次 trigger，取较大值。"""
        s = Sampler(trace_interval=9999)
        s.advance(0)
        s.trigger_dense(5)
        # consume() 本 step 已触发（_triggered_this_step=True），跳过消耗 → remaining=5
        s.consume()

        s.advance(1)  # 推进到新 step，_triggered_this_step 重置
        assert s._triggered_remaining == 5
        # 再次 trigger 更大的数
        s.trigger_dense(100)
        assert s._triggered_remaining == 100


# ════════════════════════════════════════════════════════════════════════════════
# advance/consume 分离语义
# ════════════════════════════════════════════════════════════════════════════════

class TestAdvanceConsumeSeparation:

    def test_advance_does_not_consume(self):
        """advance() 不改变 triggered_remaining。"""
        s = Sampler(trace_interval=100)
        s.advance(0)
        s.trigger_dense(5)
        remaining_before = s._triggered_remaining

        s.advance(1)  # 只推进 step，不消耗
        assert s._triggered_remaining == remaining_before

    def test_consume_decrements_once_per_call(self):
        """每个 step 调用一次 consume()，精确递减一次。"""
        s = Sampler(trace_interval=100)
        # 在 step=0 触发，consume() 因 _triggered_this_step=True 跳过 → remaining=10
        s.advance(0)
        s.trigger_dense(10)
        s.consume()
        assert s._triggered_remaining == 10

        # 从 step=1 开始，每 advance + consume 递减一次
        for i in range(1, 11):
            s.advance(i)
            s.consume()
            assert s._triggered_remaining == 10 - i

    def test_consume_is_noop_in_periodic_mode(self):
        """周期模式下 consume() 不改变状态。"""
        s = Sampler(trace_interval=10)
        s.advance(5)
        assert s._triggered_remaining == 0
        s.consume()  # 不应改变任何状态
        assert s._triggered_remaining == 0
        s.consume()
        assert s._triggered_remaining == 0

    def test_should_trace_idempotent_within_step(self):
        """同一 step 内多次调用 should_trace() 结果一致（hooks 会多次调用）。"""
        s = Sampler(trace_interval=10)
        s.advance(10)
        results = [s.should_trace() for _ in range(100)]
        assert all(r is True for r in results)

        s.consume()
        s.advance(11)
        results = [s.should_trace() for _ in range(100)]
        assert all(r is False for r in results)


# ════════════════════════════════════════════════════════════════════════════════
# 属性与 repr
# ════════════════════════════════════════════════════════════════════════════════

class TestProperties:

    def test_current_step_property(self):
        s = Sampler(trace_interval=10)
        assert s.current_step == 0
        s.advance(42)
        assert s.current_step == 42

    def test_is_triggered_false_initially(self):
        s = Sampler(trace_interval=10)
        assert not s.is_triggered

    def test_is_triggered_true_after_trigger_dense(self):
        s = Sampler(trace_interval=10)
        s.advance(0)
        s.trigger_dense(5)
        assert s.is_triggered

    def test_is_triggered_false_after_expiry(self):
        """trigger_dense(2)：接下来精确 2 个 step is_triggered=True，第 3 个为 False。"""
        s = Sampler(trace_interval=9999)
        s.advance(0)
        s.trigger_dense(2)
        s.consume()  # 本 step 触发，跳过消耗 → remaining=2

        s.advance(1)   # remaining=2
        assert s.is_triggered
        s.consume()    # remaining→1

        s.advance(2)   # remaining=1
        assert s.is_triggered
        s.consume()    # remaining→0

        s.advance(3)   # remaining=0
        assert not s.is_triggered

    def test_repr_contains_key_info(self):
        s = Sampler(trace_interval=50)
        r = repr(s)
        assert "Sampler" in r
        assert "50" in r          # interval
        assert "PERIODIC" in r

    def test_repr_shows_triggered_mode(self):
        s = Sampler(trace_interval=50)
        s.advance(0)
        s.trigger_dense(7)
        r = repr(s)
        assert "TRIGGERED" in r
        assert "7" in r           # remaining
