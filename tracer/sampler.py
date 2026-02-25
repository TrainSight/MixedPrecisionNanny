"""
采样策略控制器

决定当前 step 是否需要执行 trace，支持两种模式：
  - 周期模式（PERIODIC）：每 trace_interval 个 step trace 一次
  - 触发模式（TRIGGERED）：检测到异常后切换，连续 N 步都 trace

使用方：每个 step 开始时调用 advance(step)，
        hook 内部调用 should_trace() 决定是否采集。
"""
from __future__ import annotations


class Sampler:
    """
    线程安全假设：advance() 和 trigger_dense() 只在训练主线程调用，
    should_trace() 在 hook 内调用（同线程），因此无需加锁。
    """

    def __init__(self, trace_interval: int = 100):
        if trace_interval < 1:
            raise ValueError(f"trace_interval must be >= 1, got {trace_interval}")
        self.trace_interval = trace_interval
        self._current_step: int = 0
        self._triggered_remaining: int = 0   # 还剩几步处于触发模式
        self._triggered_this_step: bool = False  # 本 step 内是否刚调用过 trigger_dense

    # ─── 公共接口 ───────────────────────────────────────────────────────────────

    @property
    def current_step(self) -> int:
        return self._current_step

    def advance(self, step: int) -> None:
        """
        推进到新的 step。每个 training step 开始前调用一次。
        重置"本 step 触发"标记；不消耗触发计数（消耗由 consume() 负责）。
        """
        self._current_step = step
        self._triggered_this_step = False

    def consume(self) -> None:
        """
        消耗一次触发计数。每个 training step 结束后调用一次。

        若本 step 内调用过 trigger_dense()，则跳过消耗——这样 trigger_dense(N)
        精确保证"接下来 N 个 step 全部 trace"，而不是 N-1 个。
        """
        if self._triggered_this_step:
            return  # 本 step 刚设置了触发计数，不消耗
        if self._triggered_remaining > 0:
            self._triggered_remaining -= 1

    def should_trace(self) -> bool:
        """
        返回当前 step 是否应该 trace。
        在 hook 内部、begin_step 之后调用。
        """
        if self._triggered_remaining > 0:
            return True
        return self._current_step % self.trace_interval == 0

    def trigger_dense(self, duration: int = 50) -> None:
        """
        切换到触发模式，接下来 duration 个 step 全部 trace。
        若已在触发模式，取二者较大值（不缩短已剩余的计数）。
        同时设置"本 step 触发"标记，确保 consume() 不会消耗本 step 的计数。
        """
        self._triggered_remaining = max(self._triggered_remaining, duration)
        self._triggered_this_step = True

    @property
    def is_triggered(self) -> bool:
        return self._triggered_remaining > 0

    def __repr__(self) -> str:
        mode = f"TRIGGERED(remaining={self._triggered_remaining})" if self.is_triggered else "PERIODIC"
        return (
            f"Sampler(step={self._current_step}, interval={self.trace_interval}, mode={mode})"
        )
