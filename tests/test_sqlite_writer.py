"""
tests/test_sqlite_writer.py — SQLiteWriter 单元测试

覆盖：
  - 初始化：自动创建 DB 文件和父目录、建表
  - write_stats / write_alert / write_loss_scale 写入后可查询
  - 批量写入多条记录
  - loss_scale 的 upsert（相同 step 覆盖）
  - overflow 标志正确写入
  - flush() 保证数据落盘后可查询
  - close() 不丢数据
  - 并发读（WAL 模式下 CLI 可同时读）
  - DB Schema：表和索引存在
"""
import sqlite3
import time

import pytest

from tests.conftest import db_count, db_query, make_stats
from storage.sqlite_writer import SQLiteWriter


# ════════════════════════════════════════════════════════════════════════════════
# 初始化
# ════════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        w = SQLiteWriter(db_path)
        w.close()
        import os
        assert os.path.exists(db_path)

    def test_creates_nested_parent_directories(self, tmp_path):
        db_path = str(tmp_path / "a" / "b" / "c" / "metrics.db")
        w = SQLiteWriter(db_path)
        w.close()
        import os
        assert os.path.exists(db_path)

    def test_schema_has_layer_stats_table(self, tmp_writer):
        _, db_path = tmp_writer
        rows = db_query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {r[0] for r in rows}
        assert "layer_stats" in table_names

    def test_schema_has_alerts_table(self, tmp_writer):
        _, db_path = tmp_writer
        rows = db_query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        assert "alerts" in {r[0] for r in rows}

    def test_schema_has_loss_scale_history_table(self, tmp_writer):
        _, db_path = tmp_writer
        rows = db_query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        assert "loss_scale_history" in {r[0] for r in rows}

    def test_schema_has_indexes(self, tmp_writer):
        _, db_path = tmp_writer
        rows = db_query(db_path, "SELECT name FROM sqlite_master WHERE type='index'")
        index_names = {r[0] for r in rows}
        assert "idx_stats_step" in index_names
        assert "idx_alerts_step" in index_names

    def test_wal_mode_enabled(self, tmp_writer):
        _, db_path = tmp_writer
        rows = db_query(db_path, "PRAGMA journal_mode")
        assert rows[0][0] == "wal"


# ════════════════════════════════════════════════════════════════════════════════
# write_stats
# ════════════════════════════════════════════════════════════════════════════════

class TestWriteStats:

    def test_write_single_stats_record(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_stats(1, "forward", "enc.0", "Linear", make_stats())
        writer.flush()
        assert db_count(db_path, "layer_stats") == 1

    def test_stats_fields_persisted_correctly(self, tmp_writer):
        writer, db_path = tmp_writer
        ts = time.time()
        stats = make_stats(
            dtype="torch.float16",
            nan_count=3,
            inf_count=1,
            max_val=2.5,
            min_nonzero=0.001,
            mean_val=0.5,
            std_val=1.2,
            p1=-1.0,
            p99=3.0,
            fp16_saturation=0.05,
            fp16_underflow=0.02,
        )
        writer.write_stats(42, "backward", "dec.proj", "Linear", stats, ts=ts)
        writer.flush()

        rows = db_query(
            db_path,
            "SELECT step, phase, layer_name, layer_type, dtype, nan_count, inf_count, "
            "max_val, fp16_sat, fp16_udf FROM layer_stats"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == 42            # step
        assert row[1] == "backward"    # phase
        assert row[2] == "dec.proj"    # layer_name
        assert row[3] == "Linear"      # layer_type
        assert row[4] == "torch.float16"
        assert row[5] == 3             # nan_count
        assert row[6] == 1             # inf_count
        assert abs(row[7] - 2.5) < 1e-6  # max_val
        assert abs(row[8] - 0.05) < 1e-6  # fp16_sat
        assert abs(row[9] - 0.02) < 1e-6  # fp16_udf

    def test_write_multiple_stats_records(self, tmp_writer):
        writer, db_path = tmp_writer
        for i in range(50):
            writer.write_stats(i, "forward", f"layer.{i}", "ReLU", make_stats())
        writer.flush()
        assert db_count(db_path, "layer_stats") == 50

    def test_forward_and_backward_separately(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_stats(1, "forward", "layer.0", "Linear", make_stats())
        writer.write_stats(1, "backward", "layer.0", "Linear", make_stats())
        writer.flush()
        assert db_count(db_path, "layer_stats", "phase='forward'") == 1
        assert db_count(db_path, "layer_stats", "phase='backward'") == 1


# ════════════════════════════════════════════════════════════════════════════════
# write_alert
# ════════════════════════════════════════════════════════════════════════════════

class TestWriteAlert:

    def test_write_single_alert(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_alert(10, "forward", "layer.0", "NAN", "ERROR", "test msg", 3.0)
        writer.flush()
        assert db_count(db_path, "alerts") == 1

    def test_alert_fields_persisted(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_alert(
            step=7, phase="backward", layer_name="enc.attn",
            alert_type="GRAD_EXPLOSION", severity="ERROR",
            message="grad too large", value=99999.9,
        )
        writer.flush()
        rows = db_query(
            db_path,
            "SELECT step, phase, layer_name, alert_type, severity, message, value "
            "FROM alerts"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == 7
        assert row[1] == "backward"
        assert row[2] == "enc.attn"
        assert row[3] == "GRAD_EXPLOSION"
        assert row[4] == "ERROR"
        assert row[5] == "grad too large"
        assert abs(row[6] - 99999.9) < 0.1

    def test_write_multiple_alerts(self, tmp_writer):
        writer, db_path = tmp_writer
        for i in range(30):
            writer.write_alert(i, "forward", f"l.{i}", "NAN", "ERROR", "msg", float(i))
        writer.flush()
        assert db_count(db_path, "alerts") == 30

    def test_severity_filter_works(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_alert(1, "f", "l", "NAN",      "ERROR",   "err",  1.0)
        writer.write_alert(2, "f", "l", "OVERFLOW", "WARNING", "warn", 0.5)
        writer.flush()
        assert db_count(db_path, "alerts", "severity='ERROR'")   == 1
        assert db_count(db_path, "alerts", "severity='WARNING'") == 1


# ════════════════════════════════════════════════════════════════════════════════
# write_loss_scale
# ════════════════════════════════════════════════════════════════════════════════

class TestWriteLossScale:

    def test_write_single_scale_record(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_loss_scale(1, 1024.0)
        writer.flush()
        assert db_count(db_path, "loss_scale_history") == 1

    def test_scale_and_overflow_fields_persisted(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_loss_scale(5, 512.0, overflow=True)
        writer.flush()
        rows = db_query(db_path, "SELECT step, scale, overflow FROM loss_scale_history")
        assert rows[0] == (5, 512.0, 1)

    def test_overflow_false_stored_as_0(self, tmp_writer):
        writer, db_path = tmp_writer
        writer.write_loss_scale(3, 2048.0, overflow=False)
        writer.flush()
        rows = db_query(db_path, "SELECT overflow FROM loss_scale_history")
        assert rows[0][0] == 0

    def test_same_step_upsert(self, tmp_writer):
        """相同 step 多次写入，后者覆盖（INSERT OR REPLACE）。"""
        writer, db_path = tmp_writer
        writer.write_loss_scale(10, 1024.0, overflow=False)
        writer.write_loss_scale(10, 512.0,  overflow=True)   # 覆盖
        writer.flush()
        rows = db_query(db_path, "SELECT scale, overflow FROM loss_scale_history WHERE step=10")
        assert len(rows) == 1
        assert rows[0] == (512.0, 1)


# ════════════════════════════════════════════════════════════════════════════════
# flush / close
# ════════════════════════════════════════════════════════════════════════════════

class TestFlushClose:

    def test_flush_ensures_data_visible(self, tmp_path):
        """flush() 后立即读取，数据应已落盘。"""
        db_path = str(tmp_path / "metrics.db")
        writer = SQLiteWriter(db_path)
        writer.write_stats(1, "forward", "l", "Linear", make_stats())
        writer.flush()
        assert db_count(db_path, "layer_stats") == 1
        writer.close()

    def test_close_does_not_lose_data(self, tmp_path):
        """close() 前未显式 flush，数据也不应丢失。"""
        db_path = str(tmp_path / "metrics.db")
        writer = SQLiteWriter(db_path)
        for i in range(10):
            writer.write_stats(i, "forward", f"l.{i}", "ReLU", make_stats())
        writer.close()   # 内部 flush 再关闭
        assert db_count(db_path, "layer_stats") == 10

    def test_batch_write_within_single_transaction(self, tmp_path):
        """批量写入 250 条（超过 _BATCH_SIZE=200），全部持久化。"""
        db_path = str(tmp_path / "metrics.db")
        writer = SQLiteWriter(db_path)
        for i in range(250):
            writer.write_stats(i, "forward", f"layer.{i}", "Linear", make_stats())
        writer.close()
        assert db_count(db_path, "layer_stats") == 250

    def test_mixed_writes_all_persisted(self, tmp_path):
        """stats + alerts + loss_scale 混合写入，全部落盘。"""
        db_path = str(tmp_path / "metrics.db")
        writer = SQLiteWriter(db_path)
        writer.write_stats(1, "forward", "l", "L", make_stats())
        writer.write_alert(1, "forward", "l", "NAN", "ERROR", "m", 1.0)
        writer.write_loss_scale(1, 1024.0)
        writer.close()
        assert db_count(db_path, "layer_stats") == 1
        assert db_count(db_path, "alerts") == 1
        assert db_count(db_path, "loss_scale_history") == 1


# ════════════════════════════════════════════════════════════════════════════════
# 并发读
# ════════════════════════════════════════════════════════════════════════════════

class TestConcurrentRead:

    def test_wal_allows_concurrent_read_during_write(self, tmp_path):
        """
        训练进程写入的同时，CLI 读进程应可打开 DB 而不被阻塞（WAL 模式）。
        此处用同线程模拟：写入部分数据后立即用新连接只读查询。
        """
        db_path = str(tmp_path / "metrics.db")
        writer = SQLiteWriter(db_path)

        # 写入部分数据
        for i in range(5):
            writer.write_stats(i, "forward", f"l.{i}", "L", make_stats())
        writer.flush()

        # 用独立的只读连接查询（模拟 CLI）
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            count = conn.execute("SELECT COUNT(*) FROM layer_stats").fetchone()[0]
            assert count == 5
        finally:
            conn.close()

        writer.close()
