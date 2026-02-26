"""
MixedPrecisionNanny CLI — 查询 metrics.db

子命令：
    summary  — 全局统计摘要
    alerts   — 查看告警列表
    stats    — 查看某 step 各层统计量

示例：
    mpnanny summary
    mpnanny alerts --severity ERROR --limit 20
    mpnanny alerts --step 1200
    mpnanny stats --step 100
    mpnanny stats --step 100 --phase backward --layer encoder.layer.0
    mpnanny --db ./my_run/metrics.db summary
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import Optional

# ANSI 颜色（若终端不支持可用 --no-color 关闭）
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_USE_COLOR = True


def _c(text: str, *codes: str) -> str:
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + _RESET


# ─── summary ──────────────────────────────────────────────────────────────────

def cmd_summary(conn: sqlite3.Connection, _args) -> None:
    c = conn.cursor()

    # 基本信息
    c.execute("SELECT MIN(step), MAX(step), COUNT(DISTINCT step) FROM layer_stats")
    row = c.fetchone()
    if row[0] is None:
        print("No trace data recorded yet.")
        return

    print(_c("=== MixedPrecisionNanny Summary ===", _BOLD))
    print(f"  Steps traced : {row[0]} → {row[1]}  ({row[2]} distinct steps)")

    c.execute("SELECT COUNT(DISTINCT layer_name) FROM layer_stats")
    print(f"  Layers       : {c.fetchone()[0]}")

    # 告警统计
    c.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity ORDER BY severity")
    rows = c.fetchall()
    print("\n  Alert summary:")
    if not rows:
        print("    " + _c("No alerts", _GREEN))
    for sev, cnt in rows:
        color = _RED if sev == "ERROR" else _YELLOW
        print(f"    {_c(sev, color):<20} {cnt}")

    # 最多告警的层（ERROR）
    c.execute("""
        SELECT layer_name, COUNT(*) AS cnt
        FROM   alerts
        WHERE  severity = 'ERROR'
        GROUP  BY layer_name
        ORDER  BY cnt DESC
        LIMIT  10
    """)
    rows = c.fetchall()
    if rows:
        print("\n  Top layers with ERROR:")
        for name, cnt in rows:
            print(f"    {name:<50} {cnt} alerts")

    # Loss Scale 历史（最近 10 条）
    c.execute("""
        SELECT step, scale, overflow
        FROM   loss_scale_history
        ORDER  BY step DESC
        LIMIT  10
    """)
    rows = c.fetchall()
    if rows:
        print("\n  Recent Loss Scale (latest first):")
        for step, scale, overflow in rows:
            flag = _c(" ← OVERFLOW", _RED) if overflow else ""
            print(f"    step={step:<8} scale={scale:<10.1f}{flag}")


# ─── alerts ───────────────────────────────────────────────────────────────────

def cmd_alerts(conn: sqlite3.Connection, args) -> None:
    severity: Optional[str] = args.severity
    step: Optional[int] = args.step
    limit: int = args.limit or 50

    c = conn.cursor()
    conditions = []
    params: list = []

    if severity:
        conditions.append("severity = ?")
        params.append(severity.upper())
    if step is not None:
        conditions.append("step = ?")
        params.append(step)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    c.execute(
        f"SELECT step, severity, alert_type, layer_name, message, value "
        f"FROM alerts {where} ORDER BY step DESC, severity LIMIT ?",
        params + [limit],
    )
    rows = c.fetchall()

    if not rows:
        print("No alerts found.")
        return

    print(_c(f"{'Step':<8} {'Severity':<10} {'Type':<18} {'Layer':<40} {'Message'}", _BOLD))
    print("-" * 120)
    for step_val, sev, atype, layer, msg, val in rows:
        color = _RED if sev == "ERROR" else _YELLOW
        print(
            f"{step_val:<8} "
            f"{_c(sev, color):<19} "   # 颜色码会加宽字符串，补偿对齐
            f"{atype:<18} "
            f"{(layer or '')[:40]:<40} "
            f"{msg or ''}"
        )


# ─── stats ────────────────────────────────────────────────────────────────────

def cmd_stats(conn: sqlite3.Connection, args) -> None:
    step: int = args.step
    phase: Optional[str] = args.phase
    layer: Optional[str] = args.layer

    c = conn.cursor()
    conditions = ["step = ?"]
    params: list = [step]

    if phase:
        conditions.append("phase = ?")
        params.append(phase)
    if layer:
        conditions.append("layer_name LIKE ?")
        params.append(f"%{layer}%")

    where = "WHERE " + " AND ".join(conditions)
    c.execute(
        f"""
        SELECT layer_name, phase, dtype,
               nan_count, inf_count,
               max_val, mean_val, std_val,
               fp16_sat, fp16_udf
        FROM   layer_stats
        {where}
        ORDER  BY layer_name, phase
        """,
        params,
    )
    rows = c.fetchall()

    if not rows:
        print(f"No stats data for step={step}.")
        return

    # 表头
    header = (
        f"{'Layer':<45} {'Phase':<10} {'dtype':<12} "
        f"{'NaN':>6} {'Inf':>6} "
        f"{'max':>12} {'mean':>12} {'std':>12} "
        f"{'fp16_sat':>9} {'fp16_udf':>9}"
    )
    print(_c(header, _BOLD))
    print("-" * 140)

    for name, ph, dtype, nan_c, inf_c, max_v, mean_v, std_v, fp16_s, fp16_u in rows:
        nan_str = _c(str(nan_c), _RED) if nan_c else "-"
        inf_str = _c(str(inf_c), _RED) if inf_c else "-"

        sat_val = fp16_s or 0.0
        sat_str = _c(f"{sat_val:.1%}", _RED if sat_val > 0.10 else (_YELLOW if sat_val > 0.01 else ""))

        udf_val = fp16_u or 0.0
        udf_str = _c(f"{udf_val:.1%}", _YELLOW if udf_val > 0.05 else "")

        fmt = lambda v: f"{v:.3e}" if v is not None else "-"   # noqa: E731
        print(
            f"{(name or '')[:45]:<45} {ph:<10} {(dtype or ''):<12} "
            f"{nan_str:>6} {inf_str:>6} "
            f"{fmt(max_v):>12} {fmt(mean_v):>12} {fmt(std_v):>12} "
            f"{sat_str:>9} {udf_str:>9}"
        )


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        prog="mpnanny",
        description="MixedPrecisionNanny — query metrics.db",
    )
    parser.add_argument(
        "--db",
        default="./nanny_logs/metrics.db",
        help="Path to metrics.db (default: ./nanny_logs/metrics.db)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # summary
    sub.add_parser("summary", help="Show global training summary")

    # alerts
    alert_p = sub.add_parser("alerts", help="Show alert records")
    alert_p.add_argument("--severity", choices=["ERROR", "WARNING"], help="Filter by severity")
    alert_p.add_argument("--step", type=int, help="Filter by step number")
    alert_p.add_argument("--limit", type=int, default=50, help="Max rows to display (default 50)")

    # stats
    stats_p = sub.add_parser("stats", help="Show per-layer stats for a given step")
    stats_p.add_argument("--step", type=int, required=True, help="Training step")
    stats_p.add_argument("--phase", choices=["forward", "backward"], help="Filter by phase")
    stats_p.add_argument("--layer", help="Filter by layer name (substring match)")

    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    try:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        print(f"[error] Cannot open database: {args.db}", file=sys.stderr)
        print("        (Run training first, or check --db path)", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "summary":
            cmd_summary(conn, args)
        elif args.command == "alerts":
            cmd_alerts(conn, args)
        elif args.command == "stats":
            cmd_stats(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
