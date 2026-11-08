"""一条命令跑完整条链路：聚合 → 元数据 → 标签 → 审计 → 建模 → 稳健性 → 导出。

    python run_all.py            # 增量跑（已有聚合结果会跳过）
    python run_all.py --refresh  # 强制重新下载并聚合全部分片（221 个，约 12 分钟）

--skip-aggregate 只跳过聚合这一步，后续步骤照常执行。
任一步骤退出码非零都会立即中止，不会在缺数据的状态下继续往下跑。

依赖：pandas / numpy / pyarrow / scikit-learn / scipy。下载走 urllib，不需要 curl。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [PY, str(ROOT / script)] + (extra or [])
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{script} 失败（code={r.returncode}）")
    print(f"  ✓ {script} 用时 {time.time() - t:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="重新下载并聚合全部分片")
    ap.add_argument("--skip-aggregate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_aggregate:
        run("src/aggregate.py", ["--force"] if args.refresh else [])
    run("src/fetch_meta.py")   # 缺发行日期表时才下载，已有则跳过
    run("src/label.py")
    run("src/check.py")
    run("src/train.py")
    run("src/robustness.py")
    run("src/export.py")
    print(f"\n全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"产物：out/metrics.json · out/predictions.json · site/index.html")


if __name__ == "__main__":
    main()
