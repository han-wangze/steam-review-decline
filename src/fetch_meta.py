"""把发行日期从完整元数据里抽出来，做成仓库可携带的小表。

背景
----
label.py 需要发行日期来校准"首条评论 ≈ 上线日"（剔除抢先体验期长达数年的游戏）。
完整元数据是 184MB 的单文件，不适合进仓库，也不该依赖某个本机绝对路径——
否则别人 clone 下来跑，会因为找不到文件而**静默产出空样本**（这类失败最难查）。

所以：抽取 appID + release_date 两列（约 1.1MB）随仓库分发；
完整元数据只在需要重新生成时下载一次，用完即删。

用法
----
    python src/fetch_meta.py                      # 缺文件时才下载（约 2 分钟）
    python src/fetch_meta.py --source <大文件>     # 直接用本地已有的完整元数据
    python src/fetch_meta.py --force              # 强制重建小表
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA, META_HF_URL, META_PARQUET

COLS = ["appID", "release_date"]
FULL_CACHE = DATA / "_full_meta.parquet"


def _download(dest: str, attempts: int = 3) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for i in range(attempts):
        try:
            print(f"下载完整元数据（184MB）… 第 {i + 1} 次")
            req = urllib.request.Request(META_HF_URL, headers={"User-Agent": "steam-risk/1.0"})
            with opener.open(req, timeout=600) as resp, open(dest, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            if os.path.getsize(dest) > 10_000_000:
                return
        except Exception as exc:                     # noqa: BLE001
            print(f"  失败：{exc!r}"[:120])
        time.sleep(3 * (i + 1))
    raise RuntimeError("下载完整元数据失败，请手动下载后用 --source 指定")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="", help="本地已有的完整元数据 parquet")
    ap.add_argument("--force", action="store_true", help="小表已存在也重建")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if META_PARQUET.exists() and not args.force:
        print(f"已存在 {META_PARQUET.name}，跳过（--force 可重建）")
        return

    if args.source:
        src = args.source
    elif FULL_CACHE.exists():
        src = str(FULL_CACHE)
    else:
        _download(str(FULL_CACHE))
        src = str(FULL_CACHE)

    df = pq.ParquetFile(src).read(columns=COLS).to_pandas()
    df.to_parquet(META_PARQUET, index=False)
    size_mb = META_PARQUET.stat().st_size / 1e6
    print(f"写出 {META_PARQUET.name}：{len(df):,} 行 × {len(df.columns)} 列，{size_mb:.1f} MB")

    # 用完就删：184MB 的中间文件没有保留价值（只有一个文件，不会触发批量删除保护）
    if not args.source and FULL_CACHE.exists():
        FULL_CACHE.unlink()
        print("已清理下载缓存")


if __name__ == "__main__":
    main()
