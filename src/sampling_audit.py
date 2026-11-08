"""抽样审计：等步长取分片，会不会让样本偏斜。

先纠正一个曾经写错的假设
------------------------
README 早期版本说"分片按 appid 升序切分，所以等步长抽样在时间轴上是铺开的"。
实测不成立：221 个已采分片里，appid 跨度的中位数占全量程的 99%，
即绝大多数分片同时含有最早和最晚的 appid。分片编号与年份中位甚至是负相关（−0.43）。

真正的切分方式是**按评论行数均衡**：每片 10.4 万–33 万行，中位 16.4 万，
没有一片低于 5 万行；游戏数与行数负相关（Spearman −0.50），
说明评论多的游戏会独占一个分片。

这带来两个推论，都要检验：
1. 因为每个游戏只落在**一个**分片里，等步长抽 1/3 的分片约等于对所有游戏做
   等概率抽样 —— 不会按时间偏斜。
2. 反过来，分片均衡于"行数"而不均衡于"游戏数"，所以游戏数在片间差 700 倍
   （1–735）。这不影响等概率性，但会让"按片平均"这类统计失去意义。

两种检验
--------
A. 片内结构（离线）：每片行数、游戏数、appid 跨度、年份中位。
B. 跳过分片抽检（联网）：挑几个没被采到的分片，在内存里聚合成游戏级，
   与已采样本比年份分布与评论量构成。若接近，说明"取 0/3/6 还是 1/4/7"不影响构成。

局限：B 只抽 4 片，能否定"相邻片差异巨大"，不能证明全库无偏。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUT, PARTS, WINDOWS_DIR
from src.aggregate import _fetch, aggregate_part


def part_table() -> pd.DataFrame:
    rows = []
    for f in sorted(WINDOWS_DIR.glob("win_part_*.parquet")):
        p = int(f.stem.split("_")[-1])
        d = pd.read_parquet(f, columns=["appid", "t0", "n_all"])
        if d.empty:
            continue
        yr = pd.to_datetime(d.t0, unit="s").dt.year
        rows.append({
            "分片": p, "游戏数": len(d), "评论行数": int(d.n_all.sum()),
            "appid跨度占比": round(float((d.appid.max() - d.appid.min()) / max(d.appid.max(), 1)), 3),
            "年份中位": int(yr.median()),
            "≥100条占比": round(float((d.n_all >= 100).mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("分片").reset_index(drop=True)


def structure(per: pd.DataFrame) -> None:
    print("=== A. 分片结构（共 %d 片）===" % len(per))
    r = per.评论行数
    print(f"  每片评论行数  中位 {r.median():.0f}  范围 {r.min()}–{r.max()}  CV={r.std() / r.mean():.3f}")
    print(f"  每片游戏数    中位 {per.游戏数.median():.0f}  范围 {per.游戏数.min()}–{per.游戏数.max()}")
    print(f"  appid 跨度占全量程的比例  中位 {per.appid跨度占比.median():.3f}"
          f"  （接近 1 表示分片不是按 appid 切的）")
    print(f"  分片编号 × 年份中位 Spearman = {per.分片.corr(per.年份中位, method='spearman'):.3f}")
    print(f"  游戏数 × 评论行数 Spearman = {per.游戏数.corr(per.评论行数, method='spearman'):.3f}"
          f"  （负值 = 评论多的游戏独占分片）")
    print(f"\n  解读：分片按评论行数均衡、不按 appid 切；每个游戏只属于一个分片，"
          f"因此等步长抽 1/3 对游戏近似等概率。")


def probe(parts: list[int], per: pd.DataFrame) -> None:
    print("\n=== B. 跳过分片抽检 ===")
    print("  单片构成差异很大（游戏数 1–735），所以不能把几片合起来当总体。")
    print("  做法：把每个被跳过的分片，放进已采 221 片的分布里看它的分位。\n")
    print(f"  {'分片':<8}{'游戏数':>7}{'年份中位':>9}{'年份分位':>9}{'≥100条':>9}{'占比分位':>9}")
    rows: list[dict] = []
    for p in parts:
        buf = _fetch(p)
        if buf is None:
            print(f"  part.{p} 下载失败，跳过")
            continue
        try:
            d = aggregate_part(buf)
        except Exception as exc:                       # noqa: BLE001
            print(f"  part.{p} 解析失败 {exc!r}"[:120])
            continue
        if d.empty:
            continue
        yr = pd.to_datetime(d.t0, unit="s").dt.year
        ymed = int(yr.median())
        share = float((d.n_all >= 100).mean())
        pct_year = float((per.年份中位 <= ymed).mean())
        pct_share = float((per["≥100条占比"] <= share).mean())
        print(f"  {f'part.{p}':<8}{len(d):>7}{ymed:>9}{pct_year:>9.0%}{share:>9.2f}{pct_share:>9.0%}")
        rows.append({"分片": p, "游戏数": len(d), "年份中位": ymed,
                     "年份分位": round(pct_year, 3), "≥100条占比": round(share, 3),
                     "占比分位": round(pct_share, 3)})

    if rows:
        pd.DataFrame(rows).to_csv(OUT / "sampling_probe.csv", index=False, encoding="utf-8-sig")

    print(f"\n  参照：已采 221 片的年份中位 {per.年份中位.min()}–{per.年份中位.max()}"
          f"（中位 {per.年份中位.median():.0f}），"
          f"≥100 条占比 {per['≥100条占比'].min():.2f}–{per['≥100条占比'].max():.2f}"
          f"（中位 {per['≥100条占比'].median():.2f}）。")
    print("  读法：分位数落在 10%–90% 之间，说明该片与已采样本没有系统性差别；"
          "落在两端也未必是偏斜，更可能是单片样本量太小（大游戏独占一片）。")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="只做离线结构审计，不下载跳过分片")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    per = part_table()
    per.to_csv(OUT / "sampling_parts.csv", index=False, encoding="utf-8-sig")
    structure(per)
    if args.offline:
        print("\n（--offline：跳过联网抽检）")
        return
    skipped = [p for p in (1, 200, 401, 601) if p not in set(PARTS)]
    probe(skipped, per)


if __name__ == "__main__":
    main()
