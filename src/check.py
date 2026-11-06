"""自动审计：把方法论纪律写成断言。

为什么需要这个文件
------------------
之前手动踩过两次坑：一次是 7 个特征里 5 个用了全量统计（AUC 从 0.866 掉到 0.821），
一次是把"最后一条评论时间"当成观察终点（观察期被评论活跃度污染）。
这两类错误都不会报错，只会静默给出一个好看的数字。

所以把检查写成代码，跑不通就不许往下走。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DAY, LABEL_WINDOW_D, OUT, SNAPSHOT_TS
from src.features import FEATURES, build_features

# 只在打标签时可用的列，绝不能进特征
FORBIDDEN = {"n_all", "up_all", "t_max", "final_ratio", "final_tier",
             "is_drop", "is_drop_all", "obs_days_all", "final_ratio_all", "final_tier_all"}

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    print(f"样本 {len(d)}，正例 {int(d.is_drop.sum())} ({d.is_drop.mean():.1%})\n")

    print("1. 特征不含未来信息")
    leaked = FORBIDDEN & set(FEATURES)
    check("FEATURES 与全量统计无交集", not leaked, str(leaked) if leaked else "")

    print("\n2. 窗口完整性")
    incomplete = int(((d.t0 + LABEL_WINDOW_D * DAY) > SNAPSHOT_TS).sum())
    check("所有样本 365 天窗口均完整", incomplete == 0, f"不完整 {incomplete} 条")
    check("观察期恒定 365 天", d.obs_days_fixed.nunique() == 1)

    print("\n3. 计数单调性（窗口越长评论越多）")
    mono = ((d.n_7 <= d.n_30) & (d.n_30 <= d.n_90) & (d.n_90 <= d.n_180) & (d.n_180 <= d.n_365)).mean()
    check("n_7 ≤ n_30 ≤ n_90 ≤ n_180 ≤ n_365", mono == 1.0, f"满足率 {mono:.1%}")
    check("n_365 ≤ n_all", (d.n_365 <= d.n_all).all())

    print("\n4. 标签与前缀一致")
    good = d.early_tier.isin({"overwhelmingly_positive", "very_positive", "positive", "mostly_positive"})
    bad = ~d.final_tier.isin({"overwhelmingly_positive", "very_positive", "positive", "mostly_positive"})
    expect = (good & bad).astype(int)
    check("is_drop = (早期好) 且 (365 天后不好)", (expect == d.is_drop.astype(int)).all())
    check("所有样本早期档位都是好的", bool(good.all()))

    print("\n5. 特征缺失率")
    X = build_features(d)
    for c in FEATURES:
        rate = X[c].isna().mean()
        check(f"{c} 缺失 < 10%", rate < 0.10, f"{rate:.1%}")

    print("\n6. 时间切分")
    tr = d[d.year <= 2018]
    te = d[(d.year >= 2019) & (d.year <= 2022)]
    check("训练与测试年份不重叠", tr.year.max() < te.year.min(),
          f"≤{tr.year.max()} vs {te.year.min()}–{te.year.max()}")
    check("训练集正例 ≥60", int(tr.is_drop.sum()) >= 60, f"{int(tr.is_drop.sum())}")
    check("测试集正例 ≥30", int(te.is_drop.sum()) >= 30, f"{int(te.is_drop.sum())}")

    print("\n7. 类别与年份分布")
    tab = d.groupby("year").agg(n=("is_drop", "size"), 正例=("is_drop", "sum"))
    tab["降档率"] = (tab["正例"] / tab["n"] * 100).round(1).astype(str) + "%"
    print(tab.to_string())

    print("\n" + "=" * 50)
    if failures:
        print(f"审计未通过：{len(failures)} 项")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("审计全部通过")


if __name__ == "__main__":
    main()
