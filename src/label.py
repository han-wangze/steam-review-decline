"""标签层：把窗口统计转成「早期档位 → 365 天后档位」的二分类样本。

这里修掉旧版三个口径问题：

1. **观察期不再浮动**。旧版用「全部评论」，观察期从 232 天到 4,749 天不等，
   降档率随之从 4.7% 涨到 15.3% —— 标签里混进了"被观察了多久"。
   新版固定 365 天，所有样本的观察期完全一致。
2. **窗口必须完整**。只保留 t0+365d <= 数据快照 的游戏，否则晚发游戏会被
   系统性判成"没降档"。
3. **左截断单独处理**。数据从 2010-10 开始，2010 年前上市的游戏，
   其"首条评论"其实是数据起点而不是上线日。用发行日期卡 |lag| <= 30 天剔除。

同时保留旧口径（全量评论）的标签列，便于新旧对照——对照结果本身就是报告里的一节。
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DAY, EXCLUDED_APPIDS, GOOD_TIERS, LABEL_WINDOW_D,
                    MAX_T0_RELEASE_LAG_D, META_PARQUET, MIN_EARLY_REVIEWS,
                    MIN_FINAL_REVIEWS, MIN_LATE_REVIEWS, OUT, SNAPSHOT_TS,
                    WINDOWS_DIR)


def tier(ratio: float, n: int) -> str | None:
    """Steam 官方评价档位。阈值来自商店页的显示规则。"""
    if n is None or not np.isfinite(ratio):
        return None
    if n < 10:
        return "insufficient"
    if ratio >= 0.95 and n >= 500:
        return "overwhelmingly_positive"
    if ratio >= 0.80 and n >= 50:
        return "very_positive"
    if ratio >= 0.80:
        return "positive"
    if ratio >= 0.70:
        return "mostly_positive"
    if ratio >= 0.40:
        return "mixed"
    if ratio >= 0.20:
        return "mostly_negative"
    if n >= 500:
        return "overwhelmingly_negative"
    if n >= 50:
        return "very_negative"
    return "negative"


def _parse_release(s) -> pd.Timestamp | None:
    """元数据里的发行日期格式是 'Aug 1, 2023'，也有只写年份的。"""
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    try:
        return pd.Timestamp(dt.datetime.strptime(s, "%b %d, %Y"))
    except ValueError:
        pass
    m = re.match(r"^(\d{4})$", s)
    if m:
        return pd.Timestamp(int(m.group(1)), 7, 1)
    return pd.NaT


def load_windows() -> pd.DataFrame:
    files = sorted(glob.glob(str(WINDOWS_DIR / "win_part_*.parquet")))
    if not files:
        raise FileNotFoundError("没有聚合结果，先跑 src/aggregate.py")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["appid"] = df["appid"].astype("int64")

    # 跨分片的游戏（评论被切到两个 part 里）：t0 与计数都只是局部值。
    # 取覆盖最全的一份，并打标，后续按需剔除。
    n_rows = len(df)
    df["split_across_parts"] = df.appid.duplicated(keep=False)
    df = (df.sort_values("n_all")
            .drop_duplicates("appid", keep="last")
            .reset_index(drop=True))
    print(f"聚合行 {n_rows} → 去重后 {len(df)}，跨分片 {int(df.split_across_parts.sum())} 个")
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ---- 早期窗口（特征来源） ----
    d["early_ratio"] = np.where(d.n_30 > 0, d.up_30 / d.n_30, np.nan)
    d["early_tier"] = [tier(r, n) for r, n in zip(d.early_ratio, d.n_30)]

    # ---- 新口径：固定 365 天 ----
    d["final_ratio"] = np.where(d.n_365 > 0, d.up_365 / d.n_365, np.nan)
    d["final_tier"] = [tier(r, n) for r, n in zip(d.final_ratio, d.n_365)]

    # ---- 旧口径：全量评论（保留，用于对照） ----
    d["final_ratio_all"] = np.where(d.n_all > 0, d.up_all / d.n_all, np.nan)
    d["final_tier_all"] = [tier(r, n) for r, n in zip(d.final_ratio_all, d.n_all)]

    # ---- 观察期：新版固定 365；旧版随快照浮动 ----
    d["obs_days_fixed"] = LABEL_WINDOW_D
    d["obs_days_all"] = (SNAPSHOT_TS - d.t0) / DAY

    # ---- 稳健性标签：只看"首月之后"的变化 ----
    # 动机：早期 30 天本身包含在 365 天窗口内，若一款游戏的评论几乎全在首月，
    # 它的 final_ratio 会被 early_ratio 主导，天生"降不了档"。
    # 实测首月评论占 365 天总量的中位 37.8%，重叠不算致命，但仍会虚高单特征 AUC
    # 约 0.02。更关键的是：不重叠的标签正例更多（8.5% vs 4.5%），统计功效翻倍。
    # 所以把它作为主口径。
    d["n_late"] = (d.n_365 - d.n_30).clip(lower=0)
    d["up_late"] = (d.up_365 - d.up_30).clip(lower=0)
    d["late_ratio"] = np.where(d.n_late > 0, d.up_late / d.n_late, np.nan)
    d["late_tier"] = [tier(r, n) for r, n in zip(d.late_ratio, d.n_late)]
    d["ratio_delta"] = d.final_ratio - d.early_ratio

    # ---- 标签 ----
    ok = d.early_tier.notna() & d.final_tier.notna()
    val = d.early_tier.isin(GOOD_TIERS) & ~d.final_tier.isin(GOOD_TIERS)
    d["is_drop"] = val.where(ok).astype("Int64")

    # 主标签：只看第 30–365 天，与特征窗完全不重叠
    ok_late = d.early_tier.notna() & d.late_tier.notna() & (d.n_late >= MIN_LATE_REVIEWS)
    val_late = d.early_tier.isin(GOOD_TIERS) & ~d.late_tier.isin(GOOD_TIERS)
    d["is_drop_late"] = val_late.where(ok_late).astype("Int64")

    ok_all = d.early_tier.notna() & d.final_tier_all.notna()
    val_all = d.early_tier.isin(GOOD_TIERS) & ~d.final_tier_all.isin(GOOD_TIERS)
    d["is_drop_all"] = val_all.where(ok_all).astype("Int64")

    # ---- 发行日校准 ----
    if META_PARQUET.exists():
        meta = pd.read_parquet(META_PARQUET, columns=["appID", "release_date"])
        meta["key"] = meta.appID.astype(str).str.strip()
        meta["rel"] = meta.release_date.map(_parse_release)
        m = dict(zip(meta.key, meta.rel))
        d["release_date"] = d.appid.astype(str).map(m)
        d["t0_dt"] = pd.to_datetime(d.t0, unit="s")
        d["rel_lag_d"] = (d.t0_dt - d.release_date).dt.days
    else:
        d["release_date"] = pd.NaT
        d["t0_dt"] = pd.to_datetime(d.t0, unit="s")
        d["rel_lag_d"] = np.nan

    # ---- 完整性 ----
    d["window_complete"] = (d.t0 + LABEL_WINDOW_D * DAY) <= SNAPSHOT_TS
    d["year"] = d.t0_dt.dt.year
    return d


def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    steps = []

    def step(name, mask):
        nonlocal d
        before = len(d)
        d = d[mask].copy()
        steps.append((name, before - len(d), len(d)))

    step("排除跨分片脏数据", ~d.appid.isin(EXCLUDED_APPIDS))
    step("排除非首月样本(n_30<10)", d.n_30 >= MIN_EARLY_REVIEWS)
    step("排除档位判不出(n_365<100)", d.n_365 >= MIN_FINAL_REVIEWS)
    step("排除观察窗不完整", d.window_complete)
    # 只保留"开局是好评"的游戏。否则负类里会混进大量本来就差的游戏，
    # 模型学的是"会不会变好"而不是"会不会变差"，问题的语义就变了。
    step("排除开局即非好评", d.early_tier.isin(GOOD_TIERS))
    step("排除后期评论不足(n_late<50)", d.n_late >= MIN_LATE_REVIEWS)
    step("排除无发行日期", d.release_date.notna())
    step("排除上线日偏差>30天", d.rel_lag_d.abs() <= MAX_T0_RELEASE_LAG_D)

    print("\n筛选过程：")
    for name, cut, left in steps:
        print(f"  {name:<26} -{cut:>5}  → {left:>5}")
    return d


def window_sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    """换标签窗口的代价：窗口越短，样本越干净但正例越少。

    d 应是**未筛选**的全量（含档位判不出的），这样才能看出各窗口的可用量。
    """
    rows = []
    for w in (90, 180, 365):
        ncol, ucol = f"n_{w}", f"up_{w}"
        if ncol not in d:
            continue
        complete = (d.t0 + w * DAY) <= SNAPSHOT_TS
        ratio = np.where(d[ncol] > 0, d[ucol] / d[ncol], np.nan)
        tier_w = [tier(r, n) for r, n in zip(ratio, d[ncol])]
        ok = d.early_tier.isin(GOOD_TIERS) & complete & (d[ncol] >= MIN_FINAL_REVIEWS)
        drop = d.early_tier.isin(GOOD_TIERS) & ~pd.Series(tier_w).isin(GOOD_TIERS).values
        sel = ok.values
        rows.append({
            "窗口": f"{w} 天",
            "可用样本": int(sel.sum()),
            "正例": int((drop & sel).sum()),
            "降档率": f"{(drop & sel).sum() / max(sel.sum(), 1):.1%}",
            "窗口不完整的游戏": int((~complete).sum()),
        })
    out = pd.DataFrame(rows)
    print("\n标签窗口敏感性（窗口越短越干净，但正例越少）：")
    print(out.to_string(index=False))
    return out


def compare_old_new(d: pd.DataFrame) -> pd.DataFrame:
    """新旧口径对照：同一批游戏，换标签定义后结论变了多少。"""
    rows = []
    for name, drop_col, obs_col in [
        ("旧：全量评论（观察期浮动）", "is_drop_all", "obs_days_all"),
        ("中：固定 365 天（窗口重叠）", "is_drop", "obs_days_fixed"),
        ("新：只看第 30–365 天（不重叠）", "is_drop_late", "obs_days_fixed"),
    ]:
        s = d.dropna(subset=[drop_col])
        rows.append({
            "口径": name,
            "样本": len(s),
            "正例": int(s[drop_col].sum()),
            "降档率": f"{s[drop_col].mean():.1%}",
            "观察期中位(天)": int(s[obs_col].median()),
            "观察期范围": f"{s[obs_col].min():.0f}–{s[obs_col].max():.0f}",
        })
    out = pd.DataFrame(rows)
    print("\n新旧口径对照：")
    print(out.to_string(index=False))

    print("\n旧口径的观察期混淆（降档率随观察期上升 = 标签被时间污染）：")
    s = d.dropna(subset=["is_drop_all"]).copy()
    s["obs_bin"] = pd.cut(s.obs_days_all, [0, 730, 1460, 2555, 3650, 99999],
                          labels=["<2年", "2-4年", "4-7年", "7-10年", ">10年"])
    g = s.groupby("obs_bin", observed=True).agg(n=("is_drop_all", "size"),
                                                降档率=("is_drop_all", "mean"))
    g["降档率"] = (g["降档率"] * 100).round(1).astype(str) + "%"
    print(g.to_string())

    print("\n新口径（固定 365 天）按上线年份：")
    s2 = d.dropna(subset=["is_drop_late"])
    g2 = s2.groupby("year").agg(n=("is_drop_late", "size"), 正例=("is_drop_late", "sum"),
                                降档率=("is_drop_late", "mean"))
    g2["降档率"] = (g2["降档率"] * 100).round(1).astype(str) + "%"
    print(g2.to_string())
    return out


def main() -> None:
    raw = load_windows()
    lab = build_labels(raw)
    lab_all = lab.copy()
    sel = apply_filters(lab)

    OUT.mkdir(parents=True, exist_ok=True)
    lab_all.to_parquet(OUT / "labels_all.parquet", index=False)
    sel.to_parquet(OUT / "labels.parquet", index=False)

    print(f"\n全量 {len(lab_all)} → 建模样本 {len(sel)}")
    print(f"其中降档 {int(sel.is_drop_late.sum())} ({sel.is_drop_late.mean():.1%})")
    window_sensitivity(lab_all)
    compare_old_new(sel)
    print(f"\n已写 {OUT / 'labels.parquet'}")


if __name__ == "__main__":
    main()
