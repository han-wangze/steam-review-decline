"""稳健性检验：滚动切分、门槛敏感性、决策曲线。

为什么单独放
------------
主结果是**一次**时间切分（≤2018 训练 / 2019–2022 测试）算出来的 0.886。
单点数字有两个风险：
  1. 切点选得好 → 0.886 可能是"运气"
  2. 门槛是拍定的 → 结论可能对参数敏感

这个脚本回答三件事：
  Q9  换切点，AUC 的分布长什么样（滚动 origin）
  Q2  后期评论门槛取 30/50/100，结论差多少
  ——  把 AUC 翻译成"什么成本结构下值得用"（决策曲线 net benefit）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GOOD_TIERS, MAX_T0_RELEASE_LAG_D, MIN_EARLY_REVIEWS,
                    MIN_FINAL_REVIEWS, OUT)
from src.features import build_features
from src.train import lagged_prior, prior_shift


def _fit_eval(d: pd.DataFrame, tr: np.ndarray, te: np.ndarray, label="is_drop_late"):
    X = build_features(d)
    y = d[label].astype(int).values
    Xv = X.fillna(X[tr].median()).values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xv[tr], y[tr])
    p = m.predict_proba(Xv[te])[:, 1]
    return {
        "n_train": int(tr.sum()), "pos_train": int(y[tr].sum()),
        "n_test": int(te.sum()), "pos_test": int(y[te].sum()),
        "auc": roc_auc_score(y[te], p),
        "ap": average_precision_score(y[te], p),
        "base": float(y[te].mean()),
    }


# ---------------- Q9 滚动切分 ----------------

def rolling_origin(d: pd.DataFrame, origins=range(2015, 2022)) -> pd.DataFrame:
    """每个 origin 用 ≤origin 训练，预测 origin+1。

    比单次切分严格：每一次都是"只用过去预测下一年"，共 7 次。
    看的是这 7 个 AUC 的分布，不是某一个值。
    """
    years = d.year.values
    rows = []
    for o in origins:
        tr = years <= o
        te = years == o + 1
        if te.sum() < 50 or y_pos(d, te) < 8:
            continue
        r = _fit_eval(d, tr, te)
        r["origin"] = o
        r["test_year"] = o + 1
        rows.append(r)
    tab = pd.DataFrame(rows)[["origin", "test_year", "n_train", "pos_train",
                              "n_test", "pos_test", "base", "auc", "ap"]]
    print("\n" + "=" * 74)
    print("Q9 滚动切分（每次只用过去预测下一年）")
    print("=" * 74)
    show = tab.copy()
    show["base"] = (show.base * 100).round(1).astype(str) + "%"
    for c in ("auc", "ap"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    aucs = tab.auc
    ref = 0.886
    print(f"\n  AUC 中位 {aucs.median():.3f}  最小 {aucs.min():.3f}  最大 {aucs.max():.3f}  "
          f"极差 {aucs.max() - aucs.min():.3f}")
    print(f"  与单次切分的 0.886 相比：中位差 {aucs.median() - ref:+.3f}")
    print(f"  判定线：极差 < 0.08 → 0.886 可信；> 0.12 → 需报区间而非单点")
    return tab


def y_pos(d, te) -> int:
    return int(d.is_drop_late[te].astype(int).sum())


# ---------------- Q2 门槛敏感性 ----------------

def rebuild_with_threshold(all_labels: pd.DataFrame, min_late: int) -> pd.DataFrame:
    """用指定的后期评论门槛重建标签，其余筛选条件保持一致。"""
    d = all_labels
    ok = (d.early_tier.notna() & d.late_tier.notna() & (d.n_late >= min_late))
    good = d.early_tier.isin(GOOD_TIERS)
    bad = ~d.late_tier.isin(GOOD_TIERS)
    d = d.assign(_label=(good & bad).where(ok).astype("Int64"))

    keep = (
        ~d.appid.isin({730})
        & (d.n_30 >= MIN_EARLY_REVIEWS)
        & (d.n_365 >= MIN_FINAL_REVIEWS)
        & d.window_complete
        & good
        & (d.n_late >= min_late)
        & d.release_date.notna()
        & (d.rel_lag_d.abs() <= MAX_T0_RELEASE_LAG_D)
    )
    out = d[keep].copy()
    out["is_drop_late"] = out["_label"]
    return out


def threshold_sensitivity(all_labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in (30, 50, 100):
        d = rebuild_with_threshold(all_labels, thr)
        years = d.year.values
        tr, te = years <= 2018, years >= 2019
        if te.sum() < 50:
            continue
        r = _fit_eval(d, tr, te)
        r["门槛"] = f"n_late ≥ {thr}"
        rows.append(r)
    tab = pd.DataFrame(rows)[["门槛", "n_train", "pos_train", "n_test", "pos_test", "base", "auc", "ap"]]
    print("\n" + "=" * 74)
    print("Q2 后期评论门槛敏感性")
    print("=" * 74)
    show = tab.copy()
    show["base"] = (show.base * 100).round(1).astype(str) + "%"
    for c in ("auc", "ap"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    spread = tab.auc.max() - tab.auc.min()
    print(f"\n  AUC 极差 {spread:.3f}   判定线：≤0.02 写进局限即可；>0.05 需三档都报")
    return tab


# ---------------- 决策曲线 ----------------

def decision_curve(d: pd.DataFrame) -> pd.DataFrame:
    """net benefit：把"概率"换算成"什么成本结构下值得用模型"。

    NB(pt) = TP/N − FP/N × pt/(1−pt)
    pt 是决策阈值，也等价于"你认为一条误报值多少条漏报"的折算比例：
    在 pt=0.2 处行动，等于说"一条误报的代价 = 4 条漏报的代价"。
    """
    years = d.year.values
    tr, te = years <= 2018, years >= 2019
    X = build_features(d)
    y = d.is_drop_late.astype(int).values
    Xv = X.fillna(X[tr].median()).values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xv[tr], y[tr])
    p = m.predict_proba(Xv)[:, 1]

    # 先验校正（与导出层一致：每年用上一年已观测的率）
    rate = {int(k): float(v) for k, v in pd.Series(y).groupby(years).mean().items()}
    pi = float(y[tr].mean())
    pc = p.copy()
    for i, yr in enumerate(years):
        prev = lagged_prior(rate, int(yr), "y-1")
        if prev and prev > 0:
            pc[i] = prior_shift(np.array([p[i]]), pi, prev)[0]

    yt, pt_ = y[te], pc[te]
    n = len(yt)
    prev = yt.mean()
    rows = []
    for thr in (0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        flag = pt_ >= thr
        tp = (flag & (yt == 1)).sum()
        fp = (flag & (yt == 0)).sum()
        w = thr / (1 - thr)
        nb_model = tp / n - fp / n * w
        nb_all = prev - (1 - prev) * w
        rows.append({
            "决策阈值": f"{thr:.0%}",
            "复核比例": f"{flag.mean():.1%}",
            "命中率": f"{(tp / flag.sum() if flag.sum() else 0):.1%}",
            "抓到正例": f"{tp}/{int(yt.sum())}",
            "模型净收益": round(nb_model, 4),
            "全部复核净收益": round(nb_all, 4),
            "每百款净增": round((nb_model - max(nb_all, 0)) * 100, 2),
        })
    tab = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("决策曲线：在什么成本结构下值得用模型")
    print("=" * 74)
    print(tab.to_string(index=False))
    print("\n  读法：『全部复核净收益』是无脑全查的基准；模型净收益低于它时，不如不筛。")
    print("  每百款净增 = 用模型比'全查或不查'多出来的净值（单位：款）。")
    return tab


def main() -> None:
    labels = pd.read_parquet(OUT / "labels.parquet")
    all_labels = pd.read_parquet(OUT / "labels_all.parquet")

    roll = rolling_origin(labels)
    thr = threshold_sensitivity(all_labels)
    dc = decision_curve(labels)

    OUT.mkdir(parents=True, exist_ok=True)
    roll.to_csv(OUT / "robust_rolling.csv", index=False, encoding="utf-8-sig")
    thr.to_csv(OUT / "robust_threshold.csv", index=False, encoding="utf-8-sig")
    dc.to_csv(OUT / "decision_curve.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT}/robust_rolling.csv · robust_threshold.csv · decision_curve.csv")


if __name__ == "__main__":
    main()
