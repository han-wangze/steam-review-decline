"""跨年基准率漂移：诊断 → 影响 → 三种处理方案对比。

为什么要单独一个文件
--------------------
训练段（≤2018）基准率 11.5%，测试段（2019–2022）只有 6.1%。差了将近一倍。
这会同时影响三件事：概率校准、固定阈值下的报警量、以及"模型看起来有多好"。

但"漂移"是一个笼统的说法。至少三种机制会产生同样的表象：

  H1 真实行为变化：后来的游戏确实更少掉档（玩家更宽容 / 评分通胀）
  H2 测量变化：后来的游戏评论更多，后期档位估计更稳，跨阈值的偶然变少
  H3 样本结构变化：进入样本的游戏类型/体量构成变了

三者的处理方式完全不同：H1 要改先验，H2 要改标签或加权，H3 要改样本框。
所以先诊断，再决定改什么。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUT
from src.features import FEATURES, build_features

TRAIN_MAX = 2018


def _fit_predict(d: pd.DataFrame, weight_col: str | None = None):
    X = build_features(d)
    y = d.is_drop_late.astype(int).values
    tr = (d.year <= TRAIN_MAX).values
    Xv = X.fillna(X[tr].median()).values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if weight_col:
        m.fit(Xv[tr], y[tr], logisticregression__sample_weight=d[weight_col].values[tr])
    else:
        m.fit(Xv[tr], y[tr])
    return m.predict_proba(Xv)[:, 1], y, Xv, tr


def ece(y, p, bins=6) -> float:
    if len(y) < 20:
        return np.nan
    e = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return np.nan
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    tot = 0.0
    for b in range(len(e) - 1):
        m = idx == b
        if m.sum():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(y)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shift_prior(p, pi_from, pi_to):
    """只改截距的先验校正（Saerens 2002）。不改排序，只平移 logit。"""
    return sigmoid(logit(p) + np.log(pi_to / (1 - pi_to)) - np.log(pi_from / (1 - pi_from)))


# ---------------- 1. 诊断 ----------------

def diagnose(d: pd.DataFrame) -> None:
    print("=" * 78)
    print("1. 漂移长什么样")
    print("=" * 78)
    d = d.copy()
    d["near_threshold"] = d.early_ratio.between(0.70, 0.80)
    d["abs_delta"] = d.ratio_delta.abs()
    g = d.groupby("year").agg(
        样本=("is_drop_late", "size"),
        降档率=("is_drop_late", "mean"),
        首月评论中位=("n_30", "median"),
        后期评论中位=("n_late", "median"),
        后期好评率均值=("late_ratio", "mean"),
        后期好评率标准差=("late_ratio", "std"),
        好评率变化均值=("ratio_delta", "mean"),
        变化绝对值中位=("abs_delta", "median"),
        临界占比=("near_threshold", "mean"),
    )
    g["降档率"] = (g["降档率"] * 100).round(1).astype(str) + "%"
    g["临界占比"] = (g["临界占比"] * 100).round(1).astype(str) + "%"
    for c in ["后期好评率均值", "后期好评率标准差", "好评率变化均值"]:
        g[c] = g[c].round(3)
    print(g.to_string())

    print("\n读法：")
    print("  · 降档率下降的同时，如果'好评率变化均值'也变小 → 偏向 H1（真实变稳）")
    print("  · 如果'后期评论中位'明显上升、'标准差'下降 → 偏向 H2（估计更稳，偶然跨阈值变少）")
    print("  · 如果'临界占比'下降 → 样本离阈值更远，跨阈值自然变少")


def test_hypotheses(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2. 哪个机制在起作用")
    print("=" * 78)

    early = d[d.year <= 2018]
    late = d[d.year >= 2019]

    print(f"{'指标':<22}{'≤2018':>10}{'≥2019':>10}{'变化':>10}")
    for col, label in [("n_late", "后期评论数中位"), ("late_ratio", "后期好评率均值"),
                       ("late_ratio", "后期好评率标准差"), ("ratio_delta", "好评率变化均值")]:
        f = np.median if "中位" in label else np.mean
        if "标准差" in label:
            a, b = early[col].std(), late[col].std()
        else:
            a, b = f(early[col]), f(late[col])
        print(f"{label:<22}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")

    # H2 的直接检验：把"后期评论数"固定在同一水平，看降档率还差不差
    print("\n控制后期评论数后，两段的降档率差（H2 检验）：")
    print(f"{'后期评论数':<16}{'≤2018 降档率':>14}{'≥2019 降档率':>14}{'差':>9}")
    for lo, hi in [(50, 150), (150, 400), (400, 1000), (1000, 10 ** 9)]:
        a = early[(early.n_late >= lo) & (early.n_late < hi)]
        b = late[(late.n_late >= lo) & (late.n_late < hi)]
        if len(a) < 30 or len(b) < 30:
            continue
        lab = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        print(f"{lab:<16}{a.is_drop_late.mean():>13.1%}{b.is_drop_late.mean():>14.1%}"
              f"{b.is_drop_late.mean() - a.is_drop_late.mean():>+9.1%}")
    print("  如果每一层内部两段差距都很小 → 漂移主要是'评论数变多'造成的（H2）")
    print("  如果层内差距依然很大 → 是真实的行为变化（H1）")

    # H3：样本体量构成
    print("\n样本结构（H3 检验）：")
    for yr in [2016, 2018, 2020, 2022]:
        s = d[d.year == yr]
        if len(s):
            print(f"  {yr}: n={len(s):>4}  首月评论中位 {s.n_30.median():>6.0f}  "
                  f"临界游戏(70–80%)占比 {s.early_ratio.between(0.70, 0.80).mean():>5.1%}")


# ---------------- 3. 影响 ----------------

def impact(d: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("3. 漂移影响了什么")
    print("=" * 78)
    tr = (d.year <= TRAIN_MAX).values

    print(f"{'年份':<8}{'样本':>6}{'实际率':>8}{'平均预测':>10}{'偏差':>8}{'ECE':>8}")
    for yr in sorted(d.year.unique()):
        m = (d.year == yr).values
        if m.sum() < 30:
            continue
        print(f"{yr:<8}{m.sum():>6}{y[m].mean():>8.1%}{p[m].mean():>10.1%}"
              f"{p[m].mean() - y[m].mean():>+8.1%}{ece(y[m], p[m]):>8.3f}")

    print("\n固定阈值下的报警量（说明为什么这件事不能不管）：")
    for thr in (0.10, 0.20, 0.30):
        print(f"  阈值 p ≥ {thr:.0%}:", end="")
        for yr in [2016, 2018, 2020, 2022]:
            m = (d.year == yr).values
            if m.sum() < 30:
                continue
            hit = (p[m] >= thr).mean()
            prec = y[m][p[m] >= thr].mean() if (p[m] >= thr).any() else np.nan
            print(f"  {yr}: 报警{hit:>4.0%}/命中{prec:>5.1%}", end="")
        print()


def corrections(d: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("4. 三种处理方案的实测效果")
    print("=" * 78)
    te = (d.year >= 2019).values
    pi_tr = y[(d.year <= TRAIN_MAX).values].mean()

    rows = []

    def add(name, pp, note=""):
        rows.append({
            "方案": name,
            "总体ECE": round(ece(y[te], pp[te]), 4),
            "2022偏差": round(pp[(d.year == 2022).values].mean() - y[(d.year == 2022).values].mean(), 4),
            "说明": note,
        })

    add("A. 不处理（现状）", p)

    # B. 用当年真实基准率做先验校正 —— 不可部署，只作为天花板
    pp = p.copy()
    for yr in range(2019, 2023):
        m = (d.year == yr).values
        if m.sum():
            pp[m] = shift_prior(p[m], pi_tr, y[m].mean())
    add("B. 先验校正（用当年真实率）", pp, "天花板，不可部署")

    # C. 用上一年已观测到的基准率 —— 可部署
    #    给 2022 打分时，2021 这批游戏的 365 天窗口已经关闭，基准率是可观测的。
    pp = p.copy()
    rate_by_year = {yr: y[(d.year == yr).values].mean() for yr in range(2013, 2023)}
    for yr in range(2019, 2023):
        m = (d.year == yr).values
        prev = rate_by_year.get(yr - 1)
        if m.sum() and prev:
            pp[m] = shift_prior(p[m], pi_tr, prev)
    add("C. 先验校正（用上一年率）", pp, "可部署：去年的标签已可观测")

    # D. 只用排序：按容量取 Top-K，不看绝对概率
    rows.append({
        "方案": "D. 只按容量取 Top-K",
        "总体ECE": None,
        "2022偏差": None,
        "说明": "概率不参与决策，天然免疫漂移；代价是给不出绝对风险",
    })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print("\n各方案在 2022 年的事后命中率（Top 10%）：")
    for name, pp in [("A. 不处理", p), ("B. 当年率", shift_prior(p, pi_tr, y[(d.year == 2022).values].mean())),
                     ("C. 上年率", shift_prior(p, pi_tr, rate_by_year.get(2021, pi_tr)))]:
        m = (d.year == 2022).values
        k = max(int(m.sum() * 0.1), 1)
        idx = np.argsort(-pp[m])[:k]
        print(f"  {name:<10} P@10% = {y[m][idx].mean():.1%}  （排序相同则此列应完全一致）")
    return out


def recency_weight(d: pd.DataFrame) -> None:
    """另一个杠杆：不改先验，改训练样本的时间权重。"""
    print("\n" + "=" * 78)
    print("5. 另一个杠杆：按时间给训练样本加权")
    print("=" * 78)
    from sklearn.metrics import roc_auc_score

    y = d.is_drop_late.astype(int).values
    te = (d.year >= 2019).values
    base_p, _, _, _ = _fit_predict(d)
    print(f"{'权重方案':<26}{'测试AUC':>9}{'测试ECE':>9}")
    print(f"{'不加权':<26}{roc_auc_score(y[te], base_p[te]):>9.3f}{ece(y[te], base_p[te]):>9.3f}")

    for half_life in (4, 2):
        w = 0.5 ** ((TRAIN_MAX - d.year.clip(upper=TRAIN_MAX)) / half_life)
        w = w.clip(lower=0.05)
        p, _, _, _ = _fit_predict(d.assign(_w=w), weight_col="_w")
        print(f"{f'半衰期 {half_life} 年':<26}{roc_auc_score(y[te], p[te]):>9.3f}{ece(y[te], p[te]):>9.3f}")
    print("  说明：加权能改变系数（模型更贴合近期样本），但不解决截距偏移。")


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    d = d[d.is_drop_late.notna()].copy()
    p, y, _, tr = _fit_predict(d)

    diagnose(d)
    test_hypotheses(d)
    impact(d, p, y)
    table = corrections(d, p, y)
    recency_weight(d)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "drift_corrections.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT / 'drift_corrections.csv'}")


if __name__ == "__main__":
    main()
