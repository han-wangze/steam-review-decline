"""条件评估：模型在哪个区间真的有分辨力。

动机
----
现有指标（全样本 AUC 0.886、增量 +0.014）是把所有游戏摊平算的。但业务上模型
只在"首月好评率接近档位线"的游戏上有用——好评率 0.98 的游戏不需要预测，
0.72 的才需要。把这两类混在一起算，会把模型的价值平均掉。

本脚本回答三个问题：
1. 临界带内，模型比单特征（1 − 早期好评率）强多少？
2. 这个"强多少"在 bootstrap 下是否稳定？
3. 全样本的 +0.014 有多少来自临界带，多少来自带外的噪声？

判定线（预先写死，防止事后解释）：
    带内 AP 增量 ≥ +0.05 且 bootstrap 下界 > 0  → 模型价值集中在临界带，继续投入
    +0.02 ~ +0.05                              → 只有诊断价值
    < +0.02                                    → 前提不成立，停止后续优化
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUT
from src.features import build_features
from src.train import make_model

BAND = (0.70, 0.85)          # 主临界带：早期档位刚过线，最可能翻档
NARROW = (0.70, 0.80)        # 窄带：紧贴 0.80（very_positive）门槛


def metrics(y: np.ndarray, s: np.ndarray, frac: float = 0.10) -> dict:
    if len(y) < 30 or y.sum() < 5 or (1 - y).sum() < 5:
        return {}
    k = max(int(len(y) * frac), 1)
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "base": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, s)), 4),
        "ap": round(float(average_precision_score(y, s)), 4),
        "p@10": round(float(y[np.argsort(-s)[:k]].mean()), 4),
    }


def boot_delta(y: np.ndarray, s_model: np.ndarray, s_single: np.ndarray,
               n_boot: int = 1000, seed: int = 0) -> dict:
    """成对 bootstrap：同一批重采样上算两个分数的差，减少方差。"""
    rng = np.random.default_rng(seed)
    da, dp = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yy = y[i]
        if yy.sum() < 5 or (1 - yy).sum() < 5:
            continue
        a = roc_auc_score(yy, s_model[i]) - roc_auc_score(yy, s_single[i])
        p = (average_precision_score(yy, s_model[i])
             - average_precision_score(yy, s_single[i]))
        da.append(a)
        dp.append(p)
    if not da:
        return {}
    return {
        "d_auc": round(float(np.mean(da)), 4),
        "d_auc_ci": [round(float(np.percentile(da, 2.5)), 4),
                     round(float(np.percentile(da, 97.5)), 4)],
        "d_ap": round(float(np.mean(dp)), 4),
        "d_ap_ci": [round(float(np.percentile(dp, 2.5)), 4),
                    round(float(np.percentile(dp, 97.5)), 4)],
    }


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    d = d[d.is_drop_late.notna()].copy()
    X = build_features(d)
    y = d.is_drop_late.astype(int).values
    yr = d.year.values

    tr = yr <= 2018
    model = make_model("logreg")
    model.fit(X.fillna(X[tr].median()).values[tr], y[tr])
    s_model = model.predict_proba(X.fillna(X[tr].median()).values)[:, 1]
    s_single = (1 - d.early_ratio.values).astype(float)     # 好评率越低越危险

    er = d.early_ratio.values
    in_band = (er >= BAND[0]) & (er <= BAND[1])
    in_narrow = (er >= NARROW[0]) & (er < NARROW[1])
    in_test = yr >= 2019

    subsets = [
        ("全样本", np.ones(len(d), bool)),
        (f"临界带 {BAND[0]:.2f}–{BAND[1]:.2f}", in_band),
        (f"窄带 {NARROW[0]:.2f}–{NARROW[1]:.2f}", in_narrow),
        (f"带外 >{BAND[1]:.2f}", er > BAND[1]),
        ("测试年（2019–2022）", in_test),
        ("测试年 ∩ 临界带", in_test & in_band),
    ]

    print("=" * 88)
    print("条件评估：模型在哪个区间有分辨力")
    print("=" * 88)
    print(f"{'子集':<24}{'n':>6}{'正例':>6}{'基准':>7}"
          f"{'单特征AUC':>10}{'模型AUC':>9}{'ΔAUC':>8}{'单特征AP':>10}{'模型AP':>8}{'ΔAP':>8}")

    res = {}
    for name, m in subsets:
        ym = y[m]
        a = metrics(ym, s_single[m])
        b = metrics(ym, s_model[m])
        if not a or not b:
            print(f"{name:<24}{len(ym):>6}{int(ym.sum()):>6}  样本不足")
            continue
        res[name] = {"single": a, "model": b,
                     "delta": boot_delta(ym, s_model[m], s_single[m])}
        print(f"{name:<24}{b['n']:>6}{b['pos']:>6}{b['base']:>7.1%}"
              f"{a['auc']:>10.3f}{b['auc']:>9.3f}{b['auc'] - a['auc']:>+8.3f}"
              f"{a['ap']:>10.3f}{b['ap']:>8.3f}{b['ap'] - a['ap']:>+8.3f}")

    print("\n成对 bootstrap（1000 次）下的增量区间：")
    for name in ("全样本", f"临界带 {BAND[0]:.2f}–{BAND[1]:.2f}", "测试年 ∩ 临界带"):
        if name not in res or not res[name]["delta"]:
            continue
        dd = res[name]["delta"]
        print(f"  {name:<24} ΔAUC {dd['d_auc']:+.3f} [{dd['d_auc_ci'][0]:+.3f}, {dd['d_auc_ci'][1]:+.3f}]"
              f"   ΔAP {dd['d_ap']:+.3f} [{dd['d_ap_ci'][0]:+.3f}, {dd['d_ap_ci'][1]:+.3f}]")

    # 正例分布：临界带到底装了多少信号
    pos_all = int(y.sum())
    pos_band = int(y[in_band].sum())
    print(f"\n临界带装了多少正例：{pos_band}/{pos_all} = {pos_band / pos_all:.1%}"
          f"（样本占 {in_band.mean():.1%}）")

    # 二维：好评率 × 评论量
    print("\n分格：好评率两段 × 首月评论量三段（模型 AUC / 正例数）")
    d["er_bin"] = np.where(er <= BAND[1], f"≤{BAND[1]:.2f}", f">{BAND[1]:.2f}")
    d["n_bin"] = pd.cut(d.n_30, [0, 100, 400, 10**9], labels=["10–100", "100–400", "400+"])
    print(f"  {'':<12}{'10–100':>18}{'100–400':>18}{'400+':>18}")
    for e in [f"≤{BAND[1]:.2f}", f">{BAND[1]:.2f}"]:
        cells = []
        for nb in ["10–100", "100–400", "400+"]:
            m = ((d.er_bin == e) & (d.n_bin == nb)).values
            mm = metrics(y[m], s_model[m])
            cells.append(f"{mm['auc']:.3f} / {mm['pos']}" if mm else "样本不足")
        print(f"  {e:<12}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")


    # ---- 带内 P@10 的同口径对照：AUC 判死、P@10 看还活着，必须给同一把尺子 ----
    print("\n" + "=" * 88)
    print("带内同口径对照（P@10）：模型 vs 单特征")
    print("=" * 88)
    print(f"  {'子集':<26}{'n':>6}{'正例':>6}{'单特征P@10':>12}{'模型P@10':>10}{'差':>8}")
    for name, m in [(f"临界带 {BAND[0]:.2f}–{BAND[1]:.2f}", in_band & (~tr)),
                    ("窄带 0.70–0.80", in_narrow & (~tr)),
                    ("测试年全样本", ~tr)]:
        yv = y[m]
        if len(yv) < 30 or yv.sum() < 5:
            continue
        a = metrics(yv, s_single[m])
        b = metrics(yv, s_model[m])
        if not a or not b:
            continue
        print(f"  {name:<26}{b['n']:>6}{b['pos']:>6}{a['p@10']:>11.1%}{b['p@10']:>10.1%}"
              f"{b['p@10'] - a['p@10']:>+8.1%}")

    # ---- 带定义敏感性：上界 0.80 / 0.85 / 0.90 ----
    print("\n带定义敏感性（仅测试年 2019–2022，下界固定 0.70）")
    print(f"  {'上界':<8}{'n':>6}{'正例':>6}{'单特征AUC':>11}{'模型AUC':>9}{'ΔAUC':>8}"
          f"{'模型P@10':>10}")
    for hi in (0.80, 0.85, 0.90):
        m = (~tr) & (er >= BAND[0]) & (er <= hi)
        yv = y[m]
        a = metrics(yv, s_single[m])
        b = metrics(yv, s_model[m])
        if not a or not b:
            print(f"  {hi:<8}样本不足")
            continue
        print(f"  {hi:<8}{b['n']:>6}{b['pos']:>6}{a['auc']:>11.3f}{b['auc']:>9.3f}"
              f"{b['auc'] - a['auc']:>+8.3f}{b['p@10']:>9.1%}")

    with open(OUT / "conditional.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\n已写 {OUT / 'conditional.json'}")

    # ---- 附加检验：带内弱，是因为训练集被"容易样本"主导吗 ----
    # 若只用临界带样本训练，带内表现会不会变好？这一条决定"模型没用"还是"训练口径不对"。
    print("\n" + "=" * 88)
    print("附加检验：只用临界带样本训练（训练年 ≤2018，测试年 ≥2019）")
    print("=" * 88)
    tr_band = tr & in_band
    te_band = (~tr) & in_band
    if tr_band.sum() >= 100 and y[tr_band].sum() >= 15:
        Xv = X.fillna(X[tr].median()).values
        m2 = make_model("logreg")
        m2.fit(Xv[tr_band], y[tr_band])
        s2 = m2.predict_proba(Xv)[:, 1]

        def one(yv, sv, tag):
            a = metrics(yv, sv)
            if not a:
                print(f"  {tag}: 样本不足")
                return
            print(f"  {tag:<34} n={a['n']:<5} 正例={a['pos']:<4} "
                  f"AUC={a['auc']:.3f}  AP={a['ap']:.3f}  P@10={a['p@10']:.1%}")

        one(y[te_band], s_model[te_band], "全样本训练 → 带内测试")
        one(y[te_band], s2[te_band], "带内训练 → 带内测试")
        one(y[te_band], s_single[te_band], "单特征（好评率）")

        bd = boot_delta(y[te_band], s2[te_band], s_model[te_band])
        if bd:
            print(f"\n  带内训练 vs 全样本训练（带内测试）："
                  f"ΔAUC {bd['d_auc']:+.3f} [{bd['d_auc_ci'][0]:+.3f}, {bd['d_auc_ci'][1]:+.3f}]"
                  f"  ΔAP {bd['d_ap']:+.3f} [{bd['d_ap_ci'][0]:+.3f}, {bd['d_ap_ci'][1]:+.3f}]")
        print(f"  （带内训练样本 {int(tr_band.sum())}，正例 {int(y[tr_band].sum())}）")


if __name__ == "__main__":
    main()
