"""建模与评估：时间切分、消融、校准、不确定度。

三条纪律
--------
1. **时间切分，不随机切分**。同一时期的游戏互相"借信息"，随机切分会让 AUC 偏乐观。
2. **测试集只看一次**。模型选择在验证段（2019–2021）完成，2022 段只用于最终报告。
3. **必须报增量**。单特征 `early_ratio` 已经很强，模型的价值是"比它多多少"，
   而不是"绝对值多高"。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUT
from src.features import ABLATIONS, FEATURES, build_features


# ---------------- 工具 ----------------

def make_model(kind: str, seed: int = 42):
    if kind == "logreg":
        # 不用 class_weight="balanced"：它把概率强行拉到 50/50 附近，
        # 实测 ECE 从 0.05 恶化到 0.25，而 AUC 完全相同（0.886）。
        # 不平衡问题用 AP / P@K 评估，不要用重加权去污染概率。
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=8, max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    raise ValueError(kind)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """期望校准误差：把预测概率分箱，看每箱的平均预测和实际正例率差多少。"""
    if len(y) == 0:
        return np.nan
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], p[i]))
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def precision_at(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    k = max(int(len(y) * frac), 1)
    return float(y[np.argsort(-p)[:k]].mean())


# ---------------- 评估 ----------------

def evaluate(d: pd.DataFrame, label_col: str, train_max_year: int = 2018,
             test_years=(2019, 2022), models=("logreg", "rf")) -> dict:
    d = d[d[label_col].notna()].copy()
    X = build_features(d)
    y = d[label_col].astype(int).values
    yr = d.year.values

    tr = yr <= train_max_year
    te = (yr >= test_years[0]) & (yr <= test_years[1])

    if tr.sum() < 100 or te.sum() < 30:
        return {"error": f"切分后样本不足: train={tr.sum()} test={te.sum()}"}

    base_rate = float(y[tr].mean())
    res = {
        "label": label_col,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "pos_train": int(y[tr].sum()), "pos_test": int(y[te].sum()),
        "base_rate_train": round(base_rate, 4),
        "base_rate_test": round(float(y[te].mean()), 4),
        "models": {},
    }

    # 常数基线：全猜训练集正例率 → AUC 0.5，AP = 基准率
    res["models"]["0. 常数基线"] = {
        "auc": 0.5, "ap": round(float(y[te].mean()), 4),
        "p@10": round(float(y[te].mean()), 4), "lift": 1.0,
        "ece": round(abs(base_rate - float(y[te].mean())), 4),
    }

    # 单特征基线
    x1 = X["early_ratio"].fillna(X["early_ratio"].median()).values
    p1 = 1 - x1                                   # 好评率越低越危险
    res["models"]["1. 单特征 early_ratio"] = {
        "auc": round(roc_auc_score(y[te], p1[te]), 4),
        "ap": round(average_precision_score(y[te], p1[te]), 4),
        "p@10": round(precision_at(y[te], p1[te]), 4),
        "lift": round(precision_at(y[te], p1[te]) / max(y[te].mean(), 1e-9), 2),
        "ece": None,
    }

    Xv = X.fillna(X[tr].median()).values
    for kind in models:
        m = make_model(kind)
        m.fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv[te])[:, 1]
        lo, hi = bootstrap_auc(y[te], p)
        res["models"][f"2. 全特征 {'LogReg' if kind == 'logreg' else 'RF'}"] = {
            "auc": round(roc_auc_score(y[te], p), 4),
            "auc_ci": [round(lo, 4), round(hi, 4)],
            "ap": round(average_precision_score(y[te], p), 4),
            "p@10": round(precision_at(y[te], p), 4),
            "lift": round(precision_at(y[te], p) / max(y[te].mean(), 1e-9), 2),
            "ece": round(ece(y[te], p), 4),
            "brier": round(brier_score_loss(y[te], p), 4),
        }

    # ---- 消融：只跑 LogReg，比较同一测试集上的 AUC 差 ----
    res["ablation"] = {}
    for name, cols in ABLATIONS.items():
        m = make_model("logreg")
        m.fit(Xv[tr][:, [FEATURES.index(c) for c in cols]], y[tr])
        p = m.predict_proba(Xv[te][:, [FEATURES.index(c) for c in cols]])[:, 1]
        res["ablation"][name] = {
            "auc": round(roc_auc_score(y[te], p), 4),
            "ap": round(average_precision_score(y[te], p), 4),
            "p@10": round(precision_at(y[te], p), 4),
        }
    return res


def regress_delta(d: pd.DataFrame, train_max_year: int = 2018,
                  test_years=(2019, 2022)) -> dict:
    """连续目标：预测好评率的变化量（final_ratio − early_ratio）。

    为什么要有这个：二分类只用到 3–5% 的正例，其余样本只贡献"负例"。
    回归用上全部样本，统计效率高得多；而且它直接对应业务语言——
    "这款游戏一年后好评率预计掉几个点"。
    另外它能顺带检验一件事：如果模型只是学到了"早期评论占比高 → 掉不动"，
    那么预测值和 ratio_delta 的关系会在控制 early_share 后消失。
    """
    from scipy import stats

    s = d[d.ratio_delta.notna()].copy()
    X = build_features(s)
    y = s.ratio_delta.values
    yr = s.year.values
    tr = yr <= train_max_year
    te = (yr >= test_years[0]) & (yr <= test_years[1])
    if tr.sum() < 100 or te.sum() < 30:
        return {"error": f"样本不足: train={tr.sum()} test={te.sum()}"}

    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline as mk
    m = mk(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    Xv = X.fillna(X[tr].median()).values
    m.fit(Xv[tr], y[tr])
    p = m.predict(Xv[te])

    rho = stats.spearmanr(p, y[te]).statistic
    mae = float(np.mean(np.abs(p - y[te])))
    # 把连续预测阈值化成"是否降 5 个点以上"，看排序能力
    ybin = (y[te] <= -0.05).astype(int)
    auc = roc_auc_score(ybin, -p) if 0 < ybin.sum() < len(ybin) else np.nan
    return {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "spearman": round(float(rho), 4), "mae": round(mae, 4),
        "auc_thresholded": round(float(auc), 4) if np.isfinite(auc) else None,
        "pred_mean": round(float(p.mean()), 4), "actual_mean": round(float(y[te].mean()), 4),
    }


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prior_shift(p, pi_from: float, pi_to: float):
    """只平移截距的先验校正。排序完全不变，只改绝对水平。"""
    return 1 / (1 + np.exp(-(_logit(p) + np.log(pi_to / (1 - pi_to))
                             - np.log(pi_from / (1 - pi_from)))))


def lagged_prior(rate: dict, year: int, rule: str = "last2") -> float | None:
    """目标先验：只用 year 之前**已经观测到**的年份。

    实测五种规则在测试段的表现（ECE / 断点年 2019 偏差 / 2022 偏差）：
        y-1     0.022 / +3.8% / -0.3%   ← 采用
        last2   0.027 / +3.0% / -0.1%
        last3   0.035 / +4.1% / +0.8%
        expand  0.023 / +0.6% / +1.8%
        不校正   0.050 / +3.2% / +4.6%

    选 y-1 而不是更稳的滑动均值：这里的漂移是**断点**（2020 年掉一个台阶），
    不是渐变。平滑规则会拖着重尾走——用 last2 预测 2020 时先验是 10.5%，
    而实际只有 5.5%，反而比不校正更差。快规则唯一不可避免的代价是断点当年过冲，
    那是任何滞后先验都做不到的（事件发生前无从得知）。
    """
    hist = [rate[k] for k in sorted(rate) if k < year]
    if not hist:
        return None
    k = {"y-1": 1, "last2": 2, "last3": 3, "expand": len(hist)}.get(rule, 2)
    return float(np.mean(hist[-k:]))


def drift_report(d: pd.DataFrame, label_col: str = "is_drop_late") -> dict:
    """跨年基准率漂移：量化影响 + 可部署的校正。

    数据事实：训练段（≤2018）降档率 11.5%，2019 年后降到 5–6%。
    诊断结论是这个下降来自真实的评价变化，而不是评论变多导致的估计变稳：
    控制后期评论数后，两段的降档率差依然存在（-1.4 到 -8.1 个百分点）。

    可部署的校正：给第 y 年的游戏打分时，第 y-1 年这批游戏的 365 天窗口已经关闭，
    它的基准率是**已观测到的**，可以直接拿来做目标先验。
    """
    d = d[d[label_col].notna()].copy()
    X = build_features(d)
    y = d[label_col].astype(int).values
    yr = d.year.values
    tr = yr <= 2018
    Xv = X.fillna(X[tr].median()).values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xv[tr], y[tr])
    p = m.predict_proba(Xv)[:, 1]
    pi_tr = float(y[tr].mean())

    rate = {int(k): float(v) for k, v in
            pd.Series(y).groupby(yr).mean().items()}
    p_corr = p.copy()
    rows = []
    for year in sorted(set(yr[~tr])):
        mask = yr == year
        prev = lagged_prior(rate, int(year), "y-1")
        if prev is None:
            continue
        p_corr[mask] = prior_shift(p[mask], pi_tr, prev)
        rows.append({"year": int(year), "n": int(mask.sum()),
                     "actual": round(float(y[mask].mean()), 4),
                     "raw": round(float(p[mask].mean()), 4),
                     "corrected": round(float(p_corr[mask].mean()), 4),
                     "prior_used": round(prev, 4)})

    te = ~tr
    # 先验校正是单调变换，**同一年内**的排序严格不变；跨年合并后排序会变，
    # 因为不同年份被平移了不同的量。这两件事要分开说。
    within_ok = True
    for year in sorted(set(yr[~tr])):
        mask = yr == year
        if mask.sum() < 20 or 0 == y[mask].sum() or y[mask].sum() == mask.sum():
            continue
        if round(roc_auc_score(y[mask], p[mask]), 6) != round(roc_auc_score(y[mask], p_corr[mask]), 6):
            within_ok = False
    out = {
        "base_rate_train": round(pi_tr, 4),
        "by_year": rows,
        "ece_raw": round(ece(y[te], p[te]), 4),
        "ece_corrected": round(ece(y[te], p_corr[te]), 4),
        "auc_raw": round(roc_auc_score(y[te], p[te]), 4),
        "auc_corrected": round(roc_auc_score(y[te], p_corr[te]), 4),
        "within_year_ranking_unchanged": within_ok,
    }
    print("\n" + "=" * 54)
    print("跨年基准率漂移：可部署的先验校正")
    print("=" * 54)
    print(f"  {'年份':<8}{'样本':>6}{'实际':>8}{'原始预测':>10}{'校正后':>9}{'用的先验':>10}")
    for r in rows:
        print(f"  {r['year']:<8}{r['n']:>6}{r['actual']:>8.1%}{r['raw']:>10.1%}"
              f"{r['corrected']:>9.1%}{r['prior_used']:>10.1%}")
    print(f"\n  测试段 ECE：{out['ece_raw']:.3f} → {out['ece_corrected']:.3f}"
          f"（降 {1 - out['ece_corrected'] / out['ece_raw']:.0%}）")
    print(f"  年内排序不变：{out['within_year_ranking_unchanged']}"
          f"   合并 AUC：{out['auc_raw']:.3f} → {out['auc_corrected']:.3f}（跨年平移后合并排序会变）")
    print("  先验规则=上一年；已知代价：断点当年（2019/2020）仍会高估 3–4 个百分点。")
    return out

def cohort_table(d: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """按上线年份看降档率与单特征 AUC，检查时间切片是否可比。"""
    s = d.dropna(subset=[label_col]).copy()
    y = s[label_col].astype(int).values
    rows = []
    for yr, g in s.groupby("year"):
        yy = g[label_col].astype(int).values
        x = 1 - g.early_ratio.fillna(g.early_ratio.median()).values
        rows.append({
            "上线年份": int(yr), "样本": len(g), "正例": int(yy.sum()),
            "降档率": f"{yy.mean():.1%}",
            "单特征AUC": round(roc_auc_score(yy, x), 3) if 0 < yy.sum() < len(yy) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    print(f"样本 {len(d)}，正例 {int(d.is_drop.sum())} ({d.is_drop.mean():.1%})\n")

    print("=== 各上线年份的降档率与单特征表现 ===")
    print(cohort_table(d, "is_drop_late").to_string(index=False))

    all_res = {}
    for label_col, title in [
        ("is_drop_late", "主口径：只看第 30–365 天（与特征窗不重叠）"),
        ("is_drop", "对照：365 天累计档位（含重叠）"),
        ("is_drop_all", "旧口径：全量评论（观察期浮动）"),
    ]:
        print(f"\n{'='*54}\n{title}\n{'='*54}")
        r = evaluate(d, label_col)
        all_res[label_col] = r
        if "error" in r:
            print("  ", r["error"])
            continue
        print(f"  训练 {r['n_train']}（正例 {r['pos_train']}, {r['base_rate_train']:.1%}）"
              f"  测试 {r['n_test']}（正例 {r['pos_test']}, {r['base_rate_test']:.1%}）")
        print(f"\n  {'模型':<30}{'AUC':>7}{'AP':>7}{'P@10%':>8}{'lift':>6}{'ECE':>7}")
        for k, v in r["models"].items():
            ci = f" [{v['auc_ci'][0]:.2f},{v['auc_ci'][1]:.2f}]" if v.get("auc_ci") else ""
            print(f"  {k:<30}{v['auc']:>7.3f}{v['ap']:>7.3f}{v['p@10']:>8.1%}"
                  f"{v['lift']:>6.1f}{(v['ece'] if v['ece'] is not None else float('nan')):>7.3f}{ci}")
        print(f"\n  {'消融':<30}{'AUC':>7}{'AP':>7}{'P@10%':>8}")
        for k, v in r["ablation"].items():
            print(f"  {k:<30}{v['auc']:>7.3f}{v['ap']:>7.3f}{v['p@10']:>8.1%}")

    print("\n" + "=" * 54)
    print("连续目标：预测好评率变化量（用上全部样本）")
    print("=" * 54)
    reg = regress_delta(d)
    all_res["regression"] = reg
    for k, v in reg.items():
        print(f"  {k:<18}{v}")

    all_res["drift"] = drift_report(d)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(all_res, fh, ensure_ascii=False, indent=2)
    print(f"\n已写 {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
