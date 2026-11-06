"""导出层：把模型输出变成前端能直接读的 JSON。

两个诚实性设计
--------------
1. **训练/测试分开标注**。模型只用 2021 年及以前的游戏训练，2022 年的预测是
   真正的样本外。前端默认只展示样本外，样本内结果标注 `in_sample`。
2. **每条预测都给归因**。LogReg 的系数 × 标准化后的特征值就是该样本的贡献，
   正贡献 = 推高风险。前三条写进 JSON，前端直接展示，不用再算。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CURVE_POINTS, OUT, SITE
from src.features import FEATURES, build_features
from src.train import lagged_prior, prior_shift

FEATURE_CN = {
    "early_ratio": "前 30 天好评率",
    "log_n_early": "前 30 天评论量",
    "log_n_7": "首周评论量",
    "share_7day": "首周评论占首月比例",
    "decay_ratio": "首月/首周评论倍数",
    "playtime_h": "早期玩家中位时长",
    "playtime_p90_h": "早期玩家 P90 时长",
    "log_numgames": "玩家账号游戏数",
    "lang_top_share": "第一大语言占比",
    "lang_n": "语言种类数",
    "rlen_med": "评论文本长度",
    "nonzero_playtime_share": "有时长记录占比",
}


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    d = d[d.is_drop_late.notna()].copy()
    X = build_features(d)
    y = d.is_drop_late.astype(int).values
    yr = d.year.values

    train = yr <= 2021
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
    model.fit(X.fillna(X[train].median()).values[train], y[train])

    Xf = X.fillna(X[train].median()).values
    p_raw = model.predict_proba(Xf)[:, 1]

    # 跨年基准率漂移校正：第 y 年用第 y-1 年**已观测**的降档率当目标先验。
    # 单调变换，不改变年内排序，只把绝对水平拉回各年份的真实基准。
    rate = {int(k): float(v) for k, v in
            pd.Series(y).groupby(d.year.values).mean().items()}
    pi_train = float(y[train].mean())
    p = p_raw.copy()
    for i, _year in enumerate(d.year.values):
        prev = lagged_prior(rate, int(_year), "y-1")
        if prev is not None and prev > 0:
            p[i] = prior_shift(np.array([p_raw[i]]), pi_train, prev)[0]
    print(f"先验校正：训练基准 {pi_train:.1%}，各年用上一年率 "
          f"({min(rate.values()):.1%}–{max(rate.values()):.1%})")

    scaler: StandardScaler = model[0]
    coef = model[-1].coef_[0]
    contrib = (Xf - scaler.mean_) / scaler.scale_ * coef        # (n, k)

    games = []
    for i, row in enumerate(d.itertuples()):
        order = np.argsort(-contrib[i])[:3]
        top = [{
            "feature": FEATURE_CN.get(FEATURES[j], FEATURES[j]),
            "contribution": round(float(contrib[i][j]), 3),
        } for j in order if contrib[i][j] > 0]

        curve_n = [int(getattr(row, f"c_n_{k:02d}")) for k in range(CURVE_POINTS)]
        curve_up = [int(getattr(row, f"c_up_{k:02d}")) for k in range(CURVE_POINTS)]

        games.append({
            "appid": int(row.appid),
            "game": str(row.game),
            "year": int(row.year),
            "risk": round(float(p[i]), 4),
            "risk_raw": round(float(p_raw[i]), 4),
            "is_drop": int(row.is_drop_late),
            "in_sample": bool(yr[i] <= 2021),
            "n_30": int(row.n_30),
            "n_365": int(row.n_365),
            "early_ratio": round(float(row.early_ratio), 4),
            "late_ratio": round(float(row.late_ratio), 4) if pd.notna(row.late_ratio) else None,
            "early_tier": row.early_tier,
            "late_tier": row.late_tier,
            "playtime_h": round(float(row.playtime_med) / 60, 1) if pd.notna(row.playtime_med) else None,
            "top": top,
            "curve_n": curve_n,
            "curve_up": curve_up,
        })

    games.sort(key=lambda g: -g["risk"])
    oos = [g for g in games if not g["in_sample"]]
    payload = {
        "meta": {
            "n_games": len(games),
            "n_out_of_sample": len(oos),
            "positives": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "train_max_year": 2021,
            "label": "上线 30 天内档位为好 → 第 30–365 天档位不再是好（窗口不重叠）",
            "calibration": "按年用上一年已观测基准率做先验校正；年内排序不变",
            "positives_oos": sum(g["is_drop"] for g in oos),
        },
        "games": games,
    }

    # 首页那句"基准 5.7% / 前 10% 命中 30.2%"从 metrics.json 取，避免页面写死数字后与模型脱节
    mj = OUT / "metrics.json"
    if mj.exists():
        with open(mj, encoding="utf-8") as fh:
            met = json.load(fh).get("is_drop_late", {})
        payload["meta"]["test_base_rate"] = met.get("base_rate_test")
        payload["meta"]["test_p_at_10"] = (met.get("models", {})
                                           .get("2. 全特征 LogReg", {}).get("p@10"))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    SITE.mkdir(parents=True, exist_ok=True)
    with open(SITE / "data.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    # 另存一份 data.js：浏览器直接双击打开 file:// 时也能读（fetch 会被 CORS 拦住）
    with open(SITE / "data.js", "w", encoding="utf-8") as fh:
        fh.write("window.__DATA__=")
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";")

    print(f"导出 {len(games)} 款游戏（样本外 {len(oos)}），"
          f"正例 {int(y.sum())}（{y.mean():.1%}）")
    print(f"已写 {OUT / 'predictions.json'} 与 {SITE / 'data.json'}")


if __name__ == "__main__":
    main()
