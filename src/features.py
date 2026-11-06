"""特征层：只使用 [t0, t0+30d] 内可观测的量。

命名约定
--------
- 带 `e_` 前缀的是"早期窗口"统计，允许进模型
- 不带前缀但实际来自 30 天窗口的（playtime_med 等）由 aggregate.py 保证窗口边界
- **任何全量统计（n_all / up_all / t_max）都不允许出现在 FEATURES 里**，
  它们只在打标签时使用，见 src/label.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 进入模型的特征列。改这个列表时，确认每一项在 t0+30d 时点都拿得到。
FEATURES = [
    "early_ratio",          # 前 30 天好评率（最强单特征）
    "log_n_early",          # 前 30 天评论量（取对数）
    "log_n_7",              # 首周评论量：能区分"爆发后熄火"与"持续有人玩"
    "share_7day",           # 首周评论占 30 天的比例
    "decay_ratio",          # n_30 / n_7，>1 表示首周之后仍在持续获得评论
    "playtime_h",           # 早期评论者中位时长（小时）
    "playtime_p90_h",       # 早期评论者 p90 时长
    "log_numgames",         # 早期评论者账号拥有游戏数（对数）
    "lang_top_share",       # 第一大语言占比（本地化集中度）
    "lang_n",               # 出现语言数
    "rlen_med",             # 评论文本长度中位
    "nonzero_playtime_share",  # 有时长记录的比例（0 = 可能没真玩）
]


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    f["early_ratio"] = d.early_ratio.astype(float)
    f["log_n_early"] = np.log1p(d.n_30.astype(float))
    f["log_n_7"] = np.log1p(d.n_7.astype(float))
    f["share_7day"] = d.share_7day.astype(float)
    f["decay_ratio"] = np.where(d.n_7 > 0, d.n_30 / d.n_7.clip(lower=1), np.nan)
    f["playtime_h"] = d.playtime_med.astype(float) / 60.0
    f["playtime_p90_h"] = d.playtime_p90.astype(float) / 60.0
    f["log_numgames"] = np.log1p(d.numgames_med.astype(float))
    f["lang_top_share"] = d.lang_top_share.astype(float)
    f["lang_n"] = d.lang_n.astype(float)
    f["rlen_med"] = d.rlen_med.astype(float)
    f["nonzero_playtime_share"] = d.nonzero_playtime_share.astype(float)
    return f[FEATURES]


# 消融用的子集定义。报告里的"增量来自哪里"就靠这几组。
ABLATIONS = {
    "A. 仅好评率": ["early_ratio"],
    "B. 好评率 + 评论量": ["early_ratio", "log_n_early"],
    "C. 去掉好评率（只看数量与人群）": [c for c in FEATURES if c != "early_ratio"],
    "D. 去掉评论量（只看质量与人群）": [c for c in FEATURES if c not in ("log_n_early", "log_n_7", "decay_ratio")],
    "E. 全特征": FEATURES,
}


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """报告用：特征在样本上的分布，以及和标签的单变量关系。"""
    from sklearn.metrics import roc_auc_score

    y = d.is_drop.astype(int).values
    rows = []
    for c in FEATURES:
        x = d[c].astype(float)
        m = x.notna().values
        auc = roc_auc_score(y[m], x[m]) if m.sum() > 20 and 0 < y[m].sum() < m.sum() else np.nan
        rows.append({
            "特征": c,
            "缺失率": f"{1 - x.notna().mean():.1%}",
            "中位": round(float(x.median()), 3) if x.notna().any() else np.nan,
            "AUC": round(float(auc), 3) if np.isfinite(auc) else np.nan,
            "方向": ("越大越易降档" if auc > 0.5 else "越小越易降档") if np.isfinite(auc) else "-",
        })
    return pd.DataFrame(rows)
