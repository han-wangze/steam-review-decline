"""标签审计：降档里有多少是噪声。

要回答的问题
------------
主标签 = "早期 [0,30] 档位好（n_30≥10） → 第 30–365 天档位不好（n_late≥50）"。
两个窗口的样本量差一个量级，早期比率的估计噪声远大于后期；而且要求"早期是好档"
意味着噪声只能单向制造正例。所以"降档"里可能混着大量回归均值。

三项检验
--------
A. 同长度对照（关键）：都取 30 天的窗口，只是位置不同
     [0,30] 好 → [30,60] 不好     第 2 个月：刚选完，回归均值最强
     [0,30] 好 → [330,360] 不好   第 12 个月：同样 30 天，噪声水平相近
   如果第 12 个月的翻转率不比第 2 个月高，说明"11 个月里在往下走"这件事没有证据。

B. 安慰剂（不同长度）：把后期窗口换成 [30,365] 的原定义，与 A 同期对比，
   看真实窗口（335 天、样本量更大）是否显著高于 30 天窗口。

C. 零假设模拟：以观测到的早期数据为条件做 Beta-Binomial 后验预测
   （p ~ Beta(up_30+1, n_30−up_30+1)，后期 ~ Binomial(n_late, p)），
   得到"真实质量不变时会被判成降档"的比例——这是纯噪声地板。

D. 跨分片切断：用元数据 positive+negative 当分母，看 n_all 是否被截断。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_EARLY_REVIEWS, OUT

THRESH = 0.70          # 本样本的 n 范围内：GOOD ⇔ 好评率 ≥ 0.70


def win(d: pd.DataFrame, i: int) -> tuple[np.ndarray, np.ndarray]:
    """第 i 个 30 天窗口的 (n, up)。i=0 → [0,30]。"""
    n = (d[f"c_n_{i:02d}"] - (d[f"c_n_{i-1:02d}"] if i else 0)).astype(float)
    u = (d[f"c_up_{i:02d}"] - (d[f"c_up_{i-1:02d}"] if i else 0)).astype(float)
    return n.values, u.values


def main() -> None:
    d = pd.read_parquet(OUT / "labels.parquet")
    d = d[d.is_drop_late.notna()].copy()

    n1, u1 = win(d, 0)          # [0,30]
    n2, u2 = win(d, 1)          # [30,60]
    n12, u12 = win(d, 11)       # [330,360]
    r1, r2, r12 = u1 / n1, np.where(n2 > 0, u2 / n2, np.nan), np.where(n12 > 0, u12 / n12, np.nan)
    early_ok = (n1 >= MIN_EARLY_REVIEWS) & (r1 >= THRESH)

    print("=" * 80)
    print("A. 同长度对照：两个 30 天窗口，只有位置不同")
    print("=" * 80)
    print(f"{'窗口门槛':<10}{'可用样本':>9}{'第2个月翻转':>13}{'第12个月翻转':>14}{'差':>9}")
    same_len = {}
    for mn in (10, 30, 50):
        m2 = early_ok & (n2 >= mn)
        m12 = early_ok & (n12 >= mn)
        if m2.sum() < 50 or m12.sum() < 50:
            continue
        p2 = float(((r2 < THRESH) & m2).sum() / m2.sum())
        p12 = float(((r12 < THRESH) & m12).sum() / m12.sum())
        same_len[mn] = {"n2": int(m2.sum()), "n12": int(m12.sum()),
                        "flip_2": round(p2, 4), "flip_12": round(p12, 4)}
        print(f"{'≥' + str(mn):<10}{int(m2.sum()):>9}{p2:>12.1%}{p12:>13.1%}{p12 - p2:>+9.1%}")
    print("  读法：两个窗口长度相同、噪声水平接近。若第 12 个月不高于第 2 个月，"
          "就没有证据支持「11 个月里在持续下滑」。")

    print("\n" + "=" * 80)
    print("B. 与真实标签对照（观测定义 n_late≥50，窗口 335 天）")
    print("=" * 80)
    real = d.is_drop_late.astype(int).values == 1
    print(f"  实测降档率 {real.mean():.1%}（{int(real.sum())}/{len(d)}）")
    if same_len:
        k = 50 if 50 in same_len else sorted(same_len)[-1]
        print(f"  第 2 个月翻转 {same_len[k]['flip_2']:.1%}，第 12 个月翻转 {same_len[k]['flip_12']:.1%}"
              f"（同门槛 ≥{k}）")

    print("\n" + "=" * 80)
    print("C. 零假设：以观测到的早期数据为条件做后验预测")
    print("=" * 80)
    rng = np.random.default_rng(0)
    S = 300
    a = (u1 + 1.0)[:, None]
    b = (n1 - u1 + 1.0)[:, None]
    nlate = d.n_late.values.astype(float)[:, None]
    p_sim = rng.beta(np.repeat(a, S, axis=1), np.repeat(b, S, axis=1))   # (n, S)
    late = rng.binomial(np.repeat(nlate, S, axis=1).astype(int), p_sim) \
        / np.repeat(nlate, S, axis=1)
    null_rate = float((late < THRESH).mean())          # 已条件在"早期观测是好档"上

    obs = float(real.mean())
    print(f"  实测降档率        {obs:.1%}")
    print(f"  零假设下的降档率  {null_rate:.1%}")
    print(f"  超出噪声的部分    {obs - null_rate:+.1%}"
          f"（占标签 {(obs - null_rate) / obs:.0%}）")

    print("\n  分首月评论量（实测 vs 零假设）：")
    print(f"  {'n_30':<12}{'样本':>6}{'实测':>9}{'零假设':>9}{'差值':>9}")
    for lo, hi in [(10, 50), (50, 100), (100, 300), (300, 10**9)]:
        m = ((d.n_30 >= lo) & (d.n_30 < hi)).values
        if m.sum() < 50:
            continue
        nr = float((late[m] < THRESH).mean())
        lab = f"{lo}–{hi if hi < 10**9 else '∞'}"
        print(f"  {lab:<12}{int(m.sum()):>6}{real[m].mean():>8.1%}{nr:>9.1%}"
              f"{real[m].mean() - nr:>+9.1%}")

    print("\n" + "=" * 80)
    print("D. 跨分片切断：n_all 与元数据总评论数之比")
    print("=" * 80)
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "..", "dataset_probe", "steam_games.parquet")
    if not os.path.exists(meta_path):
        print("  本地没有完整元数据，跳过")
    else:
        meta = pd.read_parquet(meta_path, columns=["appID", "positive", "negative"])
        meta["key"] = meta.appID.astype(str).str.strip()
        meta["tot"] = (pd.to_numeric(meta.positive, errors="coerce").fillna(0)
                       + pd.to_numeric(meta.negative, errors="coerce").fillna(0))
        mp = dict(zip(meta.key, meta.tot))
        d["meta_tot"] = d.appid.astype(str).map(mp)
        cov = (d.n_all / d.meta_tot).replace([np.inf, -np.inf], np.nan)
        ok = cov.notna() & (d.meta_tot > 100)
        c = cov[ok]
        print(f"  可比样本 {int(ok.sum())} 款，覆盖率中位 {c.median():.2f}")
        for th in (0.9, 0.7, 0.5, 0.3):
            print(f"    覆盖率 < {th:.1f}：{int((c < th).sum()):>4} 款（{(c < th).mean():.1%}）")
        low = d[ok.values & (cov < 0.5).values]
        if len(low):
            print(f"  疑似被切断（覆盖率<0.5）：{len(low)} 款，示例：")
            for _, r in low.nlargest(5, "n_all").iterrows():
                print(f"    {str(r.game)[:36]:<36} n_all={int(r.n_all):>7} "
                      f"元数据={int(r.meta_tot):>7} 覆盖={r.n_all / r.meta_tot:.2f}")

    with open(OUT / "label_audit.json", "w", encoding="utf-8") as fh:
        json.dump({"observed": round(obs, 4), "null_beta_binom": round(null_rate, 4),
                   "same_length": {str(k): v for k, v in same_len.items()}},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n已写 {OUT / 'label_audit.json'}")


if __name__ == "__main__":
    main()
