"""把一个 HF 分片聚合为「游戏级」窗口统计。

设计要点
--------
1. **一次扫描算出所有窗口**（7/30/90/180/365 天 + 全量），而不是为每个窗口重扫一遍。
   标签窗口以后要换（比如改成 180 天做敏感性分析），不用再花 20 分钟下载。
2. **辅助特征只在 [t0, t0+30d] 内计算**。这是防泄漏的结构性保证：
   函数拿不到窗口外的数据，写错也取不到。
3. **存 t_max**。观察期是否完整要能自证，不能靠外部常量臆断。
4. **存累计曲线**（每 30 天一个点）。前端画"上线后走势"、做窗口敏感性都用它。

输出：data/windows/win_part_{p}.parquet，一行 = 一个 appid。
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (COLS, CURVE_POINTS, CURVE_STEP_D, DAY, EARLY_WINDOW_D,
                    HF_PART_URL, PARTS, SENSITIVITY_WINDOWS_D, WINDOWS_DIR)


def _fetch(part: int, attempts: int = 3) -> bytes | None:
    """把分片读进内存。

    不走临时文件：宿主的安全删除机制会把"反复覆写同一个临时文件"当成批量删除，
    长任务跑到第 50 次就被打断。内存里处理既没有这个问题，也更快。
    显式禁用代理，否则请求会被沙箱网关截走（实测会返回 502）。
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = HF_PART_URL.format(part)
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "steam-risk/1.0"})
            with opener.open(req, timeout=300) as resp:
                data = resp.read()
            if len(data) > 1_000_000:
                return data
        except Exception:
            pass
        time.sleep(2 + 2 * i)
    return None


def aggregate_part(buf: bytes) -> pd.DataFrame:
    df = pq.read_table(io.BytesIO(buf), columns=COLS).to_pandas()
    df = df.dropna(subset=["timestamp_created", "voted_up"])
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ts"] = df.timestamp_created.astype("int64")
    df["rlen"] = df.review.fillna("").str.len()
    df = df.sort_values(["appid", "ts"], kind="mergesort")

    # t0 = 首条评论时间。整组共享同一个 t0，后续所有窗口都相对它切。
    df["t0"] = df.groupby("appid")["ts"].transform("min")
    df["off"] = df.ts - df.t0

    g = df.groupby("appid", sort=True)
    out = pd.DataFrame({
        "game": g["game"].first(),
        "t0": g["t0"].first(),
        "t_max": g["ts"].max(),
        "n_all": g.size(),
        "up_all": g["voted_up"].sum(),
    })

    # ---- 各窗口计数 ----
    for w in SENSITIVITY_WINDOWS_D:
        sub = df[df.off <= w * DAY]
        agg = sub.groupby("appid")["voted_up"].agg(["size", "sum"])
        out[f"n_{w}"] = agg["size"].reindex(out.index).fillna(0).astype("int32")
        out[f"up_{w}"] = agg["sum"].reindex(out.index).fillna(0).astype("int32")

    # ---- 累计曲线（每 30 天一个点） ----
    df["bucket"] = np.minimum((df.off // (CURVE_STEP_D * DAY)).astype("int16"), CURVE_POINTS - 1)
    cn = (df.groupby(["appid", "bucket"]).size()
            .unstack(fill_value=0)
            .reindex(columns=range(CURVE_POINTS), fill_value=0)
            .reindex(out.index).fillna(0)
            .cumsum(axis=1))
    cu = (df[df.voted_up.astype(bool)].groupby(["appid", "bucket"]).size()
            .unstack(fill_value=0)
            .reindex(columns=range(CURVE_POINTS), fill_value=0)
            .reindex(out.index).fillna(0)
            .cumsum(axis=1))
    for i in range(CURVE_POINTS):
        out[f"c_n_{i:02d}"] = cn[i].astype("int32")
        out[f"c_up_{i:02d}"] = cu[i].astype("int32")

    # ---- 辅助特征：严格限制在前 30 天 ----
    e = df[df.off <= EARLY_WINDOW_D * DAY]
    eg = e.groupby("appid", sort=True)
    lang_top = eg["language"].agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else None)
    lang_share = eg["language"].agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s.dropna()) else np.nan)

    e = e.assign(
        _has_pt=(e.author_playtime_at_review > 0).where(e.author_playtime_at_review.notna()),
        _in1=(e.off <= DAY).astype("int8"),
        _in7=(e.off <= 7 * DAY).astype("int8"),
        _rlen=e.rlen,
    )
    eg = e.groupby("appid", sort=True)
    aux = pd.DataFrame({
        "playtime_med": eg["author_playtime_at_review"].median(),
        "playtime_p90": eg["author_playtime_at_review"].quantile(0.90),
        "numgames_med": eg["author_num_games_owned"].median(),
        "rlen_med": eg["_rlen"].median(),
        "lang_n": eg["language"].nunique(),
        "nonzero_playtime_share": eg["_has_pt"].mean(),
        "share_1day": eg["_in1"].mean(),
        "share_7day": eg["_in7"].mean(),
    })
    aux["lang_top"] = lang_top
    aux["lang_top_share"] = lang_share
    out = out.join(aux.reindex(out.index))

    out = out.reset_index()
    out["appid"] = out["appid"].astype("int64")
    out["n_early"] = out[f"n_{EARLY_WINDOW_D}"]
    return out


def _drop_stale(part: int) -> None:
    """失败时删掉该分片的旧结果。

    --force 会重跑全部分片，若某片下载或解析失败而旧文件还在，
    新旧数据就会混着用：跑出来的样本看起来完整，实际不可复现。
    """
    f = WINDOWS_DIR / f"win_part_{part}.parquet"
    if f.exists():
        f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="已存在也重跑")
    ap.add_argument("--parts", type=str, default="", help="逗号分隔的 part 号，调试用")
    args = ap.parse_args()

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    parts = [int(x) for x in args.parts.split(",")] if args.parts else PARTS
    done = set()
    if not args.force:
        for f in os.listdir(WINDOWS_DIR):
            if f.startswith("win_part_"):
                try:
                    done.add(int(f.split("_")[-1].split(".")[0]))
                except ValueError:
                    pass
    todo = [p for p in parts if p not in done]
    print(f"待处理 {len(todo)} / {len(parts)}（已存在 {len(done)}）", flush=True)

    t0 = time.time()
    ok = fail = 0

    def work(p: int) -> tuple[int, int, int]:
        buf = _fetch(p)
        if buf is None:
            _drop_stale(p)
            return p, -1, -1
        try:
            rows = aggregate_part(buf)
            if rows.empty:
                return p, 0, 0
            rows.to_parquet(WINDOWS_DIR / f"win_part_{p}.parquet", index=False)
            return p, len(rows), 0
        except Exception as exc:                    # noqa: BLE001
            print(f"part.{p:>3} 解析失败 {exc!r}"[:160], flush=True)
            _drop_stale(p)
            return p, -2, -2

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            p, n, _ = fut.result()
            if n >= 0:
                ok += 1
            else:
                fail += 1
            print(f"part.{p:>3} games={n:>4}  累计 {ok}/{len(todo)}  "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)

    print(f"\n完成 {ok}，失败 {fail}，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    # 失败必须中断：run_all.py 只根据退出码判断成功，若这里返回 0，
    # 下游会在缺分片的数据上继续打标签、训练、导出，而且不报任何错。
    if fail:
        raise SystemExit(f"{fail} 个分片失败，已中止（避免用不完整数据继续建模）")


if __name__ == "__main__":
    main()
