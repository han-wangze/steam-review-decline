"""全局配置：路径、时间窗、标签口径。

所有口径常量集中在这里，避免散落在脚本里各写一套。
"""
from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WINDOWS_DIR = DATA / "windows"        # 每个分片的聚合结果
OUT = ROOT / "out"
SITE = ROOT / "site"

# 发行日期小表（两列，约 3MB，随仓库一起分发）。
# 由 src/fetch_meta.py 从完整元数据抽取而来，不依赖本机路径。
META_PARQUET = DATA / "game_release_dates.parquet"
META_HF_URL = ("https://huggingface.co/datasets/FronkonGames/steam-games-dataset"
               "/resolve/main/data/train-00000-of-00001.parquet")

# ---- 数据源 ----
HF_PART_URL = "https://huggingface.co/datasets/adde023/steam-reviews/resolve/main/part.{}.parquet"
PART_STEP = 3                          # 步长 3 → 221 个分片
# 数据集实有 part.0–662（part.663/664 返回 404），所以上界写 663 而不是 665。
# 用 665 会让 --refresh 每次都去请求一个不存在的分片。
PARTS = list(range(0, 663, PART_STEP))
COLS = [
    "appid", "game", "timestamp_created", "voted_up",
    "author_playtime_at_review", "author_num_games_owned",
    "language", "review",
]

# ---- 时间口径 ----
# 数据快照：HF 数据集发布时间。所有"观察期是否完整"的判断都以它为准。
# 用它而不是"最后一条评论时间"——后者会把"没人评论"误算成"观察期短"。
SNAPSHOT_TS = 1698883200               # 2023-11-02 UTC
DAY = 86400

EARLY_WINDOW_D = 30                    # 特征窗口
LABEL_WINDOW_D = 365                   # 标签窗口（固定，不随数据浮动）
SENSITIVITY_WINDOWS_D = [7, 30, 90, 180, 365]

# 曲线：每 30 天一个累计点，共 13 个（0–390 天），供前端画早期走势
CURVE_STEP_D = 30
CURVE_POINTS = 13

# ---- 样本口径 ----
MIN_EARLY_REVIEWS = 10                 # 特征窗内至少 10 条，否则好评率不可信
MIN_FINAL_REVIEWS = 100                # 365 天窗口内至少 100 条，否则档位判不出来
MIN_LATE_REVIEWS = 50                  # 第 30–365 天至少 50 条，否则"后期档位"不可信
MAX_T0_RELEASE_LAG_D = 30              # 首条评论与发行日相差过大 → 不是"上线期"，剔除

# Steam 官方评价档位（按好评率 + 评论量阈值）
GOOD_TIERS = {"overwhelmingly_positive", "very_positive", "positive", "mostly_positive"}

# 已知的跨分片脏数据（appid 730 = CS:GO/CS2，评论跨多个 part，
# 每片只看到一部分，t0 与总数都不对）。样本里直接排除。
EXCLUDED_APPIDS = {730}
