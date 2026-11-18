"""抽出結果を、医師×製品×四半期の特徴量にまとめる。

- 評価は、MRごとの書き方の癖（好意的に書く傾向）を引いてから集計する。
- 面談が少ない医師の値は、製品全体の平均に近づける（縮小推定）。
- 特徴量は、その四半期までの日報だけから作る（未来の情報を使わない）。
- 「他の製品」の特徴量は、対象の製品を除いた、同じ医師の他の自社品の日報から作る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BARRIERS, STAGE_ORDER, TOPICS

ATT_SCORE = {"肯定的": 1.0, "中立": 0.0, "否定的": -1.0}
WINDOW = 4  # 直近4四半期
PRIOR_K = 3.0  # 縮小推定の強さ（面談何件分を平均とみなすか）
TRAIT_BARRIERS = {"安全性への懸念": "safety", "エビデンス不足": "evidence", "患者負担": "price", "競合品で十分": "competitor"}


def mr_offsets(ext: pd.DataFrame, until_q: int, k: float = 20.0) -> pd.Series:
    """MRごとの「好意的に書く傾向」。学習期間の日報だけで計算する。"""
    d = ext[(ext["q_index"] <= until_q) & ext["score"].notna()]
    g = d.groupby("mr_id")["score"].agg(["mean", "count"])
    overall = d["score"].mean()
    return ((g["mean"] - overall) * g["count"] / (g["count"] + k)).rename("mr_offset")


def build_panel(reports: pd.DataFrame, extracted: pd.DataFrame, grid: pd.DataFrame, train_until_q: int) -> pd.DataFrame:
    """grid は doctor_id, product, q_index（と必要な列）を持つ、分析の単位の一覧。"""
    ext = reports[["report_id", "doctor_id", "product", "mr_id", "q_index"]].merge(extracted, on="report_id")
    ext["score"] = ext["attitude"].map(ATT_SCORE)
    off = mr_offsets(ext, train_until_q)
    ext = ext.merge(off, left_on="mr_id", right_index=True, how="left")
    ext["mr_offset"] = ext["mr_offset"].fillna(0.0)
    ext["adj_score"] = ext["score"] - ext["mr_offset"]
    ext["has_reaction"] = ext["score"].notna()
    for b in BARRIERS[:-1]:
        ext[f"b_{b}"] = ext["barriers"].str.split("|").apply(lambda v, b=b: b in v)
    for t in TOPICS:
        ext[f"t_{t}"] = ext["topics"].str.split("|").apply(lambda v, t=t: t in v) & ext["has_reaction"]
    ext["stage_num"] = ext["stage"].map(STAGE_ORDER)
    ext.loc[ext["stage"] == "中止", "stage_num"] = -1

    keys = ["doctor_id", "product", "q_index"]
    agg = ext.groupby(keys).agg(
        n_reports=("report_id", "count"), n_reaction=("has_reaction", "sum"),
        sum_adj=("adj_score", lambda s: s.fillna(0).sum()),
        **{f"n_b_{b}": (f"b_{b}", "sum") for b in BARRIERS[:-1]},
        **{f"n_t_{t}": (f"t_{t}", "sum") for t in TOPICS},
        stage_last=("stage_num", lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
    ).reset_index()

    panel = grid[keys].drop_duplicates().merge(agg, on=keys, how="left").sort_values(keys)
    count_cols = [c for c in agg.columns if c.startswith("n_") or c == "sum_adj"]
    panel[count_cols] = panel[count_cols].fillna(0)

    # 直近4四半期の合計
    g = panel.groupby(["doctor_id", "product"], sort=False)
    win = g[count_cols].rolling(WINDOW, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    panel[[f"w_{c}" for c in count_cols]] = win[count_cols].values

    # 製品ごとの平均評価（学習期間）を事前分布にして、縮小推定する
    prior = ext[ext["q_index"] <= train_until_q].groupby("product")["adj_score"].mean()
    panel["prior"] = panel["product"].map(prior).fillna(0)
    panel["att_mean"] = (panel["w_sum_adj"] + PRIOR_K * panel["prior"]) / (panel["w_n_reaction"] + PRIOR_K)
    panel["reaction_rate"] = panel["w_n_reaction"] / panel["w_n_reports"].replace(0, np.nan)
    for b in BARRIERS[:-1]:
        panel[f"barrier_{b}"] = panel[f"w_n_b_{b}"] / (panel["w_n_reaction"] + 1)
    for t in TOPICS:
        panel[f"topic_{t}"] = panel[f"w_n_t_{t}"] / (panel["w_n_reaction"] + 1)
    # 日報で最後に分かった処方の段階（過去から引き継ぐ）
    panel["stage_obs"] = g["stage_last"].ffill()
    panel["quarters_since_reaction"] = _quarters_since(panel, "n_reaction")

    panel = panel.merge(_other_product_traits(panel), on=keys, how="left")
    return panel


def _quarters_since(panel: pd.DataFrame, col: str) -> pd.Series:
    last = panel["q_index"].where(panel[col] > 0)
    last = last.groupby([panel["doctor_id"], panel["product"]]).ffill()
    return panel["q_index"] - last


def _other_product_traits(panel: pd.DataFrame) -> pd.DataFrame:
    """同じ医師の、他の自社品の日報から作る特性。その四半期までの累積。"""
    keys = ["doctor_id", "product", "q_index"]
    cols = ["n_reaction", "sum_adj"] + [f"n_b_{b}" for b in TRAIT_BARRIERS] + ["n_t_有効性"]
    own = panel[keys + cols].copy()
    own[cols] = own.groupby(["doctor_id", "product"])[cols].cumsum()
    # 同じ医師の全製品の累積（四半期ごと）。製品によって発売時期が違うので、四半期を揃えてから合計する
    wide = own.pivot_table(index=["doctor_id", "q_index"], columns="product", values=cols, aggfunc="sum")
    wide = wide.groupby(level=0).ffill().fillna(0)
    total = wide.T.groupby(level=0).sum().T  # 列＝cols
    total = total.reset_index()
    m = own.merge(total, on=["doctor_id", "q_index"], suffixes=("", "_all"))
    out = m[keys].copy()
    n_other = m["n_reaction_all"] - m["n_reaction"]
    out["other_n_reaction"] = n_other
    out["other_att_mean"] = (m["sum_adj_all"] - m["sum_adj"]) / (n_other + PRIOR_K)
    for b, name in TRAIT_BARRIERS.items():
        out[f"other_{name}"] = (m[f"n_b_{b}_all"] - m[f"n_b_{b}"]) / (n_other + 1)
    out["other_efficacy_focus"] = (m["n_t_有効性_all"] - m["n_t_有効性"]) / (n_other + 1)
    return out


FEATURES_TEXT = (
    ["att_mean", "reaction_rate", "stage_obs", "quarters_since_reaction"]
    + [f"barrier_{b}" for b in BARRIERS[:-1]]
    + [f"topic_{t}" for t in TOPICS]
)
FEATURES_OTHER = ["other_n_reaction", "other_att_mean", "other_efficacy_focus"] + [f"other_{n}" for n in TRAIT_BARRIERS.values()]
FEATURES_BASE = ["w_n_reports", "n_reports"]
