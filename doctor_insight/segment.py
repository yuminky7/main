"""医師のカテゴリ分け。

まず説明しやすいルールで分け、次にクラスタリングでルールでは見えないまとまりがないかを見る。
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .schema import BARRIERS

STAGE_NAME = {-1: "中止", 0: "認知", 1: "関心", 2: "試用", 3: "継続"}


def rule_segments(panel_q: pd.DataFrame, min_reaction: int = 2) -> pd.Series:
    """1つの四半期の医師×製品について、「処方の段階×主なネック」でカテゴリを付ける。"""
    barrier_cols = [f"barrier_{b}" for b in BARRIERS[:-1]]
    main = panel_q[barrier_cols].idxmax(axis=1).str.replace("barrier_", "", regex=False)
    main = main.where(panel_q[barrier_cols].max(axis=1) > 0.15, "ネック不明")
    stage = panel_q["stage_obs"].map(STAGE_NAME).fillna("段階不明")
    seg = stage + "×" + main
    return seg.where(panel_q["w_n_reaction"] >= min_reaction, "日報に評価がほぼない")


def cluster_doctors(panel_q: pd.DataFrame, cols: list[str], k: int = 5, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    X = panel_q[cols].fillna(panel_q[cols].median())
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    labels = pd.Series(km.labels_, index=panel_q.index, name="cluster")
    means = X.groupby(labels).mean()
    # 「高・低」は、クラスタの平均どうしではなく、医師全体の平均と標準偏差を基準にする
    z = (means - X.mean()) / (X.std() + 1e-9)
    profile = means.round(2)
    profile.insert(0, "n", labels.value_counts().sort_index())
    profile.attrs["z"] = z
    return labels, profile


def describe_clusters(profile: pd.DataFrame) -> pd.Series:
    """各クラスタの特徴を、医師全体の平均から最も離れた特徴量3つで言葉にする。"""
    z = profile.attrs["z"]
    names = {}
    for c, row in z.iterrows():
        top = row.abs().sort_values(ascending=False).index[:3]
        names[c] = "、".join(f"{t}{'高' if row[t] > 0 else '低'}" for t in top)
    return pd.Series(names, name="特徴")


def segment_outcome(panel_q: pd.DataFrame, seg: pd.Series, outcome: str) -> pd.DataFrame:
    df = pd.DataFrame({"segment": seg, "outcome": panel_q[outcome].astype(float)})
    return df.groupby("segment")["outcome"].agg(["count", "mean"]).sort_values("count", ascending=False).round(3)

