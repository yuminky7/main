"""「こうすれば変えられるかも」を推定する：説明会への参加が、次の四半期に段階を上げるか。

観察データからの推定は、交絡（新薬を早く採用する医師ほど説明会に参加し、段階も上がりやすい）で、
効果を大きく見積もりやすい。ここでは次を比べる。
1. 単純な比較（参加した医師と、しなかった医師の差）
2. 共変量で補正した推定（傾向スコアによる重み付け。日報から作った特徴量で補正する）
3. 医師ごとの効果（T-learner。医師単位で分けた交差適合で、学習に使った行では推定しない）

推定には医師単位のブートストラップで95%の区間を付ける。
架空データでは本当の効果が分かるので、どの推定がどれだけずれるかを確かめられる。
実データでは本当の効果は分からないため、推定は「試す施策を選ぶため」だけに使い、
効果の大きさとしては報告しない。設計書の「『こうすれば変えられる』の出し方」を参照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def naive_effect(df: pd.DataFrame, treat: str, outcome: str) -> float:
    return float(df.loc[df[treat], outcome].mean() - df.loc[~df[treat], outcome].mean())


def ipw_effect(df: pd.DataFrame, treat: str, outcome: str, cols: list[str]) -> float:
    """正規化した重み（Hajek型）での平均効果。傾向スコアは0.05〜0.95に切り詰める。"""
    X = df[cols].fillna(df[cols].median())
    ps_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, df[treat])
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.05, 0.95)
    t = df[treat].astype(float).values
    y = df[outcome].astype(float).values
    return float(np.sum(t * y / ps) / np.sum(t / ps) - np.sum((1 - t) * y / (1 - ps)) / np.sum((1 - t) / (1 - ps)))


def bootstrap_ci(df: pd.DataFrame, fn, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """医師単位のブートストラップで、fn(df) の95%区間を求める。"""
    rng = np.random.default_rng(seed)
    groups = {d: g for d, g in df.groupby("doctor_id")}
    ids = np.array(list(groups))
    vals = []
    for _ in range(n_boot):
        sample = pd.concat([groups[d] for d in rng.choice(ids, len(ids))], ignore_index=True)
        vals.append(fn(sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def t_learner_crossfit(df: pd.DataFrame, treat: str, outcome: str, cols: list[str], n_splits: int = 5) -> pd.Series:
    """医師ごとの効果の推定（参加した場合の確率 − しなかった場合の確率）。医師単位の交差適合。"""
    cate = pd.Series(np.nan, index=df.index, name="cate")
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=df["doctor_id"]):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        t = d_tr[treat].values
        m1 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        m0 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        m1.fit(d_tr.loc[t, cols], d_tr.loc[t, outcome].astype(int))
        m0.fit(d_tr.loc[~t, cols], d_tr.loc[~t, outcome].astype(int))
        cate.iloc[te] = m1.predict_proba(d_te[cols])[:, 1] - m0.predict_proba(d_te[cols])[:, 1]
    return cate
