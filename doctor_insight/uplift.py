"""「こうすれば変えられるかも」を推定する：説明会への参加が、次の四半期に段階を上げるか。

観察データからの推定は、交絡（関心の高い医師ほど説明会に参加する）で効果を大きく見積もりやすい。
ここでは3つを比べる。
1. 単純な比較（参加した医師と、しなかった医師の差）
2. 共変量で補正した推定（傾向スコアによる重み付け。日報から作った特徴量で補正する）
3. 医師のカテゴリごとの効果（T-learner）

架空データでは本当の効果が分かるので、どの推定がどれだけずれるかを確かめられる。
実データでは本当の効果は分からないため、推定は仮説として扱い、訪問計画で試して確かめる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def naive_effect(df: pd.DataFrame, treat: str, outcome: str) -> float:
    return float(df.loc[df[treat], outcome].mean() - df.loc[~df[treat], outcome].mean())


def ipw_effect(df: pd.DataFrame, treat: str, outcome: str, cols: list[str]) -> float:
    X = df[cols].fillna(df[cols].median())
    ps_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, df[treat])
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.05, 0.95)
    t = df[treat].astype(float).values
    y = df[outcome].astype(float).values
    return float(np.mean(t * y / ps) / np.mean(t / ps) - np.mean((1 - t) * y / (1 - ps)) / np.mean((1 - t) / (1 - ps)))


def t_learner(df: pd.DataFrame, treat: str, outcome: str, cols: list[str]) -> pd.Series:
    """医師ごとの効果の推定（参加した場合の確率 − しなかった場合の確率）。"""
    X = df[cols]
    m1 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    m0 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    t = df[treat].values
    m1.fit(X[t], df.loc[t, outcome].astype(int))
    m0.fit(X[~t], df.loc[~t, outcome].astype(int))
    return pd.Series(m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1], index=df.index, name="cate")
