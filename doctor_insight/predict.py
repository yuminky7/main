"""将来予測。時間で学習と評価を分け、単純な予測と比べる。

- 特徴量は四半期 t までの日報から作り、t+1 の結果を予測する。
- 学習は t <= train_until_q、評価はそれより後の四半期。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:  # LightGBM が無ければ scikit-learn で代用する
    lgb = None
from sklearn.ensemble import HistGradientBoostingClassifier


def _model():
    if lgb is not None:
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def top_decile_precision(y: np.ndarray, p: np.ndarray) -> float:
    n = max(int(len(p) * 0.1), 1)
    return float(np.asarray(y)[np.argsort(-p)[:n]].mean())


def compare_feature_sets(df: pd.DataFrame, target: str, feature_sets: dict[str, list[str]], train_until_q: int) -> pd.DataFrame:
    train = df[df["q_index"] <= train_until_q]
    test = df[df["q_index"] > train_until_q]
    rows = []
    for name, cols in feature_sets.items():
        m = _model().fit(train[cols], train[target].astype(int))
        p = m.predict_proba(test[cols])[:, 1]
        y = test[target].astype(int).values
        rows.append({
            "特徴量": name, "評価の件数": len(test), "陽性の割合": round(float(y.mean()), 3),
            "AUC": round(float(roc_auc_score(y, p)), 3),
            "上位10%の的中率": round(top_decile_precision(y, p), 3),
        })
    return pd.DataFrame(rows)


def fit_full(df: pd.DataFrame, target: str, cols: list[str], train_until_q: int):
    train = df[df["q_index"] <= train_until_q]
    return _model().fit(train[cols], train[target].astype(int))


def feature_importance(model, cols: list[str], top: int = 12) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame()
    return pd.DataFrame({"特徴量": cols, "重要度": imp}).sort_values("重要度", ascending=False).head(top)
