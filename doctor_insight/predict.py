"""将来予測。時間で学習と評価を分け、単純な予測と比べる。

- 特徴量は四半期 t までの日報から作り、t+1 の結果を予測する。
- 学習に使う行は t <= fit_until_q（その結果は t+1 <= fit_until_q + 1）。
- 評価に使う行は t >= test_from_q。fit_until_q + 1 < test_from_q とし、学習の結果と評価の特徴量が重ならないようにする。
- AUCの差には、医師単位のブートストラップで95%の区間を付ける。区間が0をまたぐなら、差は誤差の範囲。
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


def _bootstrap_auc_diff(doctors: np.ndarray, y: np.ndarray, p: np.ndarray, p0: np.ndarray,
                        n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(doctors)
    idx_by_doc = {d: np.flatnonzero(doctors == d) for d in uniq}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_doc[d] for d in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], p0[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare_feature_sets(df: pd.DataFrame, target: str, feature_sets: dict[str, list[str]],
                         fit_until_q: int, test_from_q: int) -> pd.DataFrame:
    train = df[df["q_index"] <= fit_until_q]
    test = df[df["q_index"] >= test_from_q]
    y = test[target].astype(int).values
    docs = test["doctor_id"].values
    preds = {}
    for name, cols in feature_sets.items():
        m = _model().fit(train[cols], train[target].astype(int))
        preds[name] = m.predict_proba(test[cols])[:, 1]
    base_name = next(iter(feature_sets))
    rows = []
    for name, p in preds.items():
        row = {
            "特徴量": name, "評価の件数": len(test), "陽性の割合": round(float(y.mean()), 3),
            "AUC": round(float(roc_auc_score(y, p)), 3), "上位10%の的中率": round(top_decile_precision(y, p), 3),
        }
        if name != base_name:
            lo, hi = _bootstrap_auc_diff(docs, y, p, preds[base_name])
            row["基準とのAUCの差（95%区間）"] = f"{roc_auc_score(y, p) - roc_auc_score(y, preds[base_name]):+.3f}（{lo:+.3f}〜{hi:+.3f}）"
        else:
            row["基準とのAUCの差（95%区間）"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def fit_full(df: pd.DataFrame, target: str, cols: list[str], fit_until_q: int):
    train = df[df["q_index"] <= fit_until_q]
    return _model().fit(train[cols], train[target].astype(int))


def feature_importance(model, cols: list[str], top: int = 12) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame()
    return pd.DataFrame({"特徴量": cols, "重要度": imp}).sort_values("重要度", ascending=False).head(top)
