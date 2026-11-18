"""抽出の精度を、人手のラベル（ここでは架空データの答え）と比べて測る。

実データでは、300〜500件に2人以上で人手のラベルを付け、その一部をここに渡す。
評価に使う日報は、プロンプトやルールの調整に使ってはいけない。
抽出に失敗した日報（結果がない、または「抽出失敗」）は、誤りとして数える。
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

FAILED = "抽出失敗"


def _multilabel_scores(gold: pd.Series, pred: pd.Series, empty: str) -> pd.DataFrame:
    # 予測にだけ出るラベルも含める（金のラベルにない誤検知を見落とさないため）
    labels = sorted({x for v in pd.concat([gold, pred]) for x in v.split("|")} - {empty, FAILED})
    rows = []
    for lab in labels:
        g = gold.str.split("|").apply(lambda v: lab in v)
        p = pred.str.split("|").apply(lambda v: lab in v)
        tp = int((g & p).sum())
        rows.append({
            "label": lab, "support": int(g.sum()),
            "precision": tp / max(int(p.sum()), 1), "recall": tp / max(int(g.sum()), 1),
        })
    return pd.DataFrame(rows)


def evaluate(gold: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """gold は gold_* の列、pred は抽出結果の列を持つ。report_id で結合する。"""
    df = gold.merge(pred, on="report_id", how="left")
    n_failed = int(df["attitude"].isna().sum() + (df["attitude"] == FAILED).sum())
    for c in ["topics", "attitude", "stage", "barriers"]:
        df[c] = df[c].fillna(FAILED)
    df["mentions_patient"] = df["mentions_patient"].fillna(False).astype(bool)
    out = {}
    summary = []
    for field in ["attitude", "stage"]:
        g, p = df[f"gold_{field}"], df[field]
        summary.append({
            "field": field, "n": len(df), "failed": n_failed, "accuracy": float((g == p).mean()),
            "kappa": float(cohen_kappa_score(g, p)),
        })
        labels = sorted(g.unique())
        prec, rec, _, sup = precision_recall_fscore_support(g, p, labels=labels, zero_division=0)
        out[f"{field}_by_class"] = pd.DataFrame({"label": labels, "support": sup, "precision": prec, "recall": rec})
    g, p = df["gold_mentions_patient"].astype(bool), df["mentions_patient"]
    summary.append({"field": "mentions_patient", "n": len(df), "failed": n_failed,
                    "accuracy": float((g == p).mean()), "kappa": float(cohen_kappa_score(g, p))})
    out["summary"] = pd.DataFrame(summary)
    out["barriers"] = _multilabel_scores(df["gold_barriers"], df["barriers"], "なし")
    out["topics"] = _multilabel_scores(df["gold_topics"], df["topics"], "その他")
    return out
