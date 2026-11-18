"""架空データで、抽出から予測と施策の効果の推定までを通して動かす。

    python -m doctor_insight.run_demo                 # ルールで抽出（GPUなしで動く）
    python -m doctor_insight.run_demo --llm-base-url http://localhost:8000/v1 --llm-model <モデル名> --llm-limit 2000

結果は outputs/report.md と outputs/*.csv に書き出す。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import extract_rules
from .evaluate import evaluate
from .features import FEATURES_BASE, FEATURES_OTHER, FEATURES_TEXT, build_panel
from .predict import compare_feature_sets, feature_importance, fit_full
from .segment import cluster_doctors, describe_clusters, rule_segments, segment_outcome
from .simulate import QUARTERS, simulate
from .topics import discover_topics, reaction_sentences
from .uplift import ipw_effect, naive_effect, t_learner

TRAIN_UNTIL_Q = 7  # 2024Q4 までで学習し、2025年で評価する


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "（なし）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-doctors", type=int, default=400)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-limit", type=int, default=None, help="LLMで抽出する日報の件数の上限（残りはルール）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: list[str] = ["# 試作の実行結果（架空データ）", ""]

    # 1. 架空データ -----------------------------------------------------------
    sim = simulate(n_doctors=args.n_doctors, seed=args.seed)
    for name in ["doctors", "reports", "states", "sales", "registrations"]:
        getattr(sim, name).to_csv(out / f"{name}.csv", index=False)
    rep += [f"医師 {len(sim.doctors)} 人、日報 {len(sim.reports):,} 件、期間 {QUARTERS[0]}〜{QUARTERS[-1]}。", ""]
    rep += ["日報の例：", ""] + [f"- {t}" for t in sim.reports["text"].sample(5, random_state=1)] + [""]

    # 2. 抽出 --------------------------------------------------------------------
    rows = [{"report_id": r.report_id, **_to_row(extract_rules.extract(r.text))} for r in sim.reports.itertuples()]
    extracted = pd.DataFrame(rows)
    method = "ルール"
    if args.llm_base_url:
        from openai import OpenAI

        from .extract_llm import extract_one, to_row
        client = OpenAI(base_url=args.llm_base_url, api_key="local")
        target = sim.reports if args.llm_limit is None else sim.reports.sample(args.llm_limit, random_state=0)
        llm_rows = {}
        for r in target.itertuples():
            s = extract_one(client, args.llm_model, r.text)
            if s is not None:
                llm_rows[r.report_id] = to_row(s)
        llm_df = pd.DataFrame.from_dict(llm_rows, orient="index").rename_axis("report_id").reset_index()
        extracted = pd.concat([llm_df, extracted[~extracted["report_id"].isin(llm_df["report_id"])]])
        method = f"LLM（{args.llm_model}、{len(llm_df):,}件）＋ルール"
    extracted.to_csv(out / "extracted.csv", index=False)

    # 3. 抽出の精度（人手のラベル500件を想定） ---------------------------------
    gold = sim.reports.sample(500, random_state=2)
    ev = evaluate(gold, extracted)
    rep += [f"## 1. 抽出の精度（{method}、人手のラベル500件と比較）", "", md(ev["summary"].round(3)), "",
            "評価の分類ごと：", "", md(ev["attitude_by_class"].round(3)), "", "ネックごと：", "", md(ev["barriers"].round(3)), ""]

    # 4. トピック抽出（フェーズ1） -------------------------------------------
    topics, _ = discover_topics(reaction_sentences(sim.reports.sample(min(20000, len(sim.reports)), random_state=3)))
    topics.to_csv(out / "topics.csv", index=False)
    rep += ["## 2. トピック抽出（観点の洗い出し）", "", md(topics[["topic", "share", "top_terms"]].round(3)), ""]

    # 5. 特徴量 ------------------------------------------------------------------
    grid = sim.states.copy()
    panel = build_panel(sim.reports, extracted, grid, TRAIN_UNTIL_Q)
    panel = panel.merge(grid, on=["doctor_id", "product", "q_index"])
    nxt = sim.registrations.assign(q_index=lambda d: d["quarter"].map({q: i for i, q in enumerate(QUARTERS)}) - 1)
    panel = panel.merge(nxt[["doctor_id", "product", "q_index", "new_registrations"]].rename(
        columns={"new_registrations": "reg_next"}), on=["doctor_id", "product", "q_index"], how="left")
    panel["y_reg_next"] = panel["reg_next"] > 0
    panel["stage_obs_next"] = panel.groupby(["doctor_id", "product"])["stage_obs"].shift(-1)
    panel["y_stage_up_obs"] = panel["stage_obs_next"] > panel["stage_obs"]
    panel = panel[panel["q_index"] < len(QUARTERS) - 1]  # 次の四半期がある行だけ
    panel.to_csv(out / "panel.csv", index=False)

    sets = {
        "基準：日報の段階と面談回数だけ": ["stage_obs"] + FEATURES_BASE,
        "＋自由記載の評価・ネック": FEATURES_TEXT + FEATURES_BASE,
        "＋他の自社品の情報": FEATURES_TEXT + FEATURES_BASE + FEATURES_OTHER,
    }

    # 6. 予測①：製品Aの症例登録（医師ごと） -----------------------------------
    a = panel[panel["product"] == "A"]
    res_a = compare_feature_sets(a, "y_reg_next", sets, TRAIN_UNTIL_Q)
    model_a = fit_full(a, "y_reg_next", sets["＋他の自社品の情報"], TRAIN_UNTIL_Q)
    rep += ["## 3. 予測：次の四半期に製品Aの症例登録があるか（医師ごと）", "", md(res_a), "",
            "重要な特徴量：", "", md(feature_importance(model_a, sets["＋他の自社品の情報"])), ""]

    # 7. 予測②：新製品Cで、日報上の段階が上がるか --------------------------------
    c = panel[(panel["product"] == "C") & panel["stage_obs"].notna()]
    res_c = compare_feature_sets(c, "y_stage_up_obs", sets, TRAIN_UNTIL_Q)
    rep += ["## 4. 予測：新製品Cで、次の四半期に日報上の段階が上がるか（医師ごと）", "", md(res_c), "",
            "製品Cは2024Q1発売。日報が少ないため、他の自社品の情報が効くかを見る。", ""]

    # 8. カテゴリ分け（2025Q3時点、製品A） --------------------------------------
    last = a[a["q_index"] == len(QUARTERS) - 2].copy()
    last["segment"] = rule_segments(last)
    seg_tab = segment_outcome(last, last["segment"], "y_reg_next").reset_index()
    cluster_cols = ["att_mean", "barrier_安全性への懸念", "barrier_患者負担", "barrier_投与の手間",
                    "other_evidence", "other_competitor", "other_att_mean", "reaction_rate"]
    labels, profile = cluster_doctors(last, cluster_cols)
    profile = profile.join(describe_clusters(profile))
    last["cluster"] = labels
    profile["次の四半期に登録あり"] = last.groupby("cluster")["y_reg_next"].mean().round(3)
    last[["doctor_id", "segment", "cluster"]].to_csv(out / "segments_A_2025Q3.csv", index=False)
    rep += ["## 5. 医師のカテゴリ（製品A、2025Q3時点）", "", "ルールによるカテゴリ（上位10）と、次の四半期に登録があった割合：", "",
            md(seg_tab.head(10)), "", "クラスタリング（k=5）：", "", profile.reset_index().to_markdown(index=False), ""]

    # 9. 施策の効果：説明会への参加 --------------------------------------------
    u = panel[(panel["q_index"] <= TRAIN_UNTIL_Q) & panel["can_move"]].copy()
    u["seminar"] = u["seminar"].astype(bool)
    true_ate = float((u["p_up_if_seminar"] - u["p_up_if_not"]).mean())
    cols = FEATURES_TEXT + FEATURES_BASE + FEATURES_OTHER
    est = pd.DataFrame([
        {"推定の方法": "本当の効果（架空データの答え）", "段階が上がる確率の差": true_ate},
        {"推定の方法": "単純な比較（参加した医師 − しなかった医師）", "段階が上がる確率の差": naive_effect(u, "seminar", "moved_up_next")},
        {"推定の方法": "日報の特徴量で補正（傾向スコアの重み付け）", "段階が上がる確率の差": ipw_effect(u, "seminar", "moved_up_next", cols)},
    ]).round(3)
    u["cate"] = t_learner(u, "seminar", "moved_up_next", [c for c in cols])
    u["true_cate"] = u["p_up_if_seminar"] - u["p_up_if_not"]
    u["日報上のエビデンス重視"] = pd.qcut(u["other_efficacy_focus"].rank(method="first"), 3, labels=["低", "中", "高"])
    het = u.groupby("日報上のエビデンス重視", observed=True)[["cate", "true_cate"]].mean().round(3).reset_index()
    het.columns = ["日報上のエビデンス重視（他の自社品の日報から）", "推定した効果", "本当の効果"]
    rep += ["## 6. 施策の効果の推定：説明会への参加が、次の四半期に段階を上げるか", "",
            "結果は架空データで本当の段階を使ったもの。実データでは、症例登録や日報上の段階で代用する。", "",
            md(est), "", "医師の特性ごとの効果：", "", md(het), "",
            "単純な比較は、関心の高い医師ほど説明会に参加するため、効果を大きく見積もる。"
            "補正しても、日報に書かれていない違いは残る。推定は仮説として扱い、訪問計画で試して確かめる。", ""]

    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


def _to_row(s) -> dict:
    return {"topics": "|".join(s.topics), "attitude": s.attitude, "stage": s.stage,
            "barriers": "|".join(s.barriers), "mentions_patient": s.mentions_patient, "evidence": s.evidence}


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
