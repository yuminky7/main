"""架空データで、抽出から予測と施策の効果の推定までを通して動かす。

    python -m doctor_insight.run_demo                 # ルールで抽出（GPUなしで動く）
    python -m doctor_insight.run_demo --seed 1        # 架空データを変えて結果の揺れを見る
    python -m doctor_insight.run_demo --llm-base-url http://localhost:8000/v1 --llm-model <モデル名> --llm-limit 500

結果は outputs/report.md と outputs/*.csv に書き出す。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import extract_rules
from .evaluate import evaluate
from .extract_llm import FAILED_ROW, to_row
from .features import FEATURES_BASE, FEATURES_OTHER, FEATURES_TEXT, build_panel
from .predict import compare_feature_sets, feature_importance, fit_full
from .segment import cluster_doctors, describe_clusters, rule_segments, segment_outcome
from .simulate import QUARTERS, simulate
from .topics import discover_topics, reaction_sentences
from .uplift import bootstrap_ci, ipw_effect, naive_effect, t_learner_crossfit

# 学習：特徴量の四半期 t <= 6（結果は t+1 <= 7 = 2024Q4）。評価：t >= 8（2025Q1〜）。
# MRの補正と製品の平均も t <= 6 の日報だけで作る。
FIT_UNTIL_Q = 6
TEST_FROM_Q = 8


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "（なし）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-doctors", type=int, default=400)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-limit", type=int, default=500, help="LLMで抽出する日報の件数（評価用の500件から順に）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: list[str] = [f"# 試作の実行結果（架空データ、seed={args.seed}）", ""]

    # 1. 架空データ -----------------------------------------------------------
    sim = simulate(n_doctors=args.n_doctors, seed=args.seed)
    for name in ["doctors", "reports", "states", "sales", "registrations"]:
        getattr(sim, name).to_csv(out / f"{name}.csv", index=False)
    rep += [f"医師 {len(sim.doctors)} 人、日報 {len(sim.reports):,} 件、期間 {QUARTERS[0]}〜{QUARTERS[-1]}。", ""]
    rep += ["日報の例：", ""] + [f"- {t}" for t in sim.reports["text"].sample(5, random_state=1)] + [""]

    # 2. 抽出 --------------------------------------------------------------------
    extracted = pd.DataFrame([{"report_id": r.report_id, **to_row(extract_rules.extract(r.text))}
                              for r in sim.reports.itertuples()])
    gold = sim.reports.sample(500, random_state=2)  # 人手のラベル500件を想定
    ev = evaluate(gold, extracted)
    rep += ["## 1. 抽出の精度（ルール、人手のラベル500件と比較）", "",
            "架空の日報は決まった文型から作っているため、実データよりかなり高く出る。", "",
            md(ev["summary"].round(3)), "", "評価の分類ごと：", "", md(ev["attitude_by_class"].round(3)), "",
            "ネックごと：", "", md(ev["barriers"].round(3)), ""]

    if args.llm_base_url:
        from openai import OpenAI

        from .extract_llm import extract_one
        client = OpenAI(base_url=args.llm_base_url, api_key="local", timeout=120)
        target = gold.head(args.llm_limit)
        llm_rows = []
        for r in target.itertuples():
            s = extract_one(client, args.llm_model, r.text)
            llm_rows.append({"report_id": r.report_id, **(to_row(s) if s else FAILED_ROW)})
        llm_df = pd.DataFrame(llm_rows)
        ev_llm = evaluate(target, llm_df)  # LLMで抽出した日報だけで評価する。失敗は誤りとして数える
        rep += [f"### LLM（{args.llm_model}）の精度：人手のラベル{len(target)}件", "",
                md(ev_llm["summary"].round(3)), ""]
        ok = llm_df[llm_df["attitude"] != "抽出失敗"]
        extracted = pd.concat([ok, extracted[~extracted["report_id"].isin(ok["report_id"])]])
    extracted.to_csv(out / "extracted.csv", index=False)

    # 3. トピック抽出（フェーズ1） -------------------------------------------
    topics, _ = discover_topics(reaction_sentences(sim.reports.sample(min(20000, len(sim.reports)), random_state=3)))
    topics.to_csv(out / "topics.csv", index=False)
    rep += ["## 2. トピック抽出（観点の洗い出し）", "",
            "患者の話題のトピックが出た場合は、分析から外す対象の候補になる。", "",
            md(topics[["topic", "share", "top_terms"]].round(3)), ""]

    # 4. 特徴量 ------------------------------------------------------------------
    grid = sim.states.copy()
    panel = build_panel(sim.reports, extracted, grid, FIT_UNTIL_Q)
    panel = panel.merge(grid, on=["doctor_id", "product", "q_index"])
    nxt = sim.registrations.assign(q_index=lambda d: d["quarter"].map({q: i for i, q in enumerate(QUARTERS)}) - 1)
    panel = panel.merge(nxt[["doctor_id", "product", "q_index", "new_registrations"]].rename(
        columns={"new_registrations": "reg_next"}), on=["doctor_id", "product", "q_index"], how="left")
    panel["y_reg_next"] = panel["reg_next"] > 0
    g = panel.groupby(["doctor_id", "product"])
    panel["stage_obs_next"] = g["stage_obs"].shift(-1)
    panel["stage_mentioned_next"] = g["stage_mentioned"].shift(-1)
    panel["y_stage_up_obs"] = panel["stage_obs_next"] > panel["stage_obs"]
    panel = panel[panel["q_index"] < len(QUARTERS) - 1]  # 次の四半期がある行だけ
    panel.to_csv(out / "panel.csv", index=False)

    sets = {
        "基準：日報の段階と面談回数だけ": ["stage_obs", "stage_age"] + FEATURES_BASE,
        "＋自由記載の評価・ネック": FEATURES_TEXT + FEATURES_BASE,
        "＋他の自社品の情報": FEATURES_TEXT + FEATURES_BASE + FEATURES_OTHER,
    }

    # 5. 予測①：製品Aの症例登録（医師ごと、MRが書かない結果） ------------------
    a = panel[panel["product"] == "A"]
    res_a = compare_feature_sets(a, "y_reg_next", sets, FIT_UNTIL_Q, TEST_FROM_Q)
    model_a = fit_full(a, "y_reg_next", sets["＋他の自社品の情報"], FIT_UNTIL_Q)
    rep += ["## 3. 予測：次の四半期に製品Aの症例登録があるか（医師ごと）", "",
            "AUCの差の区間は医師単位のブートストラップ。区間が0をまたぐ場合は、差は誤差の範囲。", "",
            md(res_a), "", "重要な特徴量：", "", md(feature_importance(model_a, sets["＋他の自社品の情報"])), ""]

    # 6. 予測②：新製品Cで、日報上の段階が上がるか（練習用） ----------------------
    # 次の四半期に段階が書かれた行だけを使う（古い段階が書き直されただけの「上昇」を減らすため）
    # その四半期にも段階が書かれた行に限る（古い段階を引き継いだだけの行を除く）
    c = panel[(panel["product"] == "C") & panel["stage_obs"].notna() & (panel["stage_mentioned_next"] == 1)
              & (panel["stage_age"] == 0)]
    res_c = compare_feature_sets(c, "y_stage_up_obs", sets, FIT_UNTIL_Q, TEST_FROM_Q)
    rep += ["## 4. 予測：新製品Cで、次の四半期に日報上の段階が上がるか（練習用）", "",
            "日報上の段階は、説明変数も結果も同じMRの記述。予測できても、MRの書き方を予測しているだけかもしれない。"
            "予測力の判断には使わない。", "", md(res_c), ""]

    # 7. カテゴリ分け（最後の四半期の1つ前、製品A） ------------------------------
    last = a[a["q_index"] == len(QUARTERS) - 2].copy()
    last["segment"] = rule_segments(last)
    seg_tab = segment_outcome(last, last["segment"], "y_reg_next").reset_index()
    cluster_cols = ["att_mean", "barrier_安全性への懸念", "barrier_患者負担", "barrier_投与の手間",
                    "other_evidence", "other_competitor", "other_att_mean", "other_efficacy_focus"]
    labels, profile = cluster_doctors(last, cluster_cols)
    profile = profile.join(describe_clusters(profile))
    last["cluster"] = labels
    profile["次の四半期に登録あり"] = last.groupby("cluster")["y_reg_next"].mean().round(3)
    last[["doctor_id", "segment", "cluster"]].to_csv(out / "segments_A.csv", index=False)
    rep += [f"## 5. 医師のカテゴリ（製品A、{QUARTERS[len(QUARTERS) - 2]}時点）", "",
            "ルールによるカテゴリ（上位10）と、次の四半期に登録があった割合：", "",
            md(seg_tab.head(10)), "", "クラスタリング（k=5、「高・低」は医師全体の平均との比較）：", "",
            profile.reset_index().to_markdown(index=False), ""]

    # 8. 施策の効果：説明会への参加 --------------------------------------------
    # 架空データの答え（can_move、moved_up_next）を使う。実データでは、対象の条件と結果を
    # 日報上の段階や症例登録から作り直す必要がある（README を参照）。
    u = panel[(panel["q_index"] <= FIT_UNTIL_Q) & panel["can_move"]].copy().reset_index(drop=True)
    u["seminar"] = u["seminar"].astype(bool)
    cols = FEATURES_TEXT + FEATURES_BASE + FEATURES_OTHER
    true_ate = float((u["p_up_if_seminar"] - u["p_up_if_not"]).mean())
    naive = naive_effect(u, "seminar", "moved_up_next")
    ipw = ipw_effect(u, "seminar", "moved_up_next", cols)
    naive_ci = bootstrap_ci(u, lambda d: naive_effect(d, "seminar", "moved_up_next"))
    ipw_ci = bootstrap_ci(u, lambda d: ipw_effect(d, "seminar", "moved_up_next", cols), n_boot=100)
    est = pd.DataFrame([
        {"推定の方法": "本当の効果（架空データの答え）", "段階が上がる確率の差": f"{true_ate:.3f}"},
        {"推定の方法": "単純な比較（参加した医師 − しなかった医師）", "段階が上がる確率の差": f"{naive:.3f}（{naive_ci[0]:.3f}〜{naive_ci[1]:.3f}）"},
        {"推定の方法": "日報の特徴量で補正（傾向スコアの重み付け）", "段階が上がる確率の差": f"{ipw:.3f}（{ipw_ci[0]:.3f}〜{ipw_ci[1]:.3f}）"},
    ])
    u["cate"] = t_learner_crossfit(u, "seminar", "moved_up_next", cols)
    u["true_cate"] = u["p_up_if_seminar"] - u["p_up_if_not"]
    u = u.merge(sim.doctors[["doctor_id", "evidence"]], on="doctor_id")
    r_proxy = float(u[["other_efficacy_focus", "evidence"]].corr().iloc[0, 1])
    r_cate = float(u[["cate", "true_cate"]].corr().iloc[0, 1])
    u["日報上のエビデンス重視"] = pd.cut(u["other_efficacy_focus"].rank(pct=True, method="average"),
                                 [0, 1 / 3, 2 / 3, 1], labels=["低", "中", "高"], include_lowest=True)
    het = u.groupby("日報上のエビデンス重視", observed=True)[["cate", "true_cate"]].agg(["mean", "count"]).round(3)
    het.columns = ["推定した効果", "件数", "本当の効果", "_"]
    het = het.drop(columns="_").reset_index()
    rep += ["## 6. 施策の効果の推定：説明会への参加が、次の四半期に段階を上げるか", "",
            "括弧内は医師単位のブートストラップによる95%区間。", "", md(est), "",
            f"医師ごとの効果（T-learner、医師単位の交差適合）と本当の効果の相関：{r_cate:.2f}。"
            f"日報から作った「エビデンス重視」の指標と、医師の本当の特性との相関：{r_proxy:.2f}。", "",
            "医師の特性ごとの効果（他の自社品の日報から作った指標で3群に分けた）：", "", md(het), "",
            "日報の特徴量で補正しても、日報に書かれない違い（新薬を早く採用する傾向）が残るため、効果は大きめに出やすい。"
            "推定は試す施策を選ぶためだけに使い、訪問計画で確かめる。", ""]

    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
