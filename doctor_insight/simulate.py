"""架空のMR日報、施設の売上、投与症例登録を作る。

実データの代わりに、パイプライン全体を動かして確かめるためのもの。
医師ごとの隠れた特性（エビデンス重視、安全性への懸念、新薬採用の早さ、患者負担への敏感さ）と、
MRごとの書き方の癖（好意的に書く傾向、反応を書く頻度）から日報を作る。
答え（本当の評価、本当の段階、説明会の本当の効果）が分かっているので、抽出や推定のずれを測れる。

医師・施設・製品はすべて架空。製品名は「製品A」などで、実在の製品とは関係ない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUARTERS = [f"{y}Q{q}" for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)]

PRODUCTS = {
    # 製品ID: (表示名, 疾患, 発売した四半期の番号, 症例登録の有無)
    "A": ("製品A", "乾癬", 0, True),
    "B": ("製品B", "アトピー性皮膚炎", 0, False),
    "C": ("製品C", "ざ瘡", 4, False),  # 2024Q1発売。発売後の採用の早さを見るため
}

STAGES = ["認知", "関心", "試用", "継続"]

TOPIC_PHRASES = {
    "有効性": ["有効性", "臨床成績", "効果のデータ"],
    "安全性": ["安全性", "副作用の情報", "安全性データ"],
    "使用感": ["使用感", "塗り心地", "剤形"],
    "投与方法": ["投与方法", "投与間隔", "用法"],
    "薬価・患者負担": ["薬価", "患者負担", "費用面"],
    "競合品": ["他剤との違い", "既存治療との比較"],
}

# 評価×話題ごとの、日報に書かれる反応の文
REACTION = {
    ("肯定的", "有効性"): ["効果に満足されている", "効果は期待どおりとの評価", "データを見て効果に納得された"],
    ("否定的", "有効性"): ["効果がまだ物足りないとのこと", "効果に疑問を持たれている"],
    ("中立", "有効性"): ["効果については様子見とのこと", "効果は症例を見て判断したいとのこと"],
    ("肯定的", "安全性"): ["安全性は特に問題ないとのこと", "副作用の心配は少ないとの評価"],
    ("否定的", "安全性"): ["安全性が気になるとのこと", "感染症のリスクを心配されている"],
    ("中立", "安全性"): ["安全性は引き続き確認したいとのこと"],
    ("肯定的", "使用感"): ["使用感が良いと好評", "患者さんにも塗りやすいと好評とのこと"],
    ("否定的", "使用感"): ["べたつきが気になるとのこと", "使用感に不満があるとのこと"],
    ("中立", "使用感"): ["使用感は他剤と変わらないとのこと"],
    ("肯定的", "投与方法"): ["投与間隔が長いのは助かるとのこと"],
    ("否定的", "投与方法"): ["投与の手間が負担とのこと", "自己注射の指導が大変とのこと"],
    ("中立", "投与方法"): ["投与方法は理解されたとのこと"],
    ("肯定的", "薬価・患者負担"): ["費用面は問題ないとのこと"],
    ("否定的", "薬価・患者負担"): ["患者負担が大きいのがネックとのこと", "高額なので勧めにくいとのこと"],
    ("中立", "薬価・患者負担"): ["費用面は患者さんと相談するとのこと"],
    ("肯定的", "競合品"): ["他剤より使いやすいとの評価"],
    ("否定的", "競合品"): ["今の治療で十分とのこと", "既存の薬で困っていないとのこと"],
    ("中立", "競合品"): ["他剤と比較して検討したいとのこと"],
}

BARRIER_SENTENCE = {
    "安全性への懸念": ["長期の安全性データが出るまで待ちたいとのこと"],
    "エビデンス不足": ["日本人のデータがもっと欲しいとのこと", "エビデンスが不十分と感じておられる"],
    "患者負担": ["患者さんの費用負担がネック"],
    "投与の手間": ["院内の体制的に投与の手間がネック"],
    "競合品で十分": ["現状の治療で困っていないため切り替える理由がないとのこと"],
    "対象患者がいない": ["対象となる患者さんが今はいないとのこと"],
}

STAGE_SENTENCE = {
    "認知": ["まだ使用経験はないとのこと"],
    "関心": ["興味を示された", "使ってみたいとのこと"],
    "試用": ["数例で使い始めたとのこと", "初めて処方いただいた"],
    "継続": ["継続して処方いただいている", "第一選択として使っているとのこと"],
    "中止": ["使用を中止したとのこと", "今は処方を控えているとのこと"],
}

# 否定的な反応の文のうち、ネックそのものを述べているもの（人手のラベルでもネックとして数える）
REACTION_BARRIER = {
    ("否定的", "薬価・患者負担"): "患者負担",
    ("否定的", "投与方法"): "投与の手間",
    ("否定的", "競合品"): "競合品で十分",
}

# エビデンスを重視する医師ほど、日報に書かれやすい文
EVIDENCE_SEEKING_SENTENCE = ["論文のデータを詳しく質問された", "エビデンスの根拠を求められた"]

PATIENT_SENTENCE = ["70代の患者さんで改善が見られたとのこと", "若い患者さんから相談があったとのこと"]

# 製品ごとに、出やすいネック（重み）
PRODUCT_BARRIER_WEIGHTS = {
    "A": {"安全性への懸念": 3, "患者負担": 3, "投与の手間": 2, "エビデンス不足": 1, "競合品で十分": 1, "対象患者がいない": 1},
    "B": {"競合品で十分": 3, "エビデンス不足": 1, "患者負担": 1, "安全性への懸念": 1, "対象患者がいない": 1},
    "C": {"競合品で十分": 2, "エビデンス不足": 2, "患者負担": 1, "安全性への懸念": 1},
}

BARRIER_TOPIC = {
    "安全性への懸念": "安全性",
    "エビデンス不足": "有効性",
    "患者負担": "薬価・患者負担",
    "投与の手間": "投与方法",
    "競合品で十分": "競合品",
    "対象患者がいない": "有効性",
}


@dataclass
class SimData:
    doctors: pd.DataFrame  # 医師の属性と、隠れた特性（答え）
    reports: pd.DataFrame  # 日報（自由記載と、構造化項目と、答えのラベル）
    states: pd.DataFrame  # 医師×製品×四半期の本当の段階と、説明会への参加
    sales: pd.DataFrame  # 施設×製品×四半期の売上
    registrations: pd.DataFrame  # 医師×四半期の製品Aの症例登録の件数
    true_seminar_effect: float  # 説明会が段階を上げる本当の効果（対数オッズ、平均）


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_doctors: int = 400, n_mrs: int = 40, seed: int = 0) -> SimData:
    rng = np.random.default_rng(seed)

    # --- 施設と医師 ---------------------------------------------------------
    doctors = []
    fac_id = 0
    while len(doctors) < n_doctors:
        is_hospital = rng.random() < 0.35
        size = int(rng.integers(3, 7)) if is_hospital else 1
        for _ in range(size):
            if len(doctors) >= n_doctors:
                break
            doctors.append({"facility_id": f"F{fac_id:04d}", "facility_type": "病院" if is_hospital else "診療所"})
        fac_id += 1
    doctors = pd.DataFrame(doctors)
    doctors["doctor_id"] = [f"D{i:04d}" for i in range(len(doctors))]
    doctors["evidence"] = rng.beta(2, 2, len(doctors))
    doctors["safety_concern"] = rng.beta(2, 3, len(doctors))
    doctors["early_adopter"] = rng.beta(2, 3, len(doctors))
    doctors["price_sensitive"] = rng.beta(2, 3, len(doctors))
    doctors["patient_volume"] = rng.gamma(2.0, 1.0, len(doctors)) * np.where(doctors["facility_type"] == "病院", 1.5, 1.0)

    # --- MRと、MRごとの書き方の癖 -----------------------------------------
    mr_ids = [f"M{i:03d}" for i in range(n_mrs)]
    mr_positivity = dict(zip(mr_ids, rng.uniform(-0.1, 0.6, n_mrs)))  # 中立を肯定的に書く確率
    mr_verbosity = dict(zip(mr_ids, rng.uniform(0.35, 0.95, n_mrs)))  # 医師の反応を書く確率
    fac_mr = {f: rng.choice(mr_ids) for f in doctors["facility_id"].unique()}
    doctors["mr_first"] = doctors["facility_id"].map(fac_mr)
    # 15%の医師は、2024Q3から担当MRが変わる
    changed = rng.random(len(doctors)) < 0.15
    doctors["mr_second"] = np.where(changed, rng.choice(mr_ids, len(doctors)), doctors["mr_first"])
    doctors["mr_change_q"] = np.where(changed, 6, 99)

    # 医師×製品ごとの主なネック（本当のネック）
    barrier_rows = {}
    for d in doctors.itertuples():
        for p, weights in PRODUCT_BARRIER_WEIGHTS.items():
            w = dict(weights)
            w["安全性への懸念"] = w.get("安全性への懸念", 0) * (0.5 + 2 * d.safety_concern)
            w["エビデンス不足"] = w.get("エビデンス不足", 0) * (0.5 + 2 * d.evidence)
            w["患者負担"] = w.get("患者負担", 0) * (0.5 + 2 * d.price_sensitive)
            w["競合品で十分"] = w.get("競合品で十分", 0) * (0.5 + 2 * (1 - d.early_adopter))
            keys = list(w)
            probs = np.array([w[k] for k in keys], dtype=float)
            barrier_rows[(d.doctor_id, p)] = keys[rng.choice(len(keys), p=probs / probs.sum())]

    # 説明会の本当の効果：エビデンスを重視する医師ほど大きい
    def seminar_effect(evidence: float) -> float:
        return 0.3 + 0.9 * evidence

    true_effect = float(np.mean([seminar_effect(e) for e in doctors["evidence"]]))

    product_appeal = {"A": 0.3, "B": -0.2, "C": 0.1}

    states, reports, regs = [], [], []
    report_id = 0
    for d in doctors.itertuples():
        for p, (pname, disease, launch_q, has_reg) in PRODUCTS.items():
            # 初期の段階
            if launch_q == 0:
                base = 0.8 * d.early_adopter + 0.3 * rng.normal()
                stage = int(np.clip(np.digitize(base, [0.25, 0.55, 0.8]), 0, 3))
            else:
                stage = 0
            stopped = False
            barrier = barrier_rows[(d.doctor_id, p)]
            for qi, q in enumerate(QUARTERS):
                if qi < launch_q:
                    continue
                mr = d.mr_first if qi < d.mr_change_q else d.mr_second
                stage_before = "中止" if stopped else STAGES[stage]
                # 説明会への参加：新薬を早く採用する医師ほど参加しやすく、段階も上がりやすい（交絡）。
                # 関心以上の段階も参加を増やすが、この架空データでは段階そのものは次の段階に影響しない
                p_seminar = _sigmoid(-1.8 + 2.0 * d.early_adopter + 0.6 * (stage >= 1))
                seminar = bool(rng.random() < p_seminar) and not stopped
                n_visits = int(rng.poisson(1.6 + 0.6 * (d.facility_type == "病院")))

                # 本当の評価（-1, 0, 1）
                att_logit = 0.5 * stage - 0.8 + product_appeal[p] + 0.8 * d.early_adopter - (
                    1.0 * d.safety_concern if p == "A" else 0.3 * d.safety_concern
                )
                u = rng.random()
                p_pos = _sigmoid(att_logit)
                p_neg = _sigmoid(-att_logit - 0.8)
                true_att = "肯定的" if u < p_pos else ("否定的" if u < p_pos + p_neg * (1 - p_pos) else "中立")
                active_barrier = barrier if (stage < 3 and not stopped) else None

                # 日報
                for _ in range(n_visits):
                    report_id += 1
                    reports.append(
                        _write_report(
                            rng, report_id, d.doctor_id, d.facility_id, mr, p, pname, q, qi,
                            stage_before, true_att, active_barrier, seminar,
                            mr_positivity[mr], mr_verbosity[mr], d.evidence,
                        )
                    )

                # 症例登録（製品A）
                if has_reg:
                    lam = {0: 0.0, 1: 0.05, 2: 0.6, 3: 1.4}[stage] * d.patient_volume if not stopped else 0.0
                    regs.append({"doctor_id": d.doctor_id, "product": p, "quarter": q, "new_registrations": int(rng.poisson(lam))})

                # 次の四半期への段階の変化
                base_logit = (
                    -1.6 + 1.4 * d.early_adopter + 0.5 * d.evidence + product_appeal[p]
                    - (1.0 * d.safety_concern + 0.7 * d.price_sensitive if p == "A" else 0.2 * d.safety_concern)
                    + 0.08 * n_visits
                )
                up_logit = base_logit + (seminar_effect(d.evidence) if seminar else 0.0)
                can_move = (not stopped) and stage < 3
                # 説明会に参加した場合と、しなかった場合の、段階が上がる本当の確率（施策の効果の答え）
                p_up_if_seminar = float(_sigmoid(base_logit + seminar_effect(d.evidence))) if can_move else 0.0
                p_up_if_not = float(_sigmoid(base_logit)) if can_move else 0.0
                moved_up = False
                if can_move and rng.random() < _sigmoid(up_logit):
                    stage += 1
                    moved_up = True
                elif not stopped and stage >= 2 and rng.random() < 0.03 + 0.06 * d.safety_concern * (p == "A"):
                    stopped = True
                states.append({
                    "doctor_id": d.doctor_id, "product": p, "quarter": q, "q_index": qi, "mr_id": mr,
                    "true_stage": stage_before, "seminar": seminar, "n_visits": n_visits,
                    "true_attitude": true_att, "true_barrier": active_barrier or "なし",
                    "moved_up_next": moved_up, "can_move": can_move,
                    "p_up_if_seminar": p_up_if_seminar, "p_up_if_not": p_up_if_not,
                })

    states = pd.DataFrame(states)
    reports = pd.DataFrame(reports)
    registrations = pd.DataFrame(regs)

    # 施設の売上：医師の段階と患者数から作る
    weight = {"認知": 0.0, "関心": 0.2, "試用": 2.0, "継続": 5.0, "中止": 0.0}
    s = states.merge(doctors[["doctor_id", "facility_id", "patient_volume"]], on="doctor_id")
    s["volume"] = s["true_stage"].map(weight) * s["patient_volume"] * rng.lognormal(0, 0.2, len(s))
    sales = s.groupby(["facility_id", "product", "quarter"], as_index=False)["volume"].sum()
    sales = sales.rename(columns={"volume": "sales"})

    return SimData(doctors, reports, states, sales, registrations, true_effect)


def _write_report(rng, rid, doctor_id, facility_id, mr, p, pname, q, qi, stage, true_att, barrier, seminar, positivity, verbosity, evidence_trait):
    """日報1件の文と、その文に書かれている内容の答え（人手のラベルに相当）を作る。"""
    # 話題：ネックがあれば半分はネックに関係する話題。エビデンスを重視する医師ほど有効性の話題が増える
    if barrier and rng.random() < 0.5:
        topic = BARRIER_TOPIC[barrier]
    elif rng.random() < 0.4 * evidence_trait:
        topic = "有効性"
    else:
        choices = ["有効性", "安全性", "使用感", "薬価・患者負担", "競合品"] + (["投与方法"] if p == "A" else [])
        topic = choices[rng.integers(len(choices))]
    phrase = TOPIC_PHRASES[topic][rng.integers(len(TOPIC_PHRASES[topic]))]
    sentences = [f"{pname}の{phrase}について{rng.choice(['説明', '資材で紹介', '情報提供'])}。"]
    label = {"topics": [topic], "attitude": "記載なし", "stage": "記載なし", "barriers": ["なし"], "mentions_patient": False}
    evidence = []

    if seminar and rng.random() < 0.5:
        sentences.append("Web講演会に参加いただいた。")

    if rng.random() < verbosity:
        # MRの書き方の癖：中立の反応を肯定的に書くことがある
        written = true_att
        if true_att == "中立" and rng.random() < max(positivity, 0):
            written = "肯定的"
        s = REACTION[(written, topic)][rng.integers(len(REACTION[(written, topic)]))] + "。"
        sentences.append(s)
        evidence.append(s)
        label["attitude"] = written
        if (written, topic) in REACTION_BARRIER:
            label["barriers"] = [REACTION_BARRIER[(written, topic)]]
        if rng.random() < 0.6 * evidence_trait:
            e = EVIDENCE_SEEKING_SENTENCE[rng.integers(len(EVIDENCE_SEEKING_SENTENCE))] + "。"
            sentences.append(e)
            evidence.append(e)
            if "有効性" not in label["topics"]:
                label["topics"].append("有効性")
        if barrier and written != "肯定的" and rng.random() < 0.7:
            b = BARRIER_SENTENCE[barrier][rng.integers(len(BARRIER_SENTENCE[barrier]))] + "。"
            sentences.append(b)
            evidence.append(b)
            label["barriers"] = sorted({barrier} | (set(label["barriers"]) - {"なし"}))
            if BARRIER_TOPIC[barrier] not in label["topics"]:
                label["topics"].append(BARRIER_TOPIC[barrier])
        if stage in STAGE_SENTENCE and rng.random() < 0.5:
            st = STAGE_SENTENCE[stage][rng.integers(len(STAGE_SENTENCE[stage]))] + "。"
            sentences.append(st)
            evidence.append(st)
            label["stage"] = stage
    if rng.random() < 0.08:
        sentences.append(PATIENT_SENTENCE[rng.integers(len(PATIENT_SENTENCE))] + "。")
        label["mentions_patient"] = True

    return {
        "report_id": f"R{rid:07d}", "doctor_id": doctor_id, "facility_id": facility_id, "mr_id": mr,
        "product": p, "quarter": q, "q_index": qi, "seminar_mentioned": "講演会" in "".join(sentences),
        "text": "".join(sentences),
        "gold_topics": "|".join(label["topics"]), "gold_attitude": label["attitude"], "gold_stage": label["stage"],
        "gold_barriers": "|".join(label["barriers"]), "gold_mentions_patient": label["mentions_patient"],
        "gold_evidence": "".join(evidence),
    }
