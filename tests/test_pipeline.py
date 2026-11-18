import numpy as np
import pandas as pd
import pytest

from doctor_insight import extract_rules
from doctor_insight.evaluate import evaluate
from doctor_insight.extract_llm import ExtractionError, check_evidence, clean_json
from doctor_insight.features import FEATURES_BASE, FEATURES_OTHER, FEATURES_TEXT, build_panel
from doctor_insight.schema import DoctorSignal
from doctor_insight.simulate import simulate

TRAIN_UNTIL_Q = 6


def _extract_all(reports: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reports.itertuples():
        s = extract_rules.extract(r.text)
        rows.append({"report_id": r.report_id, "topics": "|".join(s.topics), "attitude": s.attitude,
                     "stage": s.stage, "barriers": "|".join(s.barriers),
                     "mentions_patient": s.mentions_patient, "evidence": s.evidence})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sim_and_ext():
    sim = simulate(n_doctors=40, n_mrs=6, seed=1)
    return sim, _extract_all(sim.reports)


def test_rule_extractor_handles_negation_words():
    assert extract_rules.extract("製品Aの安全性について説明。安全性は特に問題ないとのこと。").attitude == "肯定的"
    assert extract_rules.extract("製品Aの安全性について説明。安全性が気になるとのこと。").attitude == "否定的"
    assert extract_rules.extract("製品Aの安全性について説明。").attitude == "記載なし"


def test_rule_extractor_ignores_the_mr_action_sentence():
    # 1文目の「患者負担」はMRの行動であり、医師の反応ではない
    assert extract_rules.extract("製品Aの患者負担について説明。").attitude == "記載なし"
    # ネックだけを述べる文で、中立の反応が否定的にならない
    s = extract_rules.extract("製品Aの有効性について説明。効果は様子見とのこと。日本人のデータがもっと欲しいとのこと。")
    assert s.attitude == "中立" and s.barriers == ["エビデンス不足"]


def test_evidence_excludes_patient_sentences():
    s = extract_rules.extract("製品Bの使用感について説明。使用感が良いと好評。70代の患者さんで改善が見られたとのこと。")
    assert s.mentions_patient
    assert "患者さんで" not in s.evidence


def test_schema_rejects_inconsistent_labels():
    with pytest.raises(Exception):
        DoctorSignal(topics=["有効性"], attitude="とても良い", stage="試用", barriers=["なし"], mentions_patient=False, evidence="")
    with pytest.raises(Exception):
        DoctorSignal(topics=["有効性"], attitude="中立", stage="試用", barriers=["なし", "患者負担"], mentions_patient=False, evidence="")
    with pytest.raises(Exception):
        DoctorSignal(topics=[], attitude="中立", stage="試用", barriers=["なし"], mentions_patient=False, evidence="")


def test_llm_evidence_check():
    text = "製品Aの安全性について説明。安全性が気になるとのこと。70代の患者さんで改善が見られたとのこと。"
    ok = DoctorSignal(topics=["安全性"], attitude="否定的", stage="記載なし", barriers=["なし"],
                      mentions_patient=True, evidence="安全性が気になるとのこと。")
    check_evidence(ok, text)
    for bad in ["製品", "70代の患者さんで改善が見られたとのこと。", "安全性が心配とのこと。"]:
        with pytest.raises(ExtractionError):
            check_evidence(ok.model_copy(update={"evidence": bad}), text)
    assert clean_json('<think>考え中</think>```json\n{"a": 1}\n```') == '{"a": 1}'


def test_evaluate_counts_missing_predictions_as_errors(sim_and_ext):
    sim, ext = sim_and_ext
    gold = sim.reports.head(50)
    ev = evaluate(gold, ext[ext["report_id"].isin(gold["report_id"].head(40))])
    assert ev["summary"].loc[0, "failed"] == 10
    assert ev["summary"].loc[0, "n"] == 50


def test_features_do_not_use_future_reports(sim_and_ext):
    """四半期 t までの日報だけで作った特徴量と、全期間の日報で作った特徴量が、t 以前の行で一致する。"""
    sim, ext = sim_and_ext
    full = build_panel(sim.reports, ext, sim.states, TRAIN_UNTIL_Q)
    cols = [c for c in FEATURES_TEXT + FEATURES_BASE + FEATURES_OTHER if c in full.columns]
    keys = ["doctor_id", "product", "q_index"]
    for t in [TRAIN_UNTIL_Q, 8, 10]:
        cut = sim.reports[sim.reports["q_index"] <= t]
        part = build_panel(cut, ext[ext["report_id"].isin(cut["report_id"])],
                           sim.states[sim.states["q_index"] <= t], TRAIN_UNTIL_Q)
        a = full[full["q_index"] <= t].set_index(keys)[cols].sort_index()
        b = part.set_index(keys)[cols].sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
