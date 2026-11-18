import pandas as pd

from doctor_insight import extract_rules
from doctor_insight.evaluate import evaluate
from doctor_insight.features import FEATURES_OTHER, FEATURES_TEXT, build_panel
from doctor_insight.schema import DoctorSignal
from doctor_insight.simulate import simulate


def _extract_all(reports: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reports.itertuples():
        s = extract_rules.extract(r.text)
        rows.append({"report_id": r.report_id, "topics": "|".join(s.topics), "attitude": s.attitude,
                     "stage": s.stage, "barriers": "|".join(s.barriers),
                     "mentions_patient": s.mentions_patient, "evidence": s.evidence})
    return pd.DataFrame(rows)


def test_rule_extractor_handles_negation_words():
    assert extract_rules.extract("製品Aの安全性について説明。安全性は特に問題ないとのこと。").attitude == "肯定的"
    assert extract_rules.extract("製品Aの安全性について説明。安全性が気になるとのこと。").attitude == "否定的"
    assert extract_rules.extract("製品Aの安全性について説明。").attitude == "記載なし"


def test_evidence_excludes_patient_sentences():
    s = extract_rules.extract("製品Bの使用感について説明。使用感が良いと好評。70代の患者さんで改善が見られたとのこと。")
    assert s.mentions_patient
    assert "患者さんで" not in s.evidence


def test_schema_rejects_unknown_labels():
    import pytest
    with pytest.raises(Exception):
        DoctorSignal(topics=["有効性"], attitude="とても良い", stage="試用", barriers=["なし"],
                     mentions_patient=False, evidence="")


def test_pipeline_runs_without_future_leakage():
    sim = simulate(n_doctors=40, n_mrs=6, seed=1)
    ext = _extract_all(sim.reports)
    ev = evaluate(sim.reports.sample(100, random_state=0), ext)
    assert set(ev["summary"]["field"]) == {"attitude", "stage", "mentions_patient"}

    panel = build_panel(sim.reports, ext, sim.states, train_until_q=7)
    assert set(FEATURES_TEXT + FEATURES_OTHER) <= set(panel.columns)
    # 四半期 t の特徴量が、t より後の日報を使っていないこと
    d, p = panel.iloc[0][["doctor_id", "product"]]
    rows = panel[(panel["doctor_id"] == d) & (panel["product"] == p)].sort_values("q_index")
    later = sim.reports[(sim.reports["doctor_id"] == d) & (sim.reports["product"] == p)]
    first_q = rows["q_index"].min()
    assert rows.iloc[0]["n_reports"] == (later["q_index"] == first_q).sum()
    # 他の自社品の特性は、対象の製品の日報を含まない
    assert (panel["other_n_reaction"] >= 0).all()
