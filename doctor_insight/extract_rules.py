"""キーワードのルールで日報から項目を抽出する。

LLMが使えない環境で試作を最後まで動かすための仮の抽出器であり、精度の比べる基準にもなる。
「安全性は特に問題ない」と「安全性が気になる」のような違いには弱い。
実データでは、ここをLLMによる抽出（extract_llm.py）に置き換える。
"""

from __future__ import annotations

import re

from .schema import DoctorSignal

TOPIC_KEYWORDS = {
    "有効性": ["有効性", "臨床成績", "効果"],
    "安全性": ["安全性", "副作用", "感染症"],
    "使用感": ["使用感", "塗り心地", "剤形", "べたつき", "塗りやすい"],
    "投与方法": ["投与方法", "投与間隔", "用法", "自己注射", "投与の手間"],
    "薬価・患者負担": ["薬価", "患者負担", "費用", "高額"],
    "競合品": ["他剤", "既存治療", "今の治療", "既存の薬"],
}
POSITIVE = ["満足", "期待どおり", "納得", "問題ない", "心配は少ない", "好評", "助かる", "使いやすい"]
NEGATIVE = ["物足りない", "疑問", "気になる", "心配され", "不満", "負担", "ネック", "大変", "勧めにくい", "十分とのこと", "困っていない"]
NEUTRAL = ["様子見", "判断したい", "確認したい", "変わらない", "理解された", "相談する", "検討したい"]
BARRIER_KEYWORDS = {
    "安全性への懸念": ["長期の安全性", "安全性データが出るまで"],
    "エビデンス不足": ["データがもっと欲しい", "エビデンスが不十分"],
    "患者負担": ["費用負担がネック", "患者負担が大きい"],
    "投与の手間": ["投与の手間"],
    "競合品で十分": ["切り替える理由がない"],
    "対象患者がいない": ["対象となる患者さんが今はいない"],
}
STAGE_KEYWORDS = {
    "継続": ["継続して処方", "第一選択"],
    "試用": ["使い始めた", "初めて処方"],
    "関心": ["興味を示", "使ってみたい"],
    "認知": ["使用経験はない"],
}
PATIENT = re.compile(r"患者さん(で|から)")


def extract(text: str) -> DoctorSignal:
    topics = [t for t, kws in TOPIC_KEYWORDS.items() if any(k in text for k in kws)] or ["その他"]
    pos = any(k in text for k in POSITIVE)
    neg = any(k in text for k in NEGATIVE)
    neu = any(k in text for k in NEUTRAL)
    if neg and not pos:
        attitude = "否定的"
    elif pos and not neg:
        attitude = "肯定的"
    elif neu or (pos and neg):
        attitude = "中立"
    else:
        attitude = "記載なし"
    barriers = [b for b, kws in BARRIER_KEYWORDS.items() if any(k in text for k in kws)] or ["なし"]
    stage = next((s for s, kws in STAGE_KEYWORDS.items() if any(k in text for k in kws)), "記載なし")
    sentences = [s + "。" for s in text.split("。") if s]
    evidence = "".join(s for s in sentences[1:] if not PATIENT.search(s))
    return DoctorSignal(
        topics=topics, attitude=attitude, stage=stage, barriers=barriers,
        mentions_patient=bool(PATIENT.search(text)), evidence=evidence,
    )
