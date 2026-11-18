"""日報1件から抽出する項目の定義。

LLMによる抽出、ルールによる抽出、人手のラベルのすべてで同じ定義を使う。
選択肢は例であり、フェーズ1のトピック抽出と製品担当者との相談で決める。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Topic = Literal["有効性", "安全性", "使用感", "投与方法", "薬価・患者負担", "競合品", "その他"]
Attitude = Literal["肯定的", "中立", "否定的", "記載なし"]
Stage = Literal["認知", "関心", "試用", "継続", "中止", "記載なし"]
Barrier = Literal[
    "安全性への懸念",
    "エビデンス不足",
    "患者負担",
    "投与の手間",
    "競合品で十分",
    "対象患者がいない",
    "なし",
]

TOPICS: list[str] = list(Topic.__args__)
ATTITUDES: list[str] = list(Attitude.__args__)
STAGES: list[str] = list(Stage.__args__)
BARRIERS: list[str] = list(Barrier.__args__)

# 処方の段階の順序（「中止」と「記載なし」は順序の外）
STAGE_ORDER = {"認知": 0, "関心": 1, "試用": 2, "継続": 3}


class DoctorSignal(BaseModel):
    """日報1件・製品1つについて、医師の評価とネックを表す。"""

    # JSONスキーマに additionalProperties: false を出す（strict な構造化出力の要件）
    model_config = ConfigDict(extra="forbid")

    topics: list[Topic] = Field(description="日報で話題になったこと。複数可")
    attitude: Attitude = Field(
        description="製品への医師の評価。医師の反応が書かれていなければ「記載なし」"
    )
    stage: Stage = Field(description="処方の段階。判断できなければ「記載なし」")
    barriers: list[Barrier] = Field(description="処方のネック。なければ「なし」のみ")
    mentions_patient: bool = Field(description="患者の話題が含まれるか")
    evidence: str = Field(description="判断の根拠にした日報の文。原文のまま引用する")

    @model_validator(mode="after")
    def _consistent(self) -> "DoctorSignal":
        if not self.topics:
            raise ValueError("topics must not be empty")
        if not self.barriers:
            raise ValueError("barriers must not be empty (use 「なし」)")
        if "なし" in self.barriers and len(self.barriers) > 1:
            raise ValueError("「なし」 must appear alone in barriers")
        return self
