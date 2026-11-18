"""社内で動かすLLMで、日報から項目を抽出する。

vLLMやOllamaなど、OpenAI互換のAPIを持つサーバーを社内に立てて使う想定。
データは社外に出さない。例：

    vllm serve tokyotech-llm/Medical-Qwen3-Swallow-8B --port 8000
    python -m doctor_insight.extract_llm --base-url http://localhost:8000/v1 \
        --model tokyotech-llm/Medical-Qwen3-Swallow-8B --input outputs/reports.csv

出力はschema.DoctorSignalのJSONスキーマで制約する（response_formatのjson_schema）。
対応していないサーバーでは、JSONとして読めない出力や、スキーマに合わない出力が返ることがある。
その場合は検証で弾き、再試行し、それでも駄目なら記録して飛ばす。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
from pydantic import ValidationError

from .schema import ATTITUDES, BARRIERS, STAGES, TOPICS, DoctorSignal

SYSTEM_PROMPT = f"""あなたは製薬会社の営業日報を読み、皮膚科医の反応を分類する担当者です。
日報1件を読み、指定のJSON形式で答えてください。日報に書かれていないことは推測しないでください。

- topics: 話題になったこと。{TOPICS} から複数選ぶ。
- attitude: 製品への医師の評価。{ATTITUDES} から1つ。
  医師の反応が書かれていない（MRが説明したことだけが書かれている）場合は「記載なし」。
  「安全性は問題ない」は肯定的、「安全性が気になる」は否定的。
- stage: 処方の段階。{STAGES} から1つ。書かれていなければ「記載なし」。
  認知=使用経験なし、関心=使ってみたい、試用=使い始めた・初めて処方、継続=継続処方・第一選択。
- barriers: 処方のネック。{BARRIERS} から複数選ぶ。なければ「なし」のみ。
- mentions_patient: 個別の患者の話題が含まれるか。
- evidence: 判断の根拠にした文を、日報から原文のまま引用する。患者の話題の文は引用しない。
"""


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"日報：{text}"},
    ]


def extract_one(client, model: str, text: str, retries: int = 2) -> DoctorSignal | None:
    schema = DoctorSignal.model_json_schema()
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(text),
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {"name": "DoctorSignal", "schema": schema}},
            )
            signal = DoctorSignal.model_validate_json(resp.choices[0].message.content)
            # 根拠が日報の中にない（作り話の）場合は弾く
            if signal.evidence and not all(s in text for s in signal.evidence.split("。") if s):
                raise ValueError("evidence is not a quote from the report")
            return signal
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            if attempt == retries:
                print(f"skip: {e}", file=sys.stderr)
                return None
            time.sleep(0.5)
    return None


def main() -> None:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="社内のOpenAI互換サーバー。例 http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="report_id と text の列を持つCSV")
    ap.add_argument("--output", default="outputs/extracted_llm.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed-for-local-server")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    rows = []
    for r in df.itertuples():
        s = extract_one(client, args.model, r.text)
        if s is None:
            continue
        rows.append({"report_id": r.report_id, **to_row(s)})
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} / {len(df)} rows to {args.output}")


def to_row(s: DoctorSignal) -> dict:
    return {
        "topics": "|".join(s.topics), "attitude": s.attitude, "stage": s.stage,
        "barriers": "|".join(s.barriers), "mentions_patient": s.mentions_patient, "evidence": s.evidence,
    }


if __name__ == "__main__":
    main()
