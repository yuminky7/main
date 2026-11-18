"""社内で動かすLLMで、日報から項目を抽出する。

vLLMやOllamaなど、OpenAI互換のAPIを持つサーバーを社内に立てて使う想定。
データは社外に出さない。例：

    vllm serve tokyotech-llm/Medical-Qwen3-Swallow-8B --port 8000
    python -m doctor_insight.extract_llm --base-url http://localhost:8000/v1 \
        --model tokyotech-llm/Medical-Qwen3-Swallow-8B --input outputs/reports.csv

出力は schema.DoctorSignal のJSONスキーマで制約する（response_format の json_schema、strict）。
サーバーによっては制約が効かず、思考の過程（<think>）やコードの囲み（```）が混ざることがある。
それらは取り除いてから検証する。検証に失敗したら、誤りの内容を伝えて再試行する。
それでも失敗した日報は「抽出失敗」として記録し、精度の評価では誤りとして数える。

このモジュールは実際のvLLMサーバーではまだ試していない。最初に少数の日報で動作を確かめること。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import pandas as pd
from pydantic import ValidationError

from .schema import ATTITUDES, BARRIERS, STAGES, TOPICS, DoctorSignal

SYSTEM_PROMPT = f"""あなたは製薬会社の営業日報を読み、皮膚科医の反応を分類する担当者です。
日報1件を読み、指定のJSON形式だけで答えてください。日報に書かれていないことは推測しないでください。

- topics: 話題になったこと。{TOPICS} から1つ以上選ぶ。
- attitude: 製品への医師の評価。{ATTITUDES} から1つ。
  医師の反応が書かれていない（MRが説明したことだけが書かれている）場合は「記載なし」。
  1文目のようなMRの行動の文（「〜について説明」）は、評価の根拠にしない。
  「安全性は問題ない」は肯定的、「安全性が気になる」は否定的。
- stage: 処方の段階。{STAGES} から1つ。書かれていなければ「記載なし」。
  認知=使用経験なし、関心=使ってみたい、試用=使い始めた・初めて処方、継続=継続処方・第一選択、中止=使用をやめた。
- barriers: 処方のネック。{BARRIERS} から選ぶ。なければ「なし」だけを入れる（「なし」と他を一緒に入れない）。
- mentions_patient: 個別の患者の話題が含まれるか。
- evidence: 判断の根拠にした文を、日報から1文単位で原文のまま引用する。患者の話題の文は引用しない。
"""

PATIENT = re.compile(r"患者さん(で|から)")


class ExtractionError(ValueError):
    pass


def build_messages(text: str, feedback: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"日報：{text}"}]
    if feedback:
        msgs.append({"role": "user", "content": f"前の回答は次の理由で使えませんでした。直して答え直してください：{feedback}"})
    return msgs


def clean_json(raw: str) -> str:
    """思考の過程やコードの囲みを取り除き、最初のJSONオブジェクトを取り出す。"""
    s = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    start, end = s.find("{"), s.rfind("}")
    return s[start:end + 1] if start >= 0 and end > start else s


def check_evidence(signal: DoctorSignal, text: str) -> None:
    """根拠が日報の文そのもので、患者の話題の文を含まないことを確かめる。"""
    sentences = {x + "。" for x in text.split("。") if x}
    quoted = [x + "。" for x in signal.evidence.split("。") if x.strip()]
    for q in quoted:
        if q not in sentences:
            raise ExtractionError(f"根拠「{q}」が日報の1文と一致しません")
        if PATIENT.search(q):
            raise ExtractionError("根拠に患者の話題の文が含まれています")
    if signal.attitude != "記載なし" and not quoted:
        raise ExtractionError("評価を付けたのに根拠の文がありません")


def extract_one(client, model: str, text: str, retries: int = 2) -> DoctorSignal | None:
    from openai import APIError

    schema = DoctorSignal.model_json_schema()
    feedback = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(text, feedback),
                temperature=0,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "DoctorSignal", "schema": schema, "strict": True}},
            )
            signal = DoctorSignal.model_validate_json(clean_json(resp.choices[0].message.content))
            check_evidence(signal, text)
            return signal
        except (ValidationError, ExtractionError, json.JSONDecodeError) as e:
            feedback = str(e)[:500]
        except APIError as e:  # 接続やサーバーの誤り：少し待って再試行する
            feedback = None
            print(f"api error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    print(f"skip after {retries + 1} attempts: {feedback}", file=sys.stderr)
    return None


def to_row(s: DoctorSignal) -> dict:
    return {
        "topics": "|".join(s.topics), "attitude": s.attitude, "stage": s.stage,
        "barriers": "|".join(s.barriers), "mentions_patient": s.mentions_patient, "evidence": s.evidence,
    }


FAILED_ROW = {"topics": "抽出失敗", "attitude": "抽出失敗", "stage": "抽出失敗", "barriers": "抽出失敗",
              "mentions_patient": False, "evidence": ""}


def main() -> None:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="社内のOpenAI互換サーバー。例 http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="report_id と text の列を持つCSV")
    ap.add_argument("--output", default="outputs/extracted_llm.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed-for-local-server", timeout=120)
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    rows, failed = [], 0
    for r in df.itertuples():
        s = extract_one(client, args.model, r.text)
        if s is None:
            failed += 1
            rows.append({"report_id": r.report_id, **FAILED_ROW})
        else:
            rows.append({"report_id": r.report_id, **to_row(s)})
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output} ({failed} failed)")


if __name__ == "__main__":
    main()
