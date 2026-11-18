"""フェーズ1：日報にどんな評価やネックが書かれているかを、トピック抽出で洗い出す。

文意知に近い考え方（文ごとにトピックへの当てはまりを数値にする）を、
scikit-learn の TF-IDF（文字のn-gram）と NMF で簡単に再現したもの。
実データでは、文の埋め込み（Ruri-v3 など）と BERTopic を使うと、言い換えに強くなる。
結果は「項目の候補」であり、製品担当者と相談して schema.py の選択肢を決める材料にする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def reaction_sentences(reports: pd.DataFrame) -> pd.Series:
    """MRの行動の文（1文目）を除き、医師の反応の文だけを取り出す。"""
    s = reports["text"].str.split("。").apply(lambda v: [x for x in v[1:] if x and "講演会" not in x])
    return s.explode().dropna()


def discover_topics(sentences: pd.Series, n_topics: int = 10, n_terms: int = 6, seed: int = 0):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=20000)
    X = vec.fit_transform(sentences)
    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)  # 文×トピックの当てはまり
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for k, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[::-1][:n_terms]
        top_sent = sentences.iloc[W[:, k].argsort()[::-1][:3]].tolist()
        rows.append({"topic": k, "share": float((W.argmax(axis=1) == k).mean()),
                     "top_terms": " / ".join(terms[top_idx]), "examples": " | ".join(top_sent)})
    return pd.DataFrame(rows).sort_values("share", ascending=False), W
