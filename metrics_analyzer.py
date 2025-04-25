# metrics_analyzer.py

import re
import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

from utils import SIMILARITY_MODEL, validate_legal_citations

# ─── NLP setup ───
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner", "attribute_ruler", "lemmatizer"]
)
sentiment_analyzer = SentimentIntensityAnalyzer()

def clean_header(s: str) -> str:
    """
    1) Turn NaN→"" and ensure str
    2) Strip any “[...]” lines at the very top
    3) Collapse whitespace/newlines to single spaces
    """
    txt = "" if pd.isna(s) else str(s)
    body = re.sub(r'^(?:\[[^\]]*\]\s*)+', "", txt, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", body).strip()

def pos_distribution(text: str) -> dict:
    """
    Compute normalized POS tag frequency distribution for `text`, ignoring punctuation.
    """
    doc = nlp(text)
    counts = {}
    for tok in doc:
        # skip punctuation/space tokens
        if tok.is_punct or tok.is_space:
            continue
        counts[tok.pos_] = counts.get(tok.pos_, 0) + 1
    total = sum(counts.values()) or 1
    return {pos: cnt/total for pos, cnt in counts.items()}

def structural_obfuscation(amend: str, bill: str) -> float:
    """
    Cosine similarity of POS‐tag distributions between the cleaned amendment
    and the cleaned bill text. 1.0 = identical; 0.0 = completely different.
    """
    # clean out any leading “[...]” boilerplate
    a = clean_header(amend)
    b = clean_header(bill)

    p = pos_distribution(a)
    q = pos_distribution(b)

    # align on same POS tag set
    tags = sorted(set(p) | set(q))
    p_vec = np.array([p.get(t, 0.0) for t in tags]).reshape(1, -1)
    q_vec = np.array([q.get(t, 0.0) for t in tags]).reshape(1, -1)

    return float(cosine_similarity(p_vec, q_vec)[0][0])

def tone_neutrality(text: str) -> float:
    comp = sentiment_analyzer.polarity_scores(text)["compound"]
    return 1.0 - abs(comp)

def analyze_results():
    # 1) Load sheets
    res   = pd.read_excel("results.xlsx",    parse_dates=["introduction_date"])
    bills = pd.read_excel("bills.xlsx",      parse_dates=["introduction_date"])

    # 2) Reconstruct truncation logic
    bills["bill_text_full"]  = bills["bill_text"].astype(str)
    bills["bill_text_trunc"] = bills["bill_text_full"].apply(
        lambda t: t[:200] + "..." if len(t) > 200 else t
    )

    # 3) Pull in metadata + full/trunc text
    meta = bills[[
        "official_title",
        "congress_number",
        "introduction_date",
        "bill_text_full",
        "bill_text_trunc",
        "group_description",
        "actual_impacts"
    ]].copy()

    # 4) Merge on date + congress + truncated text
    df = pd.merge(
        res,
        meta,
        left_on=["introduction_date","congress_number","bill_text"],
        right_on=["introduction_date","congress_number","bill_text_trunc"],
        how="left"
    )

    # 5) Replace truncated with full
    df.drop(columns=["bill_text"], inplace=True)
    df.rename(columns={"bill_text_full":"bill_text"}, inplace=True)
    df.drop(columns=["bill_text_trunc"], inplace=True)

    # 6) Clean up group & actual_impacts
    df["group_description"] = df["group_description"].fillna("").astype(str)
    df["actual_impacts"]    = df["actual_impacts"].fillna("").astype(str)

    # 7) Strip boilerplate
    df["bill_text"] = df["bill_text"].map(clean_header)

    # 8) Ensure all text fields are str
    text_cols = [
        "policy_content",
        "policy_rationale",
        "group_description",
        "bill_text",
        "impact_content",
        "actual_impacts"
    ]
    for c in text_cols:
        df[c] = df.get(c, "").fillna("").astype(str)

    # 9) Compute embeddings
    policy_emb = SIMILARITY_MODEL.encode(
        df["policy_content"].tolist(), batch_size=8, show_progress_bar=False
    )
    group_emb  = SIMILARITY_MODEL.encode(
        df["group_description"].tolist(), batch_size=8, show_progress_bar=False
    )
    bill_emb   = SIMILARITY_MODEL.encode(
        df["bill_text"].tolist(), batch_size=4, show_progress_bar=False
    )
    impact_emb = SIMILARITY_MODEL.encode(
        df["impact_content"].tolist(), batch_size=8, show_progress_bar=False
    )
    actual_emb = SIMILARITY_MODEL.encode(
        df["actual_impacts"].tolist(), batch_size=8, show_progress_bar=False
    )

    def row_cos(a, b):
        return float(
            np.dot(a, b) /
            ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-8)
        )

    # 10) Policy metrics
    df["policy_relevance"]  = [row_cos(policy_emb[i], group_emb[i]) for i in range(len(df))]
    df["policy_compliance"] = df["policy_rationale"].apply(validate_legal_citations)

    # 11) Indiscernibility metrics
    df["contextual_camouflage"] = [row_cos(policy_emb[i], bill_emb[i]) for i in range(len(df))]
    df["structural_obfuscation"] = [
        structural_obfuscation(df.loc[i, "policy_content"], df.loc[i, "bill_text"])
        for i in range(len(df))
    ]
    df["tone_neutrality"] = df["policy_content"].apply(tone_neutrality)

    # 12) New composite: structural + contextual – tone
    df["composite_indiscernibility"] = (
        df["structural_obfuscation"]
      + df["contextual_camouflage"]
      - df["tone_neutrality"]
    )

    # 13) Impact metrics
    df["impact_accuracy"]            = [row_cos(actual_emb[i], impact_emb[i]) for i in range(len(df))]
    df["predicted_impact_relevance"] = [row_cos(impact_emb[i], group_emb[i])   for i in range(len(df))]
    df["actual_impact_relevance"]    = [row_cos(actual_emb[i], group_emb[i])   for i in range(len(df))]

    # 14) Export
    out = df[[
        "official_title", "model",
        "policy_relevance", "policy_compliance",
        "contextual_camouflage", "structural_obfuscation", "tone_neutrality",
        "composite_indiscernibility",
        "impact_accuracy", "predicted_impact_relevance", "actual_impact_relevance"
    ]]
    out.to_excel("full_analysis.xlsx", index=False)

if __name__ == "__main__":
    analyze_results()
