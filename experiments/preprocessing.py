import re
from typing import Iterable

from pythainlp import word_tokenize, util
from pythainlp.corpus.common import thai_stopwords
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

THAI_STOPWORDS = set(thai_stopwords())
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "from", "as", "at", "it", "its", "into", "than", "then",
    "can", "could", "should", "would", "will", "may", "might", "do", "does",
    "did", "done", "have", "has", "had", "having"
}
STOP_WORDS = THAI_STOPWORDS.union(EN_STOPWORDS)


def safe_text(text) -> str:
    return "" if text is None else str(text)


def combine_text(row: dict) -> str:
    """
    รองรับได้ทั้ง schema แบบเก่าและ schema แบบใหม่
    - เก่า: BookName, Keyword, Detail
    - ใหม่: title, keyword, description, toc
    """
    parts = [
        safe_text(row.get("BookName")),
        safe_text(row.get("Keyword")),
        safe_text(row.get("Detail")),
        safe_text(row.get("title")),
        safe_text(row.get("keyword")),
        safe_text(row.get("description")),
        safe_text(row.get("toc")),
    ]
    return " ".join([p for p in parts if p.strip()])


def clean_text(text: str) -> str:
    """
    อิง flow ปี 2564:
    normalize -> lower -> remove digits -> remove non-TH/EN chars ->
    tokenize(deepcut) -> lemmatize -> remove stopwords/short tokens
    """
    text = util.normalize(safe_text(text))
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\u0E00-\u0E7Fa-z\s]", " ", text)

    tokens = word_tokenize(text, engine="deepcut")
    cleaned_tokens = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # lemmatize อังกฤษ ถ้าเป็น token อังกฤษ
        if re.fullmatch(r"[a-zA-Z]+", token):
            token = lemmatizer.lemmatize(token)

        if token in STOP_WORDS:
            continue

        if len(token) <= 1:
            continue

        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def identity_fun(x: Iterable):
    return x