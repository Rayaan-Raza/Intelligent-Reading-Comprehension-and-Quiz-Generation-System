"""Model A: question generation — RACE mode (dataset question) vs custom passage (template + TF‑IDF)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text, split_sentences


def _fit_passage_sentence_tfidf(sentences_clean: List[str]) -> Tuple[TfidfVectorizer, Any]:
    """One TF-IDF fit over all sentences (each sentence = one document)."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(sentences_clean)
    return vectorizer, X


def _score_sentence_rows(sentences_clean: List[str], X) -> List[float]:
    """Importance score per sentence row (matches best-sentence heuristic)."""
    n = X.shape[0]
    location_bonus_terms = ("located", "capital", "city", "country", "museum", "tower", "river")
    prefer = [
        i
        for i in range(n)
        if ("located" in sentences_clean[i].lower())
        or re.search(r"\bin\s+[a-z]+\b", sentences_clean[i].lower())
    ]
    candidate_indices = prefer if prefer else list(range(n))
    scores = [0.0] * n
    for i in candidate_indices:
        row = X[i]
        if row.nnz == 0:
            continue
        sc = float(row.max(axis=1).toarray().ravel()[0])
        sent_i = sentences_clean[i].lower()
        if any(t in sent_i for t in location_bonus_terms):
            sc += 0.05
        scores[i] = sc
    for i in range(n):
        if scores[i] == 0.0:
            row = X[i]
            if row.nnz:
                scores[i] = float(row.max(axis=1).toarray().ravel()[0])
    return scores


def _answer_phrase_for_row(
    vectorizer: TfidfVectorizer,
    X,
    sentences_clean: List[str],
    idx: int,
) -> str:
    row = X[idx]
    feature_names = vectorizer.get_feature_names_out()
    dense = row.toarray().ravel()
    if dense.sum() == 0:
        words = re.findall(r"[a-z0-9']+", sentences_clean[idx].lower())
        return max(words, key=len) if words else "answer"
    best_col = int(np.argmax(dense))
    return str(feature_names[best_col])


class QuestionGenerationMode(str, Enum):
    RACE_SAMPLE = "race_sample"
    CUSTOM_PASSAGE = "custom_passage"


@dataclass
class GeneratedQuestion:
    """One generated or sourced question with optional metadata."""

    question: str
    correct_answer: str
    mode: QuestionGenerationMode
    source_sentence: Optional[str] = None
    masked_sentence: Optional[str] = None
    template_name: Optional[str] = None


def generate_from_race_sample(original_question: str) -> GeneratedQuestion:
    """
    Mode 1 — RACE sample: use the dataset question as-is (reliable, high quality).

    Parameters
    ----------
    original_question :
        The `question` field from a RACE row.
    """
    q = str(original_question).strip()
    if not q:
        raise ValueError("RACE mode requires a non-empty original_question.")
    return GeneratedQuestion(
        question=q,
        correct_answer="",  # MCQ answer is A/B/C/D from options elsewhere
        mode=QuestionGenerationMode.RACE_SAMPLE,
        source_sentence=None,
        masked_sentence=None,
        template_name="race_original",
    )


def _pick_best_sentence_and_answer(
    sentences_clean: List[str],
) -> Tuple[int, str, object]:
    """
    Rank sentences by TF-IDF (each sentence = one document); pick sentence with highest max term weight.
    Returns (index, answer_phrase, fitted_vectorizer) for argmax term in that sentence.
    """
    if not sentences_clean:
        raise ValueError("No sentences after splitting the passage.")

    vectorizer, X = _fit_passage_sentence_tfidf(sentences_clean)
    scores = _score_sentence_rows(sentences_clean, X)
    best_idx = int(np.argmax(scores)) if scores else 0

    row = X[best_idx]
    if row.nnz == 0 or max(scores) == 0.0:
        lengths = [len(s.split()) for s in sentences_clean]
        best_idx = int(lengths.index(max(lengths)))

    answer_phrase = _answer_phrase_for_row(vectorizer, X, sentences_clean, best_idx)
    return best_idx, answer_phrase, vectorizer


def _is_year_token(s: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}", s.strip()))


def _place_stem_and_answer(sentence: str) -> Optional[Tuple[str, str]]:
    """
    If the sentence matches ``... is located in <place>``, return (stem, place) for
    ``Where is <stem> located?`` — avoids duplicating "is located" in the stem.
    """
    s = sentence.strip()
    m = re.match(
        r"^(.*?)\s+is\s+located\s+in\s+([a-z][a-z'\-]*)\s*[.!?]?$",
        s,
        re.I,
    )
    if not m:
        return None
    stem = m.group(1).strip()
    place = m.group(2).strip()
    if not stem or not place:
        return None
    return stem, place


def _is_date_like(text: str) -> bool:
    t = text.lower()
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d", t):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
        return True
    return False


def _apply_templates(sentence: str, answer: str) -> Tuple[str, str]:
    """
    Replace answer with blank and choose a Wh-question template.
    Returns (question, template_name).
    """
    ans = answer.strip()
    sent_lower = sentence.lower()
    ans_lower = ans.lower()

    # Mask answer (first occurrence, word-boundary safe for single tokens)
    if " " in ans:
        masked = sentence.replace(ans, "_____", 1)
        if masked == sentence:
            masked = sentence.replace(ans_lower, "_____", 1)
    else:
        pattern = re.compile(r"\b" + re.escape(ans_lower) + r"\b", re.I)
        masked = pattern.sub("_____", sentence, count=1)

    if masked == sentence:
        masked = re.sub(re.escape(ans), "_____", sentence, count=1, flags=re.I)

    # When — year or explicit date language
    if _is_year_token(ans) or _is_date_like(sentence):
        q = f"When did the following take place: {masked.strip()}"
        return q.strip(), "when_date"

    # Where — location cues or "in <answer>" tail
    where_cues = ("located", "situated", "found in", "based in", "capital of", "city of", "country")
    has_place_like_answer = bool(re.search(r"[a-z]", ans_lower)) and not _is_year_token(ans)
    if has_place_like_answer and (
        any(c in sent_lower for c in where_cues)
        or re.search(r"\bin\s+" + re.escape(ans_lower) + r"\b", sent_lower)
    ):
        tail = re.compile(r"\s+in\s+" + re.escape(ans_lower) + r"[.!?\s]*$", re.I)
        if tail.search(sentence):
            stem = tail.sub("", sentence).strip()
            stem = re.sub(r"[.!?]+$", "", stem).strip()
            q = f"Where is {stem} located?"
            return q, "where_place"
        ms = masked.strip().rstrip(".!? ")
        q = f"According to the passage, where is this happening: {ms}?"
        return q.strip(), "where_generic"

    # Who — person cue words (lightweight; no NER)
    who_cues = (
        "president", "minister", "director", "author", "researcher", "scientist",
        "said", "according to", "born", "named",
    )
    if any(c in sent_lower for c in who_cues):
        q = f"Who is referred to here: {masked.strip()}"
        return q.strip(), "who_person"

    # Generic cloze (project-style fallback)
    ms = masked.strip().rstrip(".!? ")
    q = f"According to the passage, what best completes this sentence: {ms}?"
    return q, "generic_cloze"


def generate_from_passage_sentence_index(
    sentences_clean: List[str],
    idx: int,
    vectorizer: TfidfVectorizer,
    X: Any,
) -> GeneratedQuestion:
    """
    Build one custom-mode question from sentence ``sentences_clean[idx]`` using a **shared**
    passage-level TF-IDF matrix ``X`` (do not refit per sentence).
    """
    sentence = sentences_clean[idx]
    initial = _answer_phrase_for_row(vectorizer, X, sentences_clean, idx)
    place_pair = _place_stem_and_answer(sentence)
    if place_pair:
        stem, answer_phrase = place_pair
        question_text = f"Where is {stem} located?"
        template_name = "where_place"
        masked = re.sub(
            r"\s+is\s+located\s+in\s+" + re.escape(answer_phrase.strip()) + r"(\s*[.!?]?)$",
            r" is located in _____\1",
            sentence,
            count=1,
            flags=re.I,
        )
    else:
        answer_phrase = initial
        question_text, template_name = _apply_templates(sentence, answer_phrase)
        masked = sentence
        ap = answer_phrase.strip()
        if " " in ap:
            masked = masked.replace(ap, "_____", 1)
        else:
            masked = re.sub(
                r"\b" + re.escape(ap.lower()) + r"\b",
                "_____",
                masked,
                count=1,
                flags=re.I,
            )

    return GeneratedQuestion(
        question=question_text,
        correct_answer=answer_phrase.strip(),
        mode=QuestionGenerationMode.CUSTOM_PASSAGE,
        source_sentence=sentence,
        masked_sentence=masked,
        template_name=template_name,
    )


def enumerate_candidate_questions(passage: str, top_k: int = 5) -> List[GeneratedQuestion]:
    """
    Produce up to ``top_k`` template questions from **different** sentences (TF-IDF ranked).

    Used before classifier ranking (SVM / Random Forest) over synthetic verification rows.
    """
    raw = str(passage).strip()
    if not raw:
        raise ValueError("Passage must be non-empty.")
    sentences_clean = split_sentences(raw)
    if not sentences_clean:
        sentences_clean = [clean_text(raw)]

    vectorizer, X = _fit_passage_sentence_tfidf(sentences_clean)
    scores = _score_sentence_rows(sentences_clean, X)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    seen_q: set[str] = set()
    out: List[GeneratedQuestion] = []
    for idx in order:
        if len(out) >= top_k:
            break
        try:
            g = generate_from_passage_sentence_index(sentences_clean, idx, vectorizer, X)
        except ValueError:
            continue
        key = (g.question.strip().lower(), g.correct_answer.strip().lower())
        if key in seen_q:
            continue
        seen_q.add(key)
        out.append(g)
    return out


def generate_from_custom_passage(passage: str) -> GeneratedQuestion:
    """
    Mode 2 — Custom passage: template-based cloze question using TF-IDF sentence ranking.

    Steps: split sentences → TF-IDF importance → best sentence → answer phrase → blank → Wh-template.
    """
    raw = str(passage).strip()
    if not raw:
        raise ValueError("Custom mode requires a non-empty passage.")

    sentences_clean = split_sentences(raw)
    if not sentences_clean:
        sentences_clean = [clean_text(raw)]

    vectorizer, X = _fit_passage_sentence_tfidf(sentences_clean)
    scores = _score_sentence_rows(sentences_clean, X)
    best_idx = int(np.argmax(scores)) if scores else 0
    if scores and max(scores) == 0.0:
        lengths = [len(s.split()) for s in sentences_clean]
        best_idx = int(lengths.index(max(lengths)))
    return generate_from_passage_sentence_index(sentences_clean, best_idx, vectorizer, X)


def generate_question(
    passage: str,
    *,
    race_question: Optional[str] = None,
    use_race_question_if_provided: bool = True,
) -> GeneratedQuestion:
    """
    Unified entry point.

    If ``race_question`` is provided and ``use_race_question_if_provided`` is True,
    returns Mode 1 (RACE). Otherwise runs Mode 2 on ``passage``.
    """
    if use_race_question_if_provided and race_question is not None:
        rq = str(race_question).strip()
        if rq:
            return generate_from_race_sample(rq)
    return generate_from_custom_passage(passage)
