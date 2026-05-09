from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.question_ranking import predict_mcq_answer


MODEL_LABEL_TO_KEY = {
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
    "Random Forest": "random_forest",
    "MLP Neural Network": "mlp_neural",
    "Ensemble": "ensemble",
}


def main() -> None:
    st.set_page_config(page_title="Model A Quiz Checker", page_icon="🧠", layout="centered")
    st.title("Model A Quiz Checker")
    st.caption("Checks A/B/C/D options using the same verifier feature pipeline used in training.")

    st.sidebar.header("Verifier Settings")
    chosen = st.sidebar.selectbox(
        "Choose Model A verifier:",
        list(MODEL_LABEL_TO_KEY.keys()),
        index=0,
    )
    model_key = MODEL_LABEL_TO_KEY[chosen]

    article = st.text_area("Article", height=180, placeholder="Paste passage/article text here...")
    question = st.text_input("Question", placeholder="Enter question text...")
    col1, col2 = st.columns(2)
    with col1:
        opt_a = st.text_input("Option A")
        opt_b = st.text_input("Option B")
    with col2:
        opt_c = st.text_input("Option C")
        opt_d = st.text_input("Option D")

    if st.button("Score Options", type="primary"):
        if not article.strip() or not question.strip():
            st.error("Article and question are required.")
            return
        options = {
            "A": opt_a.strip(),
            "B": opt_b.strip(),
            "C": opt_c.strip(),
            "D": opt_d.strip(),
        }
        if any(not options[k] for k in ["A", "B", "C", "D"]):
            st.error("Please provide all 4 options (A, B, C, D).")
            return

        with st.spinner("Scoring options..."):
            try:
                pred, scores = predict_mcq_answer(
                    article=article,
                    question=question,
                    options=options,
                    model_name=model_key,  # type: ignore[arg-type]
                )
            except Exception as exc:
                st.error(f"Could not score with {chosen}: {exc}")
                return

        st.subheader(f"{chosen} scores")
        for label in ["A", "B", "C", "D"]:
            st.write(f"{label}) {options[label]}  ->  score: `{scores[label]:.4f}`")
        st.success(f"Predicted answer: {pred}) {options[pred]}")


if __name__ == "__main__":
    main()
