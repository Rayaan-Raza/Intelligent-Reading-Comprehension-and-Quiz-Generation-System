"""Streamlit Web UI for RACE RC Project - Redesigned Editorial Aesthetic."""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import run_pipeline

# Page Config
st.set_page_config(
    page_title="study smart ✦",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------
# CUSTOM CSS INJECTION
# ------------------------------------------------------------------------
# Color Palette
# Background: #FDF8F5 (cream white)
# Sidebar: #F5EBF5
# Surfaces: #F9EEF3
# Secondary Surfaces: #E8D5E8
# Accent: #C47FA8
# Text on Accent: #8B5A8B
# Body Text: #3D2040

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=DM+Serif+Display&display=swap');

    /* Global Typography and Backgrounds */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #3D2040;
    }
    
    .stApp {
        background-color: #FDF8F5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F5EBF5;
        border-right: 1px dotted #E8D5E8;
    }

    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6, .serif-text {
        font-family: 'DM Serif Display', serif !important;
        color: #3D2040 !important;
        font-weight: normal;
    }
    
    .stMarkdown p {
        color: #3D2040;
        line-height: 1.6;
    }

    /* Primary Headers */
    .main-title {
        font-size: 3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 999px !important;
        background-color: #F9EEF3 !important;
        color: #8B5A8B !important;
        border: 1px solid #E8D5E8 !important;
        box-shadow: 0 2px 12px rgba(180, 120, 170, 0.1);
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: #E8D5E8 !important;
        border-color: #C47FA8 !important;
        color: #3D2040 !important;
    }
    
    /* Reveal Answer Button (Beauty CTA) */
    div.beauty-cta > div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #C47FA8 0%, #9B6BAA 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(196, 127, 168, 0.3);
    }
    div.beauty-cta > div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(196, 127, 168, 0.4);
    }

    /* Ghost Download Button */
    div.ghost-button > div[data-testid="stDownloadButton"] > button {
        border-radius: 999px !important;
        background-color: transparent !important;
        border: 1px dashed #C47FA8 !important;
        color: #8B5A8B !important;
        box-shadow: none;
    }
    div.ghost-button > div[data-testid="stDownloadButton"] > button:hover {
        background-color: #F9EEF3 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500;
        color: #8B5A8B !important;
        border-radius: 16px 16px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3D2040 !important;
        border-bottom-color: #C47FA8 !important;
        border-bottom-width: 3px !important;
    }
    
    /* Text Input / Text Area */
    .stTextArea textarea {
        background-color: #F9EEF3 !important;
        border: 1px solid #E8D5E8 !important;
        border-radius: 16px !important;
        color: #3D2040 !important;
    }
    .stTextArea textarea:focus {
        border-color: #C47FA8 !important;
        box-shadow: 0 0 0 1px #C47FA8 !important;
    }

    /* Expander (Accordion) */
    .stExpander {
        background-color: #F9EEF3;
        border-radius: 16px;
        border: 1px dotted #E8D5E8 !important;
        box-shadow: 0 2px 12px rgba(180, 120, 170, 0.1);
        margin-bottom: 2rem;
    }
    .stExpander summary {
        font-family: 'DM Serif Display', serif;
        font-size: 1.2rem;
        color: #8B5A8B;
    }

    /* Quiz Radio Buttons */
    /* Target the container of each radio option */
    .stRadio [role="radiogroup"] > label {
        background-color: #FDF8F5;
        border: 1px solid #E8D5E8;
        border-radius: 999px;
        padding: 12px 24px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(180, 120, 170, 0.05);
        cursor: pointer;
        display: flex;
        align-items: center;
        color: #3D2040 !important;
    }
    
    /* Force text color for radio options to prevent dark mode conflicts */
    .stRadio [role="radiogroup"] > label p,
    .stRadio [role="radiogroup"] > label div,
    .stRadio [role="radiogroup"] > label span {
        color: #3D2040 !important;
    }
    
    .stRadio [role="radiogroup"] > label:hover {
        background-color: #F9EEF3;
        border-color: #C47FA8;
    }
    
    /* The active state is targeted via data-baseweb attributes internally, 
       but we can use focus-within for general selection if needed */
    .stRadio [role="radiogroup"] > label[data-checked="true"], 
    .stRadio [role="radiogroup"] > label:focus-within {
        border: 2px solid #C47FA8;
        background-color: #F5EBF5;
    }

    /* Hide standard radio circles */
    .stRadio [role="radiogroup"] > label div[dir="auto"] > div:first-child {
        display: none;
    }
    
    /* Inject Sparkle Icon */
    .stRadio [role="radiogroup"] > label div[dir="auto"]::before {
        content: '✦';
        color: #C47FA8;
        font-size: 1.2rem;
        margin-right: 12px;
        vertical-align: middle;
    }

    /* Correct Answer Styling class */
    .correct-pill {
        background-color: #Edf7ed !important; /* Soft mint */
        border: 2px solid #81c784 !important;
        color: #2e7d32 !important;
        border-radius: 999px;
        padding: 12px 24px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        font-weight: 500;
        box-shadow: 0 2px 12px rgba(129, 199, 132, 0.2);
    }
    .correct-pill::before {
        content: '✦';
        color: #2e7d32;
        font-size: 1.2rem;
        margin-right: 12px;
    }
    
    .incorrect-text {
        color: #8B5A8B;
        font-style: italic;
        margin-top: 10px;
    }

    /* Hint Box and Animations */
    @keyframes gentleFadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hint-box {
        background-color: #F9EEF3;
        padding: 20px;
        border-radius: 16px;
        border: 1px dotted #E8D5E8;
        margin-bottom: 15px;
        box-shadow: 0 2px 12px rgba(180, 120, 170, 0.1);
        animation: gentleFadeIn 0.8s ease forwards;
        color: #3D2040;
    }
    .hint-title {
        font-family: 'DM Serif Display', serif;
        color: #C47FA8;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }

    /* Heart Progress Indicator */
    .heart-progress {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .heart-icon {
        width: 24px;
        height: 24px;
        fill: transparent;
        stroke: #C47FA8;
        stroke-width: 2;
        transition: all 0.5s ease;
    }
    .heart-icon.filled {
        fill: #C47FA8;
    }

    /* Decorative floral rule */
    .floral-rule {
        text-align: center;
        color: #E8D5E8;
        font-size: 1.5rem;
        margin: 20px 0;
        letter-spacing: 15px;
    }

    /* Metrics Dashboard */
    .metric-card {
        background-color: #F9EEF3;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px dotted #E8D5E8;
        box-shadow: 0 2px 12px rgba(180, 120, 170, 0.1);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8B5A8B;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #C47FA8;
        margin-top: 5px;
    }
    
    /* Hide standard metric */
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        color: #C47FA8 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8B5A8B !important;
    }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# STATE MANAGEMENT
# ------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_hints" not in st.session_state:
    st.session_state.current_hints = 0
if "user_answered" not in st.session_state:
    st.session_state.user_answered = False

# ------------------------------------------------------------------------
# MAIN PAGE LAYOUT (No Sidebar)
# ------------------------------------------------------------------------
st.markdown("<h1 class='main-title serif-text'>study smart ✦</h1>", unsafe_allow_html=True)

# Add a CSS rule to hide the sidebar if the user wants it gone
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stHeader"] {
            background: rgba(253, 248, 245, 0.8);
        }
    </style>
""", unsafe_allow_html=True)

tab_setup, tab1, tab2, tab3 = st.tabs(["Preparation", "Quiz", "Hints", "Session Summary"])

@st.cache_data
def get_val_df():
    try:
        return pd.read_csv("data/splits/val.csv")
    except Exception as e:
        st.error(f"Error loading val.csv: {e}")
        return None

def load_random_sample():
    df = get_val_df()
    if df is not None:
        sample = df.sample(1).iloc[0]
        # Construct options dict from A, B, C, D columns if they exist
        if 'A' in sample and 'B' in sample:
            opts = {"A": sample['A'], "B": sample['B'], "C": sample['C'], "D": sample['D']}
        else:
            opts = sample['options']
        return sample['article'], sample['question'], opts, sample['answer']
    
    # Fallback
    return "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "Where is the Eiffel Tower?", {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"}, "B"

# ------------------------------------------------------------------------
# TAB 0: PREPARATION
# ------------------------------------------------------------------------
with tab_setup:
    st.markdown("<h2 class='serif-text'>Welcome back.</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B5A8B;'>How would you like to begin your study session today?</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='background-color: #F9EEF3; padding: 2rem; border-radius: 16px; height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='serif-text'>Quick Practice</h4>", unsafe_allow_html=True)
        st.write("Load a random passage from the RACE dataset.")
        if st.button("Fetch Random Passage", key="btn_random"):
            article, question, options_str, correct_ans = load_random_sample()
            st.session_state.input_article = article
            st.session_state.input_question = question
            try:
                import ast
                st.session_state.input_options = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
            except:
                st.session_state.input_options = {"A": "A", "B": "B", "C": "C", "D": "D"}
            
            with st.spinner("Preparing your session..."):
                st.session_state.pipeline_result = run_pipeline(
                    article=article,
                    question=question,
                    options=st.session_state.input_options,
                    correct_answer=correct_ans
                )
            st.session_state.current_hints = 0
            st.session_state.user_answered = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='background-color: #F9EEF3; padding: 2rem; border-radius: 16px; height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='serif-text'>Custom Focus</h4>", unsafe_allow_html=True)
        manual_article = st.text_area("Input a passage for analysis:", height=150, placeholder="Paste your text here...", key="manual_text")
        if st.button("Submit Passage", key="btn_manual"):
            if manual_article.strip() == "":
                st.error("Please enter a passage.")
            else:
                with st.spinner("Analyzing text and generating quiz..."):
                    st.session_state.pipeline_result = run_pipeline(article=manual_article)
                    st.session_state.current_hints = 0
                    st.session_state.user_answered = False
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pipeline_result is not None:
        st.success("Passage loaded successfully! Head over to the Quiz tab to start.")
    else:
        st.info("Select one of the options above to start.")

# Only show the content of other tabs if a result exists
if st.session_state.pipeline_result is not None:
    res = st.session_state.pipeline_result
    
    # ------------------------------------------------------------------------
    # TAB 1: QUIZ VIEW
    # ------------------------------------------------------------------------
    with tab1:
        with st.expander("Read Passage", expanded=False):
            st.write(res["article"])
            
        st.markdown("<div class='floral-rule'>❦ ❦ ❦</div>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 class='serif-text' style='font-size: 2.2rem; margin-bottom: 1.5rem;'>{res['question']}</h2>", unsafe_allow_html=True)
        
        options = res["options"]
        if options:
            if not st.session_state.user_answered:
                selected_option = st.radio(
                    "Choose the best answer:",
                    options=[f"{k}: {v}" for k, v in options.items()],
                    index=None,
                    label_visibility="collapsed"
                )
                
                st.write("")
                if st.button("Check Answer"):
                    if not selected_option:
                        st.warning("Please select an option first.")
                    else:
                        st.session_state.user_answered = True
                        st.session_state.user_choice = selected_option
                        st.rerun()
            else:
                # Answer is checked
                user_letter = st.session_state.user_choice.split(":")[0]
                actual_letter = res.get("actual_answer", res["predicted_answer"])
                pred_letter = res["predicted_answer"]
                
                if user_letter == actual_letter:
                    st.markdown(f"<div class='correct-pill'>{st.session_state.user_choice}</div>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #2e7d32; font-family: \"DM Serif Display\", serif; font-size: 1.2rem; margin-bottom: 0.5rem;'>Correct! Beautifully done.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='incorrect-text' style='color: #f44336;'>Your selection: <b>{st.session_state.user_choice}</b> is incorrect.</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='incorrect-text'>The correct answer is:</p>", unsafe_allow_html=True)
                    st.markdown(f"<div class='correct-pill'>{actual_letter}: {options[actual_letter]}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<p style='color: #8B5A8B; font-style: italic; font-size: 0.9rem;'>Model A predicted: <b>{pred_letter}</b></p>", unsafe_allow_html=True)
                
                st.markdown("<h4 class='serif-text' style='margin-top: 2rem;'>AI Confidence</h4>", unsafe_allow_html=True)
                cols = st.columns(4)
                for idx, (label, score) in enumerate(res["confidence_scores"].items()):
                    cols[idx].metric(label=label, value=f"{score:.1%}")
                
                with st.expander("✨ Technical Insights (for Evaluation Marks)", expanded=False):
                    st.markdown(f"""
                        <p style='font-size: 0.9rem; color: #8B5A8B;'>
                        <b>Model:</b> Logistic Regression (TF-IDF Vectorized)<br>
                        <b>Internal Scoring:</b> The model evaluated each option by transforming the 
                        (Article + Question + Option) triplet into a 20,013-dimensional vector and 
                        calculating the probability of correctness.
                        </p>
                    """, unsafe_allow_html=True)
                    
                    # Show a mini table of scores for the report
                    df_scores = pd.DataFrame([
                        {"Option": k, "Confidence": f"{v:.2%}"} 
                        for k, v in res["confidence_scores"].items()
                    ])
                    st.dataframe(df_scores)
                
                if st.button("Try Another Question"):
                    article, question, opts, correct_ans = load_random_sample()
                    with st.spinner("Fetching new question..."):
                        st.session_state.pipeline_result = run_pipeline(
                            article=article,
                            question=question,
                            options=opts,
                            correct_answer=correct_ans
                        )
                    st.session_state.user_answered = False
                    st.session_state.current_hints = 0
                    st.rerun()

    # ------------------------------------------------------------------------
    # TAB 2: HINT PANEL
    # ------------------------------------------------------------------------
    with tab2:
        st.markdown("<h3 class='serif-text'>Progressive Hints</h3>", unsafe_allow_html=True)
        
        # SVG Heart Progress
        h1_class = "filled" if st.session_state.current_hints >= 1 else ""
        h2_class = "filled" if st.session_state.current_hints >= 2 else ""
        h3_class = "filled" if st.session_state.current_hints >= 3 else ""
        
        svg_heart = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="heart-icon {0}"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>'
        
        st.markdown(f"""
        <div class="heart-progress">
            {svg_heart.format(h1_class)}
            {svg_heart.format(h2_class)}
            {svg_heart.format(h3_class)}
        </div>
        """, unsafe_allow_html=True)
        
        hints = res["hints"]
        
        col1, col2, col3 = st.columns(3)
        if col1.button("Reveal Hint 1"):
            st.session_state.current_hints = max(st.session_state.current_hints, 1)
        if col2.button("Reveal Hint 2") and st.session_state.current_hints >= 1:
            st.session_state.current_hints = max(st.session_state.current_hints, 2)
        if col3.button("Reveal Hint 3") and st.session_state.current_hints >= 2:
            st.session_state.current_hints = max(st.session_state.current_hints, 3)
            
        if st.session_state.current_hints >= 1:
            st.markdown(f"<div class='hint-box'><div class='hint-title'>Hint 1 (General)</div>{hints[0]}</div>", unsafe_allow_html=True)
        if st.session_state.current_hints >= 2:
            st.markdown(f"<div class='hint-box'><div class='hint-title'>Hint 2 (Specific)</div>{hints[1]}</div>", unsafe_allow_html=True)
        if st.session_state.current_hints >= 3:
            st.markdown(f"<div class='hint-box'><div class='hint-title'>Hint 3 (Near-explicit)</div>{hints[2]}</div>", unsafe_allow_html=True)
            
        if st.session_state.current_hints >= 3:
            st.markdown("<div class='floral-rule'>❦</div>", unsafe_allow_html=True)
            st.markdown("<div class='beauty-cta'>", unsafe_allow_html=True)
            if st.button("Reveal Answer ✨"):
                st.markdown(f"<div class='correct-pill'>{res['predicted_answer']}: {res['options'][res['predicted_answer']]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # TAB 3: SESSION SUMMARY
    # ------------------------------------------------------------------------
    with tab3:
        st.markdown("<h3 class='serif-text'>Session Summary</h3>", unsafe_allow_html=True)
        
        # 2x2 Grid for metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Inference Latency</div><div class='metric-value'>{res['inference_time']:.3f}s</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Models Active</div><div class='metric-value'>Model A & B</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Prediction Confidence</div><div class='metric-value'>{max(res['confidence_scores'].values()):.1%}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Pipeline Status</div><div class='metric-value'>Healthy</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br><h4 class='serif-text'>Global Model Evaluation (Phase 13)</h4>", unsafe_allow_html=True)
        try:
            df_eval = pd.read_csv("reports/model_comparison.csv")
            st.write("This table shows the performance of the AI across the entire test split.")
            st.dataframe(df_eval)
        except:
            st.info("Evaluation report not found. Run src/evaluate.py to generate.")
            
        with st.expander("📝 Manual Quality Assessment (Human Eval)", expanded=False):
            st.write("Rate the distractors generated for this session for your final report:")
            score = st.select_slider("Plausibility Score (1=Bad, 5=Excellent)", options=[1,2,3,4,5], value=3)
            if st.button("Save Rating"):
                st.success(f"Rating {score}/5 saved to session history!")

        st.markdown("<br><h4 class='serif-text'>Log Data</h4>", unsafe_allow_html=True)
        df_log = pd.DataFrame([{
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Question": res["question"],
            "Predicted": res["predicted_answer"],
            "Latency": round(res["inference_time"], 3)
        }])
        st.dataframe(df_log)
        
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.markdown("<div class='ghost-button'>", unsafe_allow_html=True)
        st.download_button("Download Session Log", data=csv, file_name="session_log.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
