# frontend/streamlit_app.py
import os
from urllib.parse import urljoin
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ML metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
)

# Safety: prefer pure-Python protobuf implementation if present (avoid some segfaults)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

st.set_page_config(page_title="SentimentFlow", layout="wide")
st.title("📰 SentimentFlow — AI-Powered News Sentiment Analyzer")

st.markdown(
    """
Safe startup: heavy ML libs load in the backend. This frontend lazy-loads only what's needed.
- Paste text
- Single URL (summary + sentiment)
- CSV batch
"""
)

# ---------------------------
# Backend URL normalization
# ---------------------------
# Priority: Streamlit secrets -> environment -> default localhost
BACKEND_URL = st.secrets.get("BACKEND_URL", os.environ.get("BACKEND_URL", "http://localhost:8000"))
# If user accidentally supplied huggingface.co/spaces/... convert to hf.space runtime URL
if "huggingface.co/spaces/" in BACKEND_URL:
    # example: https://huggingface.co/spaces/USER/NAME -> https://USER-NAME.hf.space
    try:
        path = BACKEND_URL.rstrip("/").split("huggingface.co/spaces/")[1]
        domain = path.replace("/", "-")
        BACKEND_URL = f"https://{domain}.hf.space"
    except Exception:
        pass

# normalize scheme
if not BACKEND_URL.startswith("http"):
    BACKEND_URL = "https://" + BACKEND_URL

def build_url(path: str) -> str:
    return urljoin(BACKEND_URL.rstrip("/") + "/", path.lstrip("/"))

# ---------------------------
# Helper: call backend
# ---------------------------
def call_backend(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    url = build_url(endpoint)
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return {"error": f"Request exception: {e}"}
    # try parse JSON, else include raw text
    try:
        data = resp.json()
    except Exception:
        data = {"_raw_text": resp.text}
    if resp.status_code >= 400:
        return {"error": f"{resp.status_code} {resp.reason}", "details": data}
    return data

# ---------------------------
# Utilities: lazy imports
# ---------------------------
def ensure_nltk_tokenizers():
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

def fetch_article_summary(url, sentences=3):
    # lazy import to reduce startup hazards
    from newspaper import Article
    ensure_nltk_tokenizers()
    art = Article(url)
    art.download()
    art.parse()
    try:
        art.nlp()
    except Exception:
        pass
    summary = (art.summary or "").strip()
    if not summary:
        from nltk import sent_tokenize
        text_full = art.text or ""
        sents = sent_tokenize(text_full)
        summary = " ".join(sents[:sentences]) if sents else text_full[:1000]
    return art, summary

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------------------
# UI: Mode selection
# ---------------------------
mode = st.radio("Choose input type:", ["Paste Text", "Single URL", "Batch (CSV Upload)"], index=0)

# ---------------------------
# Mode 1: Paste text
# ---------------------------
if mode == "Paste Text":
    txt = st.text_area("Enter text for sentiment analysis", height=220)
    if st.button("Analyze Text"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Calling backend..."):
                result = call_backend("/analyze", {"text": txt}, timeout=300)
            if "error" in result:
                # show details if available
                st.error(f"Backend error: {result['error']}")
                if result.get("details"):
                    st.json(result["details"])
            else:
                st.subheader("Result")
                st.json(result)

# ---------------------------
# Mode 2: Single URL
# ---------------------------
elif mode == "Single URL":
    url = st.text_input("Enter article URL")
    if st.button("Fetch & Analyze Article"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching article..."):
                try:
                    art, summary = fetch_article_summary(url)
                except Exception as e:
                    st.error(f"Failed to fetch article: {e}")
                    art, summary = None, ""
            if not summary:
                st.warning("Could not extract summary from the URL.")
            else:
                with st.spinner("Calling backend..."):
                    result = call_backend("/analyze", {"text": summary}, timeout=300)
                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                    if result.get("details"):
                        st.json(result["details"])
                else:
                    st.subheader("Article Title")
                    st.write(art.title if art else "—")
                    st.subheader("Article Summary")
                    st.write(summary)
                    st.subheader("Sentiment Result")
                    st.json(result)

# ---------------------------
# Mode 3: Batch CSV
# ---------------------------
elif mode == "Batch (CSV Upload)":
    st.info("CSV must have a column named 'url'. Optionally include a 'true_label' column (POSITIVE/NEGATIVE or 1/0).")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = pd.DataFrame()

        if "url" not in df.columns:
            st.error("CSV must have a column named 'url'.")
        else:
            # batch size for chunking
            chunk_size = st.number_input("Chunk size for batch requests", min_value=10, max_value=200, value=50, step=10)
            if st.button("Analyze All URLs"):
                texts = []
                with st.spinner("Fetching articles..."):
                    for u in df["url"].fillna(""):
                        if not u:
                            texts.append("")
                            continue
                        try:
                            _, txt = fetch_article_summary(u)
                        except Exception:
                            txt = ""
                        texts.append(txt)

                # chunk and send
                all_results = []
                failed = False
                with st.spinner("Calling backend for batch prediction..."):
                    for chunk in chunk_list(texts, chunk_size):
                        resp = call_backend("/batch", {"texts": chunk}, timeout=300)
                        if "error" in resp:
                            st.error(f"Backend error during chunk: {resp['error']}")
                            if resp.get("details"):
                                st.json(resp["details"])
                            failed = True
                            break
                        if "results" not in resp:
                            st.error("Backend returned unexpected response (no 'results').")
                            failed = True
                            break
                        all_results.extend(resp["results"])

                if not failed:
                    df_results = pd.DataFrame(all_results)
                    if df_results.empty:
                        st.error("Backend returned empty/invalid results.")
                    else:
                        df_out = pd.concat([df.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)
                        st.subheader("Batch Results")
                        st.dataframe(df_out)

                        # validate
                        if "label" not in df_out.columns or "score" not in df_out.columns:
                            st.warning("Backend results missing 'label' or 'score' fields. Metrics unavailable.")
                        else:
                            # compute prob_pos if provided else approximate
                            def get_prob_pos(row):
                                if "prob_pos" in row and pd.notna(row["prob_pos"]):
                                    try:
                                        return float(row["prob_pos"])
                                    except Exception:
                                        pass
                                lbl = str(row["label"]).upper()
                                try:
                                    s = float(row["score"])
                                except Exception:
                                    s = 0.0
                                return s if lbl == "POSITIVE" else (1.0 - s)

                            df_out["pred_prob_pos"] = df_out.apply(get_prob_pos, axis=1)
                            df_out["pred_label_norm"] = df_out["label"].astype(str).str.upper()

                            # Sentiment distribution (counts)
                            st.subheader("Sentiment Distribution (counts)")
                            counts = df_out["pred_label_norm"].value_counts()
                            fig, ax = plt.subplots()
                            counts.plot(kind="bar", ax=ax)
                            ax.set_xlabel("Predicted label")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                            # Histogram of predicted positive probabilities
                            st.subheader("Predicted Positive Probability (score) Distribution")
                            fig2, ax2 = plt.subplots()
                            ax2.hist(df_out["pred_prob_pos"].dropna(), bins=25, range=(0,1))
                            ax2.set_xlabel("Probability of Positive")
                            ax2.set_ylabel("Number of examples")
                            st.pyplot(fig2)

                            # metrics (if true labels exist)
                            true_col_candidates = [c for c in df_out.columns if c.lower() in ("true_label", "label_true", "gold", "target")]
                            if true_col_candidates:
                                true_col = true_col_candidates[0]
                                st.markdown(f"**Using true labels from column:** `{true_col}`")

                                def to_binary_true(v):
                                    if pd.isna(v):
                                        return None
                                    s = str(v).strip()
                                    if s.isdigit():
                                        return int(s)
                                    s_up = s.upper()
                                    if s_up in ("POSITIVE", "POS", "P", "1", "TRUE", "T"):
                                        return 1
                                    if s_up in ("NEGATIVE", "NEG", "N", "0", "FALSE", "F"):
                                        return 0
                                    try:
                                        fv = float(s)
                                        return 1 if fv >= 0.5 else 0
                                    except Exception:
                                        return None

                                df_out["true_binary"] = df_out[true_col].apply(to_binary_true)
                                eval_df = df_out.dropna(subset=["true_binary"])
                                if eval_df.shape[0] < 2:
                                    st.warning("Not enough valid true-labeled rows to compute metrics (need >=2).")
                                else:
                                    y_true = eval_df["true_binary"].astype(int).values
                                    y_pred = (eval_df["pred_label_norm"] == "POSITIVE").astype(int).values
                                    y_score = eval_df["pred_prob_pos"].astype(float).values

                                    precision, recall, f1, _ = precision_recall_fscore_support(
                                        y_true, y_pred, average="binary", zero_division=0
                                    )
                                    st.metric("Precision", f"{precision:.3f}")
                                    st.metric("Recall", f"{recall:.3f}")
                                    st.metric("F1", f"{f1:.3f}")

                                    st.text("Classification Report:")
                                    st.text(classification_report(y_true, y_pred, digits=3))

                                    # Confusion matrix (seaborn optional)
                                    try:
                                        import seaborn as sns
                                        from sklearn.metrics import confusion_matrix
                                        cm = confusion_matrix(y_true, y_pred)
                                        fig3, ax3 = plt.subplots()
                                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
                                        ax3.set_xlabel("Predicted")
                                        ax3.set_ylabel("True")
                                        ax3.set_xticklabels(["NEG", "POS"])
                                        ax3.set_yticklabels(["NEG", "POS"])
                                        st.subheader("Confusion Matrix")
                                        st.pyplot(fig3)
                                    except Exception:
                                        from sklearn.metrics import confusion_matrix
                                        st.text("Confusion matrix (sklearn):")
                                        st.write(confusion_matrix(y_true, y_pred))

                                    # ROC-AUC & ROC curve
                                    try:
                                        if len(set(y_true)) == 2:
                                            roc_auc = roc_auc_score(y_true, y_score)
                                            st.metric("ROC-AUC", f"{roc_auc:.3f}")

                                            fpr, tpr, thresholds = roc_curve(y_true, y_score)
                                            roc_auc_val = auc(fpr, tpr)
                                            fig4, ax4 = plt.subplots()
                                            ax4.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
                                            ax4.plot([0, 1], [0, 1], "k--", lw=0.8)
                                            ax4.set_xlabel("False Positive Rate")
                                            ax4.set_ylabel("True Positive Rate")
                                            ax4.set_title("ROC Curve")
                                            ax4.legend(loc="lower right")
                                            st.pyplot(fig4)
                                        else:
                                            st.info("ROC-AUC requires both positive and negative true labels present.")
                                    except Exception as e:
                                        st.error(f"Could not compute ROC-AUC: {e}")
                            else:
                                st.info("No true_label column detected in uploaded CSV. Metrics require ground-truth labels.")

                        # download results
                        st.download_button("Download Results as CSV", df_out.to_csv(index=False), "sentiment_results.csv")
