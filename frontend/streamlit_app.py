import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
import streamlit as st
from newspaper import Article

# frontend/streamlit_app.py (safer, lazy-loading)
import os
# fallback to pure-Python protobuf implementation to avoid C-extension segfaults on some hosts
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Only import nltk/newspaper when needed to avoid heavy startup work.
# We'll lazy-import in the functions that need them.

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SentimentFlow", layout="wide")
st.title("📰 SentimentFlow — AI-Powered News Sentiment Analyzer")

st.markdown(
    """
Safe startup: heavy ML libs are loaded in backend. This frontend lazy-loads only what it needs.
- Paste text
- Single URL (summary + sentiment)
- CSV batch
"""
)


# -------------------------------
# Helper to call backend
# -------------------------------
def call_backend(endpoint, payload):
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# helper: safe extraction from backend result
def results_to_df(results):
    try:
        df = pd.DataFrame(results)
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------
# Mode selection
# -------------------------------
mode = st.radio("Choose input type:", ["Paste Text", "Single URL", "Batch (CSV Upload)"], index=0)

# -------------------------------
# Mode 1: Paste text
# -------------------------------
if mode == "Paste Text":
    txt = st.text_area("Enter text for sentiment analysis", height=200)
    if st.button("Analyze Text"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            result = call_backend("/analyze", {"text": txt})
            if "error" in result:
                st.error(f"Backend error: {result['error']}")
            else:
                st.subheader("Result")
                st.json(result)

# -------------------------------
# Mode 2: Single URL (improved with fallback summary)
# -------------------------------
elif mode == "Single URL":
    url = st.text_input("Enter article URL")
    if st.button("Fetch & Analyze Article"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                art = Article(url)
                art.download()
                art.parse()
                try:
                    art.nlp()
                except Exception:
                    pass

                # Try newspaper summary, else fallback to first sentences
                summary = (art.summary or "").strip()
                if not summary:
                    try:
                        nltk.data.find("tokenizers/punkt")
                    except Exception:
                        nltk.download("punkt")
                    from nltk import sent_tokenize
                    text_full = art.text or ""
                    sents = sent_tokenize(text_full)
                    summary = " ".join(sents[:3]) if sents else text_full[:1000]

                # Call backend for sentiment
                result = call_backend("/analyze", {"text": summary})

                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                else:
                    st.subheader("Article Title")
                    st.write(art.title or "—")

                    st.subheader("Sentiment Result")
                    st.json(result)

                    st.subheader("Article Summary")
                    st.write(summary)

            except Exception as e:
                st.error(f"Error fetching article: {e}")

# -------------------------------
# Mode 3: Batch (CSV Upload)
# -------------------------------
elif mode == "Batch (CSV Upload)":
    file = st.file_uploader("Upload a CSV file with a column named 'url' (optionally include a true_label column)", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = pd.DataFrame()

        if "url" not in df.columns:
            st.error("CSV must have a column named 'url'")
        else:
            if st.button("Analyze All URLs"):
                texts = []
                with st.spinner("Fetching articles..."):
                    for u in df["url"]:
                        try:
                            art = Article(u)
                            art.download()
                            art.parse()
                            try:
                                art.nlp()
                                txt = art.summary or art.text or ""
                            except Exception:
                                txt = art.summary or art.text or ""
                            texts.append(txt)
                        except Exception:
                            texts.append("")

                with st.spinner("Calling backend for batch prediction..."):
                    backend_resp = call_backend("/batch", {"texts": texts})

                if "error" in backend_resp:
                    st.error(f"Backend error: {backend_resp['error']}")
                elif "results" not in backend_resp:
                    st.error("Backend returned unexpected response (no 'results' field).")
                else:
                    df_results = results_to_df(backend_resp["results"])
                    if df_results.empty:
                        st.error("Backend returned empty or invalid results.")
                    else:
                        # merge results
                        df_out = pd.concat([df.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)
                        st.subheader("Batch Results")
                        st.dataframe(df_out)

                        # verify required columns
                        if "label" not in df_out.columns or "score" not in df_out.columns:
                            st.warning("Backend results missing 'label' or 'score' fields. Metrics unavailable.")
                        else:
                            # compute pred_prob_pos using explicit prob_pos if present
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

                            # 1) Sentiment distribution counts (bar)
                            st.subheader("Sentiment Distribution (counts)")
                            counts = df_out["pred_label_norm"].value_counts()
                            fig, ax = plt.subplots()
                            counts.plot(kind="bar", ax=ax)
                            ax.set_xlabel("Predicted label")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                            # 2) Histogram of predicted positive probabilities
                            st.subheader("Predicted Positive Probability (score) Distribution")
                            fig2, ax2 = plt.subplots()
                            ax2.hist(df_out["pred_prob_pos"].dropna(), bins=25, range=(0,1))
                            ax2.set_xlabel("Probability of Positive")
                            ax2.set_ylabel("Number of examples")
                            st.pyplot(fig2)

                            # find possible true label column
                            true_col_candidates = [c for c in df_out.columns if c.lower() in ("true_label", "label_true", "gold", "target")]
                            if true_col_candidates:
                                true_col = true_col_candidates[0]
                                st.markdown(f"**Using true labels from column:** `{true_col}`")

                                # normalize true labels
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

                                    # Precision, Recall, F1
                                    precision, recall, f1, _ = precision_recall_fscore_support(
                                        y_true, y_pred, average="binary", zero_division=0
                                    )
                                    st.metric("Precision", f"{precision:.3f}")
                                    st.metric("Recall", f"{recall:.3f}")
                                    st.metric("F1", f"{f1:.3f}")

                                    st.text("Classification Report:")
                                    st.text(classification_report(y_true, y_pred, digits=3))

                                    # Confusion matrix heatmap
                                    try:
                                        import seaborn as sns
                                        from sklearn.metrics import \
                                            confusion_matrix
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
                                        # seaborn optional; fall back to text matrix display
                                        from sklearn.metrics import \
                                            confusion_matrix
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

                        # Download results
                        st.download_button("Download Results as CSV", df_out.to_csv(index=False), "sentiment_results.csv")
