import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
import streamlit as st
from newspaper import Article

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# -------------------------------
# Config
# -------------------------------
BACKEND_URL = "http://localhost:8000"  

st.set_page_config(page_title="SentimentFlow", layout="wide")
st.title("📰 SentimentFlow — AI-Powered News Sentiment Analyzer")

st.markdown("""
Analyze sentiment of text or articles using a transformer model.
Options:
- Paste custom text
- Provide a single article URL
- Upload CSV with multiple URLs
""")

# -------------------------------
# Helper to call backend
# -------------------------------
def call_backend(endpoint, payload):
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=60)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

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
                    import nltk
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
    file = st.file_uploader("Upload a CSV file with a column named 'url'", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "url" not in df.columns:
            st.error("CSV must have a column named 'url'")
        else:
            if st.button("Analyze All URLs"):
                texts = []
                for u in df["url"]:
                    try:
                        art = Article(u)
                        art.download(); art.parse(); art.nlp()
                        texts.append(art.summary if art.summary else art.text)
                    except Exception:
                        texts.append("")
                result = call_backend("/batch", {"texts": texts})

                if "results" in result:
                    df_results = pd.DataFrame(result["results"])
                    df = pd.concat([df, df_results], axis=1)
                    st.subheader("Batch Results")
                    st.dataframe(df)

                    # Visualization
                    st.subheader("Sentiment Distribution")
                    fig, ax = plt.subplots()
                    df["label"].value_counts().plot(kind="bar", ax=ax)
                    st.pyplot(fig)

                    # Download results
                    st.download_button(
                        "Download Results as CSV",
                        df.to_csv(index=False),
                        "sentiment_results.csv"
                    )
                else:
                    st.error(f"Backend error: {result}")
