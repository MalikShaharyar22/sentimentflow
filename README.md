# 📰 SentimentFlow — AI-Powered News Sentiment Analyzer

SentimentFlow is an end-to-end **AI pipeline** that ingests raw text or news articles, processes them through a **Transformer-based sentiment analysis model**, and presents interactive insights via a **Streamlit dashboard**. It demonstrates **data engineering, machine learning, backend services, and frontend integration** — aligned with modern AI Engineer workflows.

---

## 🚀 Features
- **Data Engineering (ETL):**
  - Fetch news articles via URLs using `newspaper3k`
  - Clean, preprocess, and tokenize text using `nltk`
  - Support for batch ingestion (CSV with multiple URLs)

- **Machine Learning:**
  - Transformer-based sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`)
  - Hugging Face `pipeline` for inference
  - Optional fine-tuning on domain-specific datasets (finance, pharma, etc.)
  - Sentence-level sentiment breakdown

- **Backend (FastAPI):**
  - `/analyze` endpoint for single text
  - `/batch` endpoint for multiple documents
  - JSON responses with labels + confidence scores

- **Frontend (Streamlit):**
  - Modes: paste text, single URL, batch CSV upload
  - Sentiment visualization (bar charts, metrics)
  - Downloadable CSV outputs
  - Automatic fallback summarization (NLTK sentence extraction)

- **Robustness:**
  - Auto-downloads required NLTK resources (`punkt`, `punkt_tab`)
  - Handles failures gracefully with fallbacks
  - Lightweight deployment (Docker/Streamlit Cloud)

---

## 🧮 Mathematical Intuitions
- **Transformer Sentiment Model:**
  - Uses pretrained DistilBERT fine-tuned on SST-2
  - Classification into POSITIVE / NEGATIVE
  - Confidence via softmax:
    \[
    P(y|x) = \frac{e^{z_y}}{\sum_{j} e^{z_j}}
    \]
    where \(z\) are logits from the model

- **Evaluation Metrics:**
  - Polarity ∈ [-1, 1]
  - Accuracy, F1, and ROC-AUC (if fine-tuned)
  - Sentence-level polarity for fine-grained insights

- **ETL Pipeline:**
  - Article text → preprocessing → tokenization → Transformer → label + score
  - Batch mode → aggregation + visualization

---

## ⚙️ Tech Stack
- **Python** (3.11)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **ML/DL:** Hugging Face Transformers, PyTorch
- **NLP:** nltk, newspaper3k
- **Data Handling:** pandas, matplotlib
- **Deployment:** Docker / Streamlit Cloud / Local

---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/sentimentflow.git
cd sentimentflow
```

### 2. Create & activate virtual environment
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

### 4. Setup NLTK resources
Run the setup script (downloads punkt + punkt_tab):
```bash
python setup_nltk.py
```

### 5. Run backend (FastAPI)
```bash
cd backend
python -m uvicorn app:app --reload --port 8000
```
Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test.

### 6. Run frontend (Streamlit)
Open a new terminal, activate venv, then:
```bash
cd frontend
streamlit run streamlit_app.py
```
Visit [http://localhost:8501](http://localhost:8501) to use the app.

---

## 📊 Example Usage
- Paste text directly:
  - *"I loved this product, it was excellent!"* → **POSITIVE**
- Analyze a news URL:
  - Fetches article, summarizes, predicts sentiment
- Batch CSV upload:
  - Input: `urls.csv` with a column `url`
  - Output: CSV with labels + scores, plus sentiment distribution chart

---

## 🔑 Skills Demonstrated
- **AI/ML:** Sentiment analysis, Transformer models, Hugging Face
- **Data Engineering:** ETL pipelines, preprocessing, batch processing
- **Backend Development:** RESTful APIs with FastAPI
- **Frontend Development:** Interactive dashboards with Streamlit
- **MLOps:** Model serving, environment management, dependency handling
- **Visualization:** Metrics, bar charts, sentiment breakdowns
- **Problem Solving:** Robust handling of missing data, fallbacks

---

## 🌟 Why this Project Stands Out
- End-to-end pipeline: ingestion → preprocessing → ML → API → UI
- Uses both **classic NLP (nltk)** and **modern transformers**
- Flexible deployment (local, Streamlit Cloud, Docker)
- Aligned with **real-world AI Engineer responsibilities**

---

## 📌 Future Improvements
- Add domain-specific fine-tuned models (finance, pharma)
- Extend to multi-class sentiment (positive, negative, neutral)
- Add topic modeling alongside sentiment
- Integrate with live news APIs (e.g., NewsAPI)
- Deploy with Docker Compose (backend + frontend in one stack)

---

## 📜 License
MIT License. Free to use, modify, and share.
