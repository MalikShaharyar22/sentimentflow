# 📰 SentimentFlow — AI-Powered News Sentiment Analyzer

SentimentFlow is an end-to-end **AI pipeline** that ingests raw text or news articles, processes them through a **Transformer-based sentiment analysis model**, and presents interactive insights via a **Streamlit dashboard**.  

It demonstrates **data engineering, natural language processing (NLP), machine learning, backend services, and frontend integration** — aligned with modern **AI Engineer** workflows.

---

## 🚀 Features
- **Data Engineering (ETL):**
  - Fetch news articles via URLs using `newspaper3k`
  - Clean, preprocess, and tokenize text using `nltk`
  - Batch ingestion support (CSV with multiple URLs)

- **Machine Learning / NLP:**
  - Transformer model: `distilbert-base-uncased-finetuned-sst-2-english`
  - Hugging Face `pipeline` for inference
  - Optional fine-tuning for domain-specific datasets (finance, pharma, manufacturing)
  - Sentence-level sentiment breakdown

- **Backend (FastAPI):**
  - `/analyze` → single text/article
  - `/batch` → multiple documents
  - JSON responses with labels + confidence scores

- **Frontend (Streamlit):**
  - Multi-input: paste text, article URL, or CSV upload
  - Sentiment visualization (bar charts, histograms, ROC curves)
  - Downloadable results
  - Automatic fallback summarization (NLTK sentence extraction)

- **Robustness:**
  - Auto-downloads NLTK resources (`punkt`, `punkt_tab`)
  - Handles parsing errors gracefully
  - Lightweight deployment with Docker & Streamlit Cloud

---

## 🧮 Mathematical Intuitions

### 1. Transformer Sentiment Model
- **DistilBERT** encodes input text into contextual embeddings:
  \\( h = \\text{Transformer}(x) \\)
- A classification head maps the [CLS] embedding to sentiment logits:
  \\( z = W h_{[CLS]} + b \\)

### 2. Softmax for Probabilities
- Converts logits into probabilities:
  \\[ P(y|x) = \\frac{e^{z_y}}{\\sum_j e^{z_j}} \\]
- Labels: **POSITIVE** or **NEGATIVE**

### 3. Training Objective (Cross-Entropy Loss)
\\[ \\mathcal{L} = - \\sum_{i} y_i \\log(\\hat{y}_i) \\]

- Encourages predicted distribution \\( \\hat{y} \\) to match true distribution \\( y \\).

### 4. Evaluation Metrics
- **Accuracy**: correct predictions / total
- **Precision**: TP / (TP + FP) → reliability of positive predictions
- **Recall**: TP / (TP + FN) → ability to detect positives
- **F1 Score**: harmonic mean of Precision & Recall
- **ROC-AUC**: probability the model ranks a random positive higher than a negative
- **Confusion Matrix**: visualization of TP, FP, TN, FN

### 5. Pipeline Intuition
\\[ \\text{Article URL} \\to \\text{Text Extraction} \\to \\text{Tokenizer} \\to \\text{Transformer} \\to \\text{Softmax} \\to \\text{Label + Score} \\]

---

## ⚙️ Tech Stack
- **Language:** Python 3.11
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **ML/DL:** Hugging Face Transformers, PyTorch
- **NLP:** NLTK, newspaper3k
- **Data Handling:** pandas, matplotlib, seaborn
- **Deployment:** Docker, Hugging Face Spaces, Streamlit Cloud

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
```bash
python setup_nltk.py
```

### 5. Run backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 6. Run frontend (Streamlit)
```bash
cd frontend
streamlit run streamlit_app.py
```

---

## 📊 Example Usage
- **Text:** *"I loved this product, it was excellent!"* → POSITIVE  
- **News URL:** Summarizes + classifies sentiment  
- **Batch CSV:** Outputs predictions + distribution plots + metrics  

---

## 🔑 Skills Demonstrated
- **AI/ML:** Sentiment analysis, Transformer-based NLP  
- **Data Engineering:** ETL pipeline (fetch → preprocess → tokenize → model)  
- **Backend Development:** REST APIs with FastAPI  
- **Frontend Development:** Streamlit dashboards with visualization  
- **MLOps:** Dockerized deployment on Hugging Face Spaces  
- **Evaluation:** Precision, Recall, F1, ROC-AUC, Confusion Matrix  
- **Problem Solving:** Robust error handling & fallbacks  

---

## 🌟 Why this Project Stands Out
- Full **end-to-end AI pipeline**: ingestion → preprocessing → ML → API → UI  
- Combines **classic NLP (nltk)** with **modern Transformers**  
- Flexible deployment: local, cloud, Docker, Hugging Face Spaces  
- Matches real-world **AI Engineer responsibilities** in ETL, ML, APIs, frontend  

---

## 📌 Future Improvements
- Multi-class sentiment (Very Positive → Very Negative → Neutral)  
- Domain-specific fine-tuning (finance, pharma, manufacturing)  
- Topic modeling integration  
- Continuous monitoring dashboard for live sentiment trends  

---

## 📜 License
MIT License © 2025 Swapnil Mohanty
