# LegalLens AI v2

**Intelligent Contract Analysis, Risk Detection & Chat**

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Summarization model | `distilbart-cnn-12-6` | `bart-large-cnn` (better quality) |
| Risk labels | 10 | 20 |
| Chunking | Character-based | Sentence-aware (NLTK) |
| Model loading | Re-loads every run | Cached with `@st.cache_resource` |
| OCR preprocessing | Fixed threshold | Adaptive threshold + denoising |
| Chat with contract | ❌ | ✅ RAG (FAISS + Claude/local QA) |
| PDF report export | ❌ | ✅ ReportLab |
| Risk deduplication | Basic | Fingerprint-based dedup |
| Session state | ❌ | ✅ Full session persistence |

---

## Project Structure

```
legallens_v2/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── packages.txt            # System deps (Tesseract, Poppler)
└── src/
    ├── __init__.py
    ├── ingestion.py        # PDF & image text extraction (OCR)
    ├── processing.py       # Sentence-aware chunking
    ├── analysis.py         # Summarization + risk detection
    ├── rag.py              # RAG chat (FAISS + Claude API)
    └── report.py           # PDF report generation (ReportLab)
```

---

## Setup

### Prerequisites
- Python 3.9+
- Tesseract OCR installed

### Install

```bash
git clone https://github.com/ardhigagan/LegalLens-AI.git
cd LegalLens-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python -m streamlit run app.py
```

---

## RAG Chat Feature

The Chat tab lets you ask questions about your contract in natural language.

**Without API key:** Uses `deepset/roberta-base-squad2` (local, free, extractive QA).

**With Anthropic API key:** Uses Claude for much richer, cited answers. Add your key in the sidebar.

---

## PDF Report

Click the **Export Report** tab after analysis to generate a professional PDF containing:
- Executive summary
- Risk overview table
- Per-risk evidence snippets
- Legal disclaimer

---

## Next Steps (Phase 3)

- Fine-tune on [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad) for higher accuracy
- Add user accounts & contract history (Supabase)
- REST API with FastAPI backend

---

## License
MIT — Built by Ardhi Gagan
