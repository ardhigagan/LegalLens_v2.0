# LegalLens AI

AI-powered contract analysis — upload any contract, get an instant risk summary, and chat with your document.

**[Live Demo →](https://huggingface.co/spaces/ardhigagan/LegalLens_v2.0)**

---

## Overview

Legal contracts are written to be confusing. LegalLens uses fine-tuned legal AI to detect risks, summarize documents, and answer questions about your contracts in seconds.

- **Risk detection** — 20 clause categories with confidence scores (financial penalties, IP transfers, non-competes, automatic renewals, and more)
- **Document chat** — RAG-powered Q&A grounded in your contract with clause citations
- **Universal ingestion** — native PDFs, scanned documents, and images via adaptive OCR
- **Executive summary** — AI-generated overview using BART
- **PDF export** — professional risk report with evidence snippets

---

## Model Performance

Fine-tuned `nlpaueb/legal-bert-base-uncased` on CUAD v1 (510 contracts) using LoRA — only 0.28% of parameters trained.

| Metric | Zero-Shot | Fine-Tuned | Δ |
|---|---|---|---|
| F1 Micro | 0.115 | **0.629** | +0.514 |
| F1 Macro | 0.067 | **0.245** | +0.177 |
| Precision | 0.437 | **0.647** | +0.210 |
| Recall | 0.066 | **0.611** | +0.545 |

**Top clause performance:**

| Clause | F1 |
|---|---|
| Governing Law Restriction | 0.938 |
| Assignment of Rights | 0.868 |
| Intellectual Property Transfer | 0.787 |
| Limitation of Liability | 0.778 |
| Financial Penalty | 0.574 |

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Summarization | facebook/bart-large-cnn |
| Risk Detection | nlpaueb/legal-bert-base-uncased + LoRA |
| Fallback | facebook/bart-large-mnli (zero-shot) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Chat LLM | Groq — Llama 3.1 8B |
| OCR | Tesseract · OpenCV · pdfplumber |
| Chunking | NLTK + LangChain |
| PDF Export | ReportLab |
| Fine-tuning | HuggingFace Transformers + PEFT/LoRA |

---

## Getting Started

**Prerequisites:** Python 3.9+, Tesseract OCR

```bash
# Install Tesseract
brew install tesseract                          # macOS
sudo apt-get install tesseract-ocr poppler-utils  # Linux
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

```bash
# Clone and install
git clone https://github.com/ardhigagan/LegalLens_v2.0.git
cd LegalLens_v2.0
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```bash
# Configure — create a .env file
GROQ_API_KEY=gsk_your_key_here
```

Free Groq key at [console.groq.com](https://console.groq.com) — 14,400 requests/day, no credit card required.

```bash
python -m streamlit run app.py
# Open http://localhost:8080
```

---

## Fine-tuning (Optional)

Train your own legal risk classifier on the CUAD dataset. LoRA trains only 310K of 110M parameters — no GPU required.

```bash
cd finetune
pip install -r requirements_finetune.txt

# Download CUAD_v1.json from https://zenodo.org/records/4595826
# Place at finetune/data/CUAD_v1.json

python 1_prepare_dataset.py   # Prepare dataset
python 2_finetune_lora.py     # Train (~2–4 hrs on CPU)
python 3_evaluate.py          # Evaluate results
python 4_integrate.py         # Plug into app
```

---

## Project Structure

```
LegalLens_v2.0/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── src/
│   ├── analysis.py         # Summarization + risk detection
│   ├── ingestion.py        # PDF & image OCR extraction
│   ├── processing.py       # Sentence-aware chunking
│   ├── rag.py              # RAG chat (FAISS + Groq)
│   └── report.py           # PDF report generation
└── finetune/
    ├── 1_prepare_dataset.py
    ├── 2_finetune_lora.py
    ├── 3_evaluate.py
    └── 4_integrate.py
```

---

## Disclaimer

LegalLens is intended for informational purposes only and does not constitute legal advice. Always consult a qualified legal professional before making decisions based on this analysis.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Ardhi Gagan](https://github.com/ardhigagan)
