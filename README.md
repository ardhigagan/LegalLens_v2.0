<div>

<br/>

```
██╗     ███████╗ ██████╗  █████╗ ██╗     ██╗     ███████╗███╗   ██╗███████╗
██║     ██╔════╝██╔════╝ ██╔══██╗██║     ██║     ██╔════╝████╗  ██║██╔════╝
██║     █████╗  ██║  ███╗███████║██║     ██║     █████╗  ██╔██╗ ██║███████╗
██║     ██╔══╝  ██║   ██║██╔══██║██║     ██║     ██╔══╝  ██║╚██╗██║╚════██║
███████╗███████╗╚██████╔╝██║  ██║███████╗███████╗███████╗██║ ╚████║███████║
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### ⚖ AI-Powered Contract Analysis · Risk Detection · Legal Chat

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ardhigagan/LegalLens_v2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Model-Legal--BERT%20LoRA-C9A84C?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ardhigagan/LegalLens_v2.0)

<br/>

> **LegalLens AI democratizes legal literacy.**
> Upload any contract. Get an instant summary, flagged risks, and chat with your document — all powered by AI trained on real legal data.

<br/>

</div>

---

## The Problem

Legal contracts are written to be confusing. The average person signs documents they don't understand — burying clauses about automatic renewals, IP transfers, non-compete restrictions, and financial penalties in walls of dense legal text. Lawyers charge hundreds per hour to review what LegalLens does in seconds.

---

## What LegalLens Does

<table>
<tr>
<td width="50%">

**📄 Universal Ingestion**
Upload native PDFs, scanned documents, or images. Adaptive OCR preprocessing handles even low-quality scans.

**🔍 20-Category Risk Detection**
Flags Financial Penalties, IP Transfers, Non-Compete clauses, Automatic Renewals, and 16 more — with confidence scores.

**🤖 Fine-tuned Legal AI**
Risk detection powered by Legal-BERT trained on 510 real commercial contracts (CUAD dataset) using LoRA.

</td>
<td width="50%">

**💬 Chat with your Contract**
Ask natural language questions. Answers are grounded in the document with clause citations via RAG + FAISS.

**📊 Executive Summary**
Concise AI-generated summary of the entire contract using BART.

**📥 PDF Report Export**
Professional risk report with evidence snippets — ready to share with stakeholders.

</td>
</tr>
</table>

---

## Live Demo

**🚀 Try it now → [huggingface.co/spaces/ardhigagan/LegalLens_v2.0](https://huggingface.co/spaces/ardhigagan/LegalLens_v2.0)**

No signup required. Upload any PDF contract and get results in under a minute.

---

## Model Performance

Fine-tuned `nlpaueb/legal-bert-base-uncased` on CUAD v1 using LoRA (Low-Rank Adaptation). Only **0.28% of model parameters** were trained — making it feasible on CPU.

| Metric | Zero-Shot Baseline | Fine-Tuned (LegalLens) | Improvement |
|---|---|---|---|
| F1 Micro | 0.115 | **0.629** | +0.514 |
| F1 Macro | 0.067 | **0.245** | +0.177 |
| Precision | 0.437 | **0.647** | +0.210 |
| Recall | 0.066 | **0.611** | +0.545 |

**Top performing labels:**

| Clause Type | F1 Score |
|---|---|
| Governing Law Restriction | 0.938 |
| Assignment of Rights | 0.868 |
| Intellectual Property Transfer | 0.787 |
| Limitation of Liability | 0.778 |
| Financial Penalty | 0.574 |

---

## Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      LegalLens AI v2                        │
├─────────────────┬───────────────────────────────────────────┤
│ Frontend        │ Streamlit · Custom dark premium UI        │
│ Summarization   │ facebook/bart-large-cnn                   │
│ Risk Detection  │ nlpaueb/legal-bert-base-uncased + LoRA    │
│ Zero-Shot       │ facebook/bart-large-mnli (fallback)       │
│ RAG Embeddings  │ sentence-transformers/all-MiniLM-L6-v2    │
│ Vector Store    │ FAISS                                     │
│ Chat LLM        │ Groq (Llama 3.1 8B) · free 14k req/day   │
│ OCR             │ Tesseract · OpenCV · pdfplumber           │
│ Chunking        │ NLTK sentence-aware + LangChain           │
│ PDF Export      │ ReportLab                                 │
│ Fine-tuning     │ HuggingFace Transformers + PEFT/LoRA      │
│ Training Data   │ CUAD v1 · 510 contracts · 41 categories  │
└─────────────────┴───────────────────────────────────────────┘
```

---

## Project Structure

```
LegalLens_v2.0/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── analysis.py                 # Summarization + risk detection
│   │                               # (auto-loads fine-tuned or zero-shot)
│   ├── ingestion.py                # PDF & image OCR extraction
│   ├── processing.py               # Sentence-aware chunking
│   ├── rag.py                      # RAG chat (FAISS + Groq/Claude)
│   └── report.py                   # PDF report generation (ReportLab)
│
└── finetune/
    ├── 1_prepare_dataset.py        # CUAD dataset preparation
    ├── 2_finetune_lora.py          # LoRA fine-tuning on Legal-BERT
    ├── 3_evaluate.py               # Evaluation vs zero-shot baseline
    ├── 4_integrate.py              # Plug fine-tuned model into app
    └── requirements_finetune.txt   # Fine-tuning dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Tesseract OCR

```bash
# Windows — download from https://github.com/UB-Mannheim/tesseract/wiki
# Mac
brew install tesseract
# Linux
sudo apt-get install tesseract-ocr poppler-utils
```

### Installation

```bash
# Clone
git clone https://github.com/ardhigagan/LegalLens_v2.0.git
cd LegalLens_v2.0

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get a **free** Groq key at [console.groq.com](https://console.groq.com) — 14,400 requests/day, no credit card needed.

### Run

```bash
python -m streamlit run app.py
```

Open [http://localhost:8080](http://localhost:8080)

---

## Fine-tuning (Optional)

Train your own legal risk classifier on the CUAD dataset:

```bash
cd finetune
pip install -r requirements_finetune.txt

# Download CUAD_v1.json from https://zenodo.org/records/4595826
# Place at finetune/data/CUAD_v1.json

python 1_prepare_dataset.py    # Prepare dataset
python 2_finetune_lora.py      # Train (~2-4 hrs on CPU)
python 3_evaluate.py           # Evaluate results
python 4_integrate.py          # Plug into app
```

LoRA trains only **310K out of 110M parameters** — no GPU required.

---

## Architecture

```
                    ┌──────────────────┐
                    │  Upload Contract │
                    │  PDF / PNG / JPG │
                    └────────┬─────────┘
                             │
               ┌─────────────▼─────────────┐
               │       Ingestion Layer      │
               │  pdfplumber + Tesseract    │
               │  Adaptive OCR Preprocess   │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │       Chunking Layer       │
               │  NLTK Sentence-Aware Split │
               │  1800-char + overlap       │
               └──────┬──────────┬──────────┘
                      │          │
          ┌───────────▼─┐    ┌───▼──────────────┐
          │ Summarizer  │    │  Risk Detector    │
          │ bart-large  │    │  legal-bert+LoRA  │
          │    -cnn     │    │  (zero-shot fb)   │
          └───────┬─────┘    └────────┬──────────┘
                  │                   │
          ┌───────▼───────────────────▼──────────┐
          │           Aggregation Layer           │
          │   Dedup · Sort by Confidence Score    │
          │   FAISS Vector Index for RAG Chat     │
          └───────────────────┬──────────────────┘
                              │
          ┌───────────────────▼──────────────────┐
          │            Streamlit UI               │
          │  Risk Analysis · Chat · PDF Export    │
          └──────────────────────────────────────┘
```

---

## Roadmap

- [x] Universal PDF/image ingestion with OCR
- [x] AI executive summarization (BART)
- [x] 20-category risk detection
- [x] Sentence-aware chunking
- [x] RAG "Chat with Contract" (FAISS + Groq)
- [x] PDF risk report export
- [x] Fine-tuned Legal-BERT on CUAD (LoRA)
- [x] Premium dark UI redesign
- [x] HuggingFace Spaces deployment
- [ ] Multi-document comparison
- [ ] User accounts & contract history
- [ ] FastAPI backend + React frontend
- [ ] Batch contract processing
- [ ] Custom risk label configuration

---

## Disclaimer

LegalLens AI is intended for informational purposes only. It does not constitute legal advice. Always consult a qualified legal professional before making decisions based on this analysis.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by Ardhi Gagan**

[![Live Demo](https://img.shields.io/badge/Try%20It%20Live-HuggingFace-FF9D00?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/ardhigagan/LegalLens_v2.0)

*Star this repo if LegalLens helped you*  ⭐

</div>
