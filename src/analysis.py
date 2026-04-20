"""
analysis.py — LegalLens AI v3 (Phase 3)
Uses fine-tuned legal-bert + LoRA for risk detection.
Much higher accuracy than zero-shot classification.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel

device_id = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device_id == 0 else 'CPU'}")

# ── Risk labels ───────────────────────────────────────────────────────
RISK_LABELS = [
    "Financial Penalty", "Privacy Violation", "Non-Compete Restriction",
    "Termination Without Cause", "Intellectual Property Transfer",
    "Mandatory Arbitration", "Indemnification Obligation", "Unilateral Amendment",
    "Jurisdiction Waiver", "Automatic Renewal", "Limitation of Liability",
    "Liquidated Damages", "Force Majeure Exclusion", "Governing Law Restriction",
    "Non-Disparagement Clause", "Exclusivity Obligation",
    "Data Sharing with Third Parties", "Penalty for Early Termination",
    "Warranty Disclaimer", "Assignment of Rights",
]
LABEL2ID   = {label: i for i, label in enumerate(RISK_LABELS)}
NUM_LABELS = len(RISK_LABELS)
THRESHOLD  = 0.45   # Fine-tuned model threshold (lower than zero-shot)

# ── Model paths ───────────────────────────────────────────────────────
_LORA_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "finetune", "models", "legallens-legal-bert-lora"
)
_USE_FINETUNED = os.path.exists(_LORA_MODEL_DIR)

if _USE_FINETUNED:
    print(f"✅ Fine-tuned LoRA model found at {_LORA_MODEL_DIR}")
else:
    print("⚠️  Fine-tuned model not found. Falling back to zero-shot classification.")


# ── Model loaders ─────────────────────────────────────────────────────
def load_summarizer():
    """Load summarization model."""
    print("Loading Summarization Model (bart-large-cnn)...")
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn",
                        device=device_id, framework="pt")
    except KeyError:
        from transformers import BartForConditionalGeneration, BartTokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        return pipeline("text2text-generation", model=model,
                        tokenizer=tokenizer, device=device_id)


def load_risk_detector():
    """Load fine-tuned LoRA model if available, else zero-shot."""
    if _USE_FINETUNED:
        print("Loading fine-tuned legal-bert (LoRA)...")
        tokenizer = AutoTokenizer.from_pretrained(_LORA_MODEL_DIR)
        base = AutoModelForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, _LORA_MODEL_DIR)
        model.eval()
        return {"type": "finetuned", "model": model, "tokenizer": tokenizer}
    else:
        print("Loading zero-shot classifier (bart-large-mnli)...")
        clf = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli",
                       device=device_id, framework="pt")
        return {"type": "zeroshot", "classifier": clf}


# ── Inference ─────────────────────────────────────────────────────────
def _predict_finetuned(text: str, detector: dict) -> list[tuple[str, float]]:
    """Run fine-tuned model on a single text chunk."""
    tokenizer = detector["tokenizer"]
    model     = detector["model"]
    encoding  = tokenizer(text, max_length=512, truncation=True,
                          padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoding).logits
    probs = torch.sigmoid(logits).squeeze().numpy()
    results = []
    for i, prob in enumerate(probs):
        if prob > THRESHOLD:
            results.append((RISK_LABELS[i], float(prob)))
    return results


def _predict_zeroshot(text: str, detector: dict) -> list[tuple[str, float]]:
    """Run zero-shot classifier on a single text chunk."""
    clf    = detector["classifier"]
    result = clf(text[:1024], RISK_LABELS, multi_label=True)
    return [
        (label, score)
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.50
    ]


def _extract_best_snippet(text: str, label: str) -> str:
    keywords = {
        "Financial Penalty": ["penalty", "fine", "fee", "liquidated", "damages"],
        "Privacy Violation": ["privacy", "personal data", "gdpr", "collect", "share"],
        "Non-Compete Restriction": ["compete", "competitor", "non-compete", "solicit"],
        "Termination Without Cause": ["terminate", "termination", "without cause"],
        "Intellectual Property Transfer": ["intellectual property", "ip", "assign", "ownership"],
        "Mandatory Arbitration": ["arbitration", "arbitrate", "dispute"],
        "Indemnification Obligation": ["indemnif", "hold harmless", "defend"],
        "Unilateral Amendment": ["amend", "modify", "sole discretion"],
        "Jurisdiction Waiver": ["jurisdiction", "venue", "waive"],
        "Automatic Renewal": ["renew", "automatic", "evergreen"],
        "Limitation of Liability": ["limit", "liability", "cap"],
        "Liquidated Damages": ["liquidated", "damages", "pre-agreed"],
        "Force Majeure Exclusion": ["force majeure", "act of god"],
        "Governing Law Restriction": ["governed by", "laws of"],
        "Non-Disparagement Clause": ["disparage", "defame", "negative"],
        "Exclusivity Obligation": ["exclusive", "sole", "only provider"],
        "Data Sharing with Third Parties": ["third party", "share data", "disclose"],
        "Penalty for Early Termination": ["early termination", "exit fee", "cancellation"],
        "Warranty Disclaimer": ["warranty", "as is", "no guarantee"],
        "Assignment of Rights": ["assign", "transfer rights", "successor"],
    }
    hints     = keywords.get(label, [])
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    for sentence in sentences:
        for hint in hints:
            if hint.lower() in sentence.lower():
                return sentence[:300] + ("..." if len(sentence) > 300 else "")
    return text[:250] + "..."


def analyze_chunk(text_chunk: str, summarizer, risk_detector) -> tuple[str, list[dict]]:
    # Summarize
    summary = ""
    try:
        input_len = len(text_chunk.split())
        max_len   = min(150, max(30, input_len // 2))
        result    = summarizer(text_chunk, max_length=max_len, min_length=20, do_sample=False)
        r         = result[0]
        summary   = r.get("summary_text") or r.get("generated_text") or ""
    except Exception as e:
        print(f"Summarization skipped: {e}")

    # Detect risks
    detected = []
    try:
        if risk_detector["type"] == "finetuned":
            predictions = _predict_finetuned(text_chunk, risk_detector)
        else:
            predictions = _predict_zeroshot(text_chunk, risk_detector)

        for label, score in predictions:
            detected.append({
                "type":         label,
                "score":        round(score, 2),
                "text_snippet": _extract_best_snippet(text_chunk, label),
                "full_chunk":   text_chunk,
                "model":        risk_detector["type"],
            })
    except Exception as e:
        print(f"Risk detection skipped: {e}")

    return summary, detected


def deduplicate_risks(risks: list[dict]) -> list[dict]:
    seen = {}
    for risk in sorted(risks, key=lambda r: r["score"], reverse=True):
        composite = f"{risk['type']}|{risk['text_snippet'][:100]}"
        if composite not in seen:
            seen[composite] = risk
    return list(seen.values())


def analyze_document(chunks: list[str], summarizer, risk_detector) -> tuple[str, list[dict]]:
    summaries, all_risks = [], []
    print(f"Analyzing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        summary, risks = analyze_chunk(chunk, summarizer, risk_detector)
        if summary:
            summaries.append(summary)
        all_risks.extend(risks)

    combined = " ".join(summaries)
    if len(combined.split()) > 300:
        try:
            recap = summarizer(combined[:3000], max_length=200, min_length=60, do_sample=False)
            r = recap[0]
            final_summary = r.get("summary_text") or r.get("generated_text") or combined[:1000]
        except Exception:
            final_summary = combined[:1000] + "..."
    else:
        final_summary = combined

    deduped = deduplicate_risks(all_risks)
    deduped.sort(key=lambda r: r["score"], reverse=True)
    return final_summary, deduped
