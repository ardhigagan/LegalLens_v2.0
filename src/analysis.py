"""
analysis.py -- LegalLens AI v3
Risk detector supports 3 modes:
  1. Fine-tuned LoRA from HuggingFace Hub (set HF_MODEL_REPO secret)
  2. Fine-tuned LoRA from local disk (finetune/models/)
  3. Zero-shot bart-large-mnli (automatic fallback)
"""

from transformers import pipeline
import torch
import os

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")

CANDIDATE_LABELS = [
    "Financial Penalty",
    "Privacy Violation",
    "Non-Compete Restriction",
    "Termination Without Cause",
    "Intellectual Property Transfer",
    "Mandatory Arbitration",
    "Indemnification Obligation",
    "Unilateral Amendment",
    "Jurisdiction Waiver",
    "Automatic Renewal",
    "Limitation of Liability",
    "Liquidated Damages",
    "Force Majeure Exclusion",
    "Governing Law Restriction",
    "Non-Disparagement Clause",
    "Exclusivity Obligation",
    "Data Sharing with Third Parties",
    "Penalty for Early Termination",
    "Warranty Disclaimer",
    "Assignment of Rights",
]

RISK_THRESHOLD = 0.50


def load_summarizer():
    print("Loading Summarization Model (bart-large-cnn)...")
    try:
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
            framework="pt",
        )
    except KeyError:
        from transformers import BartForConditionalGeneration, BartTokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)


def load_risk_detector():
    """
    Load risk detector.
    Priority:
      1. Fine-tuned LoRA from HF Hub  (HF_MODEL_REPO env var)
      2. Fine-tuned LoRA from local disk
      3. Zero-shot fallback (always works)
    """
    # Option 1: HuggingFace Hub
    hf_repo = os.getenv("HF_MODEL_REPO", "").strip()
    if hf_repo:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
            print(f"Loading fine-tuned model from HF Hub: {hf_repo}")
            tokenizer = AutoTokenizer.from_pretrained(hf_repo)
            base = AutoModelForSequenceClassification.from_pretrained(
                "nlpaueb/legal-bert-base-uncased",
                num_labels=len(CANDIDATE_LABELS),
                ignore_mismatched_sizes=True,
            )
            model = PeftModel.from_pretrained(base, hf_repo)
            model.eval()
            print("Fine-tuned model loaded successfully.")
            return {"type": "finetuned", "model": model, "tokenizer": tokenizer}
        except Exception as e:
            print(f"HF Hub load failed ({e}). Trying local disk...")

    # Option 2: Local disk
    local_path = os.path.join(
        os.path.dirname(__file__), "..", "finetune", "models", "legallens-legal-bert-lora"
    )
    if os.path.exists(local_path):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
            print(f"Loading fine-tuned model from local: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            base = AutoModelForSequenceClassification.from_pretrained(
                "nlpaueb/legal-bert-base-uncased",
                num_labels=len(CANDIDATE_LABELS),
                ignore_mismatched_sizes=True,
            )
            model = PeftModel.from_pretrained(base, local_path)
            model.eval()
            print("Local fine-tuned model loaded.")
            return {"type": "finetuned", "model": model, "tokenizer": tokenizer}
        except Exception as e:
            print(f"Local load failed ({e}). Using zero-shot fallback.")

    # Option 3: Zero-shot fallback
    print("Using zero-shot classification (bart-large-mnli)...")
    return {
        "type": "zeroshot",
        "classifier": pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
            framework="pt",
        )
    }


def _predict_risks(text_chunk: str, risk_detector) -> list:
    """Runs risk prediction handling both finetuned and zeroshot detectors."""
    results = []

    dtype = risk_detector.get("type") if isinstance(risk_detector, dict) else "legacy"

    if dtype == "finetuned":
        tokenizer = risk_detector["tokenizer"]
        model     = risk_detector["model"]
        encoding  = tokenizer(
            text_chunk, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**encoding).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]
        for i, prob in enumerate(probs):
            if prob > 0.45 and i < len(CANDIDATE_LABELS):
                results.append((CANDIDATE_LABELS[i], float(prob)))

    elif dtype == "zeroshot":
        clf = risk_detector["classifier"]
        out = clf(text_chunk[:1024], CANDIDATE_LABELS, multi_label=True)
        for label, score in zip(out["labels"], out["scores"]):
            if score > RISK_THRESHOLD:
                results.append((label, score))

    else:
        # Legacy bare pipeline
        out = risk_detector(text_chunk, CANDIDATE_LABELS, multi_label=True)
        for label, score in zip(out["labels"], out["scores"]):
            if score > RISK_THRESHOLD:
                results.append((label, score))

    return results


def _extract_best_snippet(text: str, label: str) -> str:
    keywords = {
        "Financial Penalty":             ["penalty", "fine", "fee", "liquidated", "damages"],
        "Privacy Violation":             ["privacy", "personal data", "gdpr", "collect", "share"],
        "Non-Compete Restriction":       ["compete", "competitor", "non-compete", "solicit"],
        "Termination Without Cause":     ["terminate", "termination", "without cause", "at will"],
        "Intellectual Property Transfer":["intellectual property", "ip", "assign", "ownership"],
        "Mandatory Arbitration":         ["arbitration", "arbitrate", "dispute resolution"],
        "Indemnification Obligation":    ["indemnif", "hold harmless", "defend"],
        "Unilateral Amendment":          ["amend", "modify", "change", "sole discretion"],
        "Jurisdiction Waiver":           ["jurisdiction", "venue", "waive", "governing law"],
        "Automatic Renewal":             ["renew", "automatic", "evergreen", "roll over"],
        "Limitation of Liability":       ["limit", "liability", "cap", "maximum"],
        "Liquidated Damages":            ["liquidated", "damages", "pre-agreed", "estimate"],
        "Force Majeure Exclusion":       ["force majeure", "act of god", "beyond control"],
        "Governing Law Restriction":     ["governed by", "laws of", "jurisdiction of"],
        "Non-Disparagement Clause":      ["disparage", "defame", "negative statement"],
        "Exclusivity Obligation":        ["exclusive", "sole", "only provider"],
        "Data Sharing with Third Parties":["third party", "third-party", "share data", "disclose"],
        "Penalty for Early Termination": ["early termination", "exit fee", "cancellation fee"],
        "Warranty Disclaimer":           ["warranty", "as is", "no guarantee", "disclaim"],
        "Assignment of Rights":          ["assign", "transfer rights", "successor", "novation"],
    }
    hints     = keywords.get(label, [])
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    for sentence in sentences:
        for hint in hints:
            if hint.lower() in sentence.lower():
                return sentence[:300] + ("..." if len(sentence) > 300 else "")
    return text[:250] + "..."


def deduplicate_risks(risks: list) -> list:
    seen = {}
    for risk in sorted(risks, key=lambda r: r["score"], reverse=True):
        composite = f"{risk['type']}|{risk['text_snippet'][:100]}"
        if composite not in seen:
            seen[composite] = risk
    return list(seen.values())


def analyze_chunk(text_chunk: str, summarizer, risk_detector) -> tuple:
    summary = ""
    try:
        input_len = len(text_chunk.split())
        max_len   = min(150, max(30, input_len // 2))
        result    = summarizer(text_chunk, max_length=max_len, min_length=20, do_sample=False)
        r         = result[0]
        summary   = r.get("summary_text") or r.get("generated_text") or ""
    except Exception as e:
        print(f"Summarization skipped: {e}")

    detected_risks = []
    try:
        for label, score in _predict_risks(text_chunk, risk_detector):
            detected_risks.append({
                "type":         label,
                "score":        round(score, 2),
                "text_snippet": _extract_best_snippet(text_chunk, label),
                "full_chunk":   text_chunk,
            })
    except Exception as e:
        print(f"Risk detection skipped: {e}")

    return summary, detected_risks


def analyze_document(chunks: list, summarizer, risk_detector) -> tuple:
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
