"""
3_evaluate.py — LegalLens Phase 3
Evaluates the fine-tuned LoRA model against the zero-shot baseline.
Generates a comparison report so you can see the accuracy improvement.

Run: python 3_evaluate.py
Output: evaluation_report.json, evaluation_report.txt
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR  = "models/legallens-legal-bert-lora"
DATA_DIR   = "data"
THRESHOLD  = 0.45

# Load label map
with open(os.path.join(DATA_DIR, "label_map.json")) as f:
    label_map = json.load(f)

LABEL2ID   = label_map["label2id"]
ID2LABEL   = {int(k): v for k, v in label_map["id2label"].items()}
RISK_LABELS = list(LABEL2ID.keys())
NUM_LABELS  = len(RISK_LABELS)


# ── Load fine-tuned model ─────────────────────────────────────────────
def load_finetuned_model():
    print("Loading fine-tuned LoRA model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    return tokenizer, model


# ── Predict with fine-tuned model ────────────────────────────────────
def predict_finetuned(texts: list[str], tokenizer, model, batch_size: int = 8) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoding = tokenizer(
            batch,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoding).logits
        probs = torch.sigmoid(logits).numpy()
        all_probs.append(probs)
        if (i // batch_size) % 10 == 0:
            print(f"  Predicted {min(i+batch_size, len(texts))}/{len(texts)}...")
    return np.vstack(all_probs)


# ── Predict with zero-shot baseline ──────────────────────────────────
def predict_zeroshot(texts: list[str], sample_size: int = 100) -> np.ndarray:
    """
    Run zero-shot on a sample (full eval would take hours on CPU).
    """
    print(f"Running zero-shot baseline on {sample_size} samples...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
        framework="pt",
    )
    all_probs = np.zeros((sample_size, NUM_LABELS))
    for i, text in enumerate(texts[:sample_size]):
        result = classifier(text[:512], RISK_LABELS, multi_label=True)
        for label, score in zip(result["labels"], result["scores"]):
            if label in LABEL2ID:
                all_probs[i, LABEL2ID[label]] = score
        if i % 10 == 0:
            print(f"  Zero-shot: {i}/{sample_size}...")
    return all_probs


# ── Metrics helper ────────────────────────────────────────────────────
def compute_full_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, name: str) -> dict:
    y_pred = (y_pred_probs > THRESHOLD).astype(int)

    f1_mi = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    f1_ma = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    prec  = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="micro",  zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_probs, average="macro")
    except ValueError:
        auc = 0.0

    per_label = {}
    for i, label in enumerate(RISK_LABELS):
        if y_true[:, i].sum() > 0:
            per_label[label] = {
                "f1":        round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 3),
                "precision": round(precision_score(y_true[:, i], y_pred[:, i], zero_division=0), 3),
                "recall":    round(recall_score(y_true[:, i], y_pred[:, i], zero_division=0), 3),
                "support":   int(y_true[:, i].sum()),
            }

    return {
        "model":     name,
        "f1_micro":  round(f1_mi, 4),
        "f1_macro":  round(f1_ma, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "roc_auc":   round(auc, 4),
        "per_label": per_label,
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LegalLens Phase 3 — Model Evaluation")
    print("=" * 60)

    # Load validation set
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    texts  = val_df["text"].tolist()
    y_true = val_df[RISK_LABELS].values.astype(int)
    print(f"Validation set: {len(texts)} examples\n")

    # 1. Evaluate fine-tuned model
    tokenizer, model = load_finetuned_model()
    ft_probs = predict_finetuned(texts, tokenizer, model)
    ft_metrics = compute_full_metrics(y_true, ft_probs, "Fine-tuned legal-bert (LoRA)")

    # 2. Evaluate zero-shot baseline (sample for speed)
    sample_size = min(100, len(texts))
    zs_probs = predict_zeroshot(texts, sample_size)
    zs_metrics = compute_full_metrics(
        y_true[:sample_size], zs_probs,
        "Zero-shot bart-large-mnli (baseline)"
    )

    # 3. Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Zero-Shot':>12} {'Fine-Tuned':>12} {'Improvement':>14}")
    print("-" * 60)

    for metric in ["f1_micro", "f1_macro", "precision", "recall", "roc_auc"]:
        zs_val = zs_metrics[metric]
        ft_val = ft_metrics[metric]
        delta  = ft_val - zs_val
        sign   = "+" if delta >= 0 else ""
        print(f"{metric:<20} {zs_val:>12.4f} {ft_val:>12.4f} {sign+str(round(delta,4)):>14}")

    print("\nPer-label F1 (fine-tuned model):")
    print("-" * 60)
    for label, m in sorted(ft_metrics["per_label"].items(), key=lambda x: -x[1]["f1"]):
        bar = "█" * int(m["f1"] * 20)
        print(f"  {label:<35} F1={m['f1']:.3f}  {bar}")

    # 4. Save report
    report = {
        "zero_shot":   zs_metrics,
        "fine_tuned":  ft_metrics,
        "improvement": {
            metric: round(ft_metrics[metric] - zs_metrics[metric], 4)
            for metric in ["f1_micro", "f1_macro", "precision", "recall", "roc_auc"]
        }
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Full report saved to evaluation_report.json")
    print(f"   Next step: run 4_integrate.py to plug this into LegalLens v2")


if __name__ == "__main__":
    main()
