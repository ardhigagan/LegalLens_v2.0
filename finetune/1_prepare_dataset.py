"""
1_prepare_dataset.py — LegalLens Phase 3 (Offline Version)
Reads CUAD_v1.json from local disk — no internet required.

Setup:
  1. Download CUAD_v1.json manually
  2. Place it at:  finetune/data/CUAD_v1.json
  3. Run:  python 1_prepare_dataset.py

Output: data/train.csv, data/val.csv, data/label_map.json
"""

import os
import json
import sys
import pandas as pd

os.makedirs("data", exist_ok=True)

CUAD_JSON_PATH = os.path.join("data", "CUAD_v1.json")

# ── Our 20 risk labels ────────────────────────────────────────────────
RISK_LABELS = [
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

LABEL2ID = {label: i for i, label in enumerate(RISK_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(RISK_LABELS)}

# ── CUAD question keyword → our risk label ────────────────────────────
QUESTION_TO_RISK = {
    "liquidated damages":           "Liquidated Damages",
    "anti-assignment":              "Assignment of Rights",
    "non-compete":                  "Non-Compete Restriction",
    "non compete":                  "Non-Compete Restriction",
    "termination for convenience":  "Termination Without Cause",
    "ip ownership":                 "Intellectual Property Transfer",
    "intellectual property":        "Intellectual Property Transfer",
    "license grant":                "Intellectual Property Transfer",
    "joint ip":                     "Intellectual Property Transfer",
    "arbitration":                  "Mandatory Arbitration",
    "indemnification":              "Indemnification Obligation",
    "insurance":                    "Indemnification Obligation",
    "governing law":                "Governing Law Restriction",
    "auto-renewal":                 "Automatic Renewal",
    "automatic renewal":            "Automatic Renewal",
    "cap on liability":             "Limitation of Liability",
    "limitation of liability":      "Limitation of Liability",
    "uncapped liability":           "Limitation of Liability",
    "warranty":                     "Warranty Disclaimer",
    "exclusivity":                  "Exclusivity Obligation",
    "most favored nation":          "Exclusivity Obligation",
    "non-disparagement":            "Non-Disparagement Clause",
    "price restrictions":           "Financial Penalty",
    "minimum commitment":           "Financial Penalty",
    "revenue/profit sharing":       "Financial Penalty",
    "penalty":                      "Financial Penalty",
    "audit rights":                 "Privacy Violation",
    "privacy":                      "Privacy Violation",
    "third party beneficiary":      "Data Sharing with Third Parties",
    "data":                         "Data Sharing with Third Parties",
    "change of control":            "Assignment of Rights",
    "assignment":                   "Assignment of Rights",
    "notice period to terminate":   "Penalty for Early Termination",
    "post-termination":             "Penalty for Early Termination",
    "termination":                  "Termination Without Cause",
    "jurisdiction":                 "Jurisdiction Waiver",
    "venue":                        "Jurisdiction Waiver",
    "amendment":                    "Unilateral Amendment",
    "force majeure":                "Force Majeure Exclusion",
}


def question_to_label(question: str):
    q = question.lower()
    for keyword, label in QUESTION_TO_RISK.items():
        if keyword in q:
            return label
    return None


def parse_cuad_json(path: str) -> list:
    """
    CUAD_v1.json is in SQuAD format:
    {
      "data": [
        {
          "title": "ContractName",
          "paragraphs": [
            {
              "context": "...contract text...",
              "qas": [
                {
                  "question": "Does this contain a Non-Compete clause?",
                  "answers": [{"text": "...", "answer_start": 0}],
                  "id": "..."
                }
              ]
            }
          ]
        }
      ]
    }
    """
    print(f"Reading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    data = cuad.get("data", [])
    print(f"Found {len(data)} contracts in CUAD.")

    # Group by context paragraph → multi-hot label vector
    context_labels = {}
    total_qas = 0

    for contract in data:
        for paragraph in contract.get("paragraphs", []):
            context = paragraph.get("context", "").strip()
            if not context or len(context.split()) < 20:
                continue

            if context not in context_labels:
                context_labels[context] = [0] * len(RISK_LABELS)

            for qa in paragraph.get("qas", []):
                total_qas += 1
                question = qa.get("question", "")
                label = question_to_label(question)
                if label is None:
                    continue

                # Positive example only if an answer exists
                answers = qa.get("answers", [])
                if answers and len(answers[0].get("text", "").strip()) > 5:
                    context_labels[context][LABEL2ID[label]] = 1

    print(f"Processed {total_qas} QA pairs across {len(context_labels)} unique paragraphs.")

    rows = []
    for context, labels in context_labels.items():
        rows.append({
            "text":     context[:512],
            "labels":   labels,
            "has_risk": int(any(labels)),
        })

    return rows


def main():
    print("=" * 55)
    print("LegalLens Phase 3 - Dataset Preparation (Offline)")
    print("=" * 55)
    print()

    # Check file exists
    if not os.path.exists(CUAD_JSON_PATH):
        print(f"ERROR: File not found at '{CUAD_JSON_PATH}'")
        print()
        print("Please download CUAD_v1.json manually:")
        print("  Option 1: https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/cuad_v1.json")
        print("  Option 2: https://www.atticusprojectai.org/s/CUAD_v1.json")
        print()
        print(f"Then place it at: {os.path.abspath(CUAD_JSON_PATH)}")
        sys.exit(1)

    rows = parse_cuad_json(CUAD_JSON_PATH)

    if not rows:
        print("ERROR: No rows generated from the JSON file.")
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(rows)
    label_cols = pd.DataFrame(df["labels"].tolist(), columns=RISK_LABELS)
    df = pd.concat([df["text"], label_cols, df["has_risk"]], axis=1)

    # Shuffle and split 85/15
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(len(df) * 0.85)
    train_df = df[:split]
    val_df   = df[split:]

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv",     index=False)

    with open("data/label_map.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    print()
    print("Done!")
    print(f"  Train : {len(train_df)} examples -> data/train.csv")
    print(f"  Val   : {len(val_df)} examples   -> data/val.csv")
    print(f"  Labels: {len(RISK_LABELS)}        -> data/label_map.json")
    print()
    print("  Risk distribution in training set:")
    for label in RISK_LABELS:
        count = int(train_df[label].sum()) if label in train_df.columns else 0
        bar   = "#" * min(count, 40)
        print(f"    {label:<40} {count:>4}  {bar}")
    print()
    print("  Next step: python 2_finetune_lora.py")


if __name__ == "__main__":
    main()
