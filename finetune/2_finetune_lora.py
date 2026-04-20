"""
2_finetune_lora.py — LegalLens Phase 3
Fine-tunes nlpaueb/legal-bert-base-uncased on CUAD using LoRA.

Why LoRA on CPU?
- Full fine-tune of BERT (110M params) on CPU = days
- LoRA trains only ~1% of weights (adapter layers) = hours
- Same quality improvement, fraction of the compute

Run: python 2_finetune_lora.py
Output: models/legallens-legal-bert-lora/  (your fine-tuned model)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL    = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR    = "models/legallens-legal-bert-lora"
DATA_DIR      = "data"
MAX_LENGTH    = 512
BATCH_SIZE    = 4       # Small batch for CPU
EPOCHS        = 3       # 3 epochs is enough with LoRA
LEARNING_RATE = 2e-4    # Higher LR works well for LoRA
THRESHOLD     = 0.45    # Confidence threshold for multi-label prediction

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load label map ────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "label_map.json")) as f:
    label_map = json.load(f)

LABEL2ID = label_map["label2id"]
ID2LABEL = {int(k): v for k, v in label_map["id2label"].items()}
NUM_LABELS = len(LABEL2ID)
RISK_LABELS = list(LABEL2ID.keys())

print(f"Labels: {NUM_LABELS}")
print(f"Base model: {BASE_MODEL}")
print(f"Device: CPU (LoRA optimized)")


# ── Dataset Class ─────────────────────────────────────────────────────
class ContractDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(csv_path)
        self.texts  = df["text"].tolist()
        self.labels = df[RISK_LABELS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.texts)} examples from {csv_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Metrics ───────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > THRESHOLD).astype(int)

    # Micro F1 (overall)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    # Macro F1 (per-label average)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    # ROC-AUC (only for labels that appear in eval set)
    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = 0.0

    return {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc":  round(auc, 4),
    }


# ── Custom Trainer for multi-label BCE loss ───────────────────────────
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Binary Cross Entropy for multi-label classification
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Main ──────────────────────────────────────────────────────────────
def main():
    # 1. Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 2. Load datasets
    print("[2/5] Loading datasets...")
    train_dataset = ContractDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_dataset   = ContractDataset(os.path.join(DATA_DIR, "val.csv"),   tokenizer)

    # 3. Load base model
    print(f"[3/5] Loading base model ({BASE_MODEL})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 4. Apply LoRA
    print("[4/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                    # Rank — higher = more params, more accurate
        lora_alpha=32,          # Scaling factor
        target_modules=["query", "value"],  # Apply LoRA to attention layers
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.5% of parameters — that's the LoRA magic

    # 5. Training arguments (CPU-optimized)
    print("[5/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_total_limit=2,
        # CPU optimizations
        no_cuda=True,
        dataloader_num_workers=0,   # Avoid multiprocessing issues on Windows
        fp16=False,                 # No FP16 on CPU
        report_to="none",           # Disable wandb
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Save the LoRA adapter + tokenizer
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training summary
    summary = {
        "base_model": BASE_MODEL,
        "output_dir": OUTPUT_DIR,
        "num_labels": NUM_LABELS,
        "labels": RISK_LABELS,
        "threshold": THRESHOLD,
        "lora_config": {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["query", "value"],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {OUTPUT_DIR}/")
    print(f"   Next step: run 3_evaluate.py to see your results")


if __name__ == "__main__":
    main()
