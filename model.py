# =============================================
# OPTIMIZED DEPRESSION DETECTION PIPELINE
# =============================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import inspect
import os
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import joblib

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv("dataset.csv")

# Encode labels
df["depressed"] = df["depressed"].astype(str).str.upper().map({"YES": 1, "NO": 0})
df = df.dropna(subset=["depressed"]).copy()
df["depressed"] = df["depressed"].astype(int)

# -----------------------------
# 3. BASIC EDA
# -----------------------------
print("Dataset Shape:", df.shape)
print(df["depressed"].value_counts())

# -----------------------------
# 4. PREPROCESSING (IMPROVED)
# -----------------------------


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


df["Text"] = df["Text"].fillna("").astype(str)
df["clean_text"] = df["Text"].apply(preprocess)

# -----------------------------
# 5. TRAIN TEST SPLIT (STRATIFIED)
# -----------------------------
X_raw = df["Text"]
X = df["clean_text"]
y = df["depressed"]

X_train_raw, X_test_raw, X_train, X_test, y_train, y_test = train_test_split(
    X_raw, X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 6. TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 7. MODEL TRAINING
# -----------------------------

# Logistic Regression (Baseline)
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_tfidf, y_train)

# Linear SVM
svm_model = SVC(kernel="linear", class_weight="balanced", probability=True)
svm_model.fit(X_train_tfidf, y_train)

# -----------------------------
# 8. EVALUATION FUNCTION
# -----------------------------


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


# Evaluate models
evaluate(log_model, X_test_tfidf, y_test, "Logistic Regression")
evaluate(svm_model, X_test_tfidf, y_test, "SVM")

# -----------------------------
# 9. CROSS VALIDATION
# -----------------------------
log_cv_scores = cross_val_score(log_model, X_train_tfidf, y_train, cv=5, scoring="f1")
print("\nCross Validation (Logistic Regression):")
print(log_cv_scores)
print(f"Mean F1: {log_cv_scores.mean():.4f} | Std: {log_cv_scores.std():.4f}")

# -----------------------------
# 10. HYPERPARAMETER TUNING
# -----------------------------
param_grid = {"C": [0.1, 1, 10]}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)

grid.fit(X_train_tfidf, y_train)

best_log_model = grid.best_estimator_
print("Best Logistic Regression Params:", grid.best_params_)

# -----------------------------
# 11. FINAL EVALUATION
# -----------------------------
evaluate(best_log_model, X_test_tfidf, y_test, "Tuned Logistic Regression")

classical_model_candidates = {
    "Tuned Logistic Regression": best_log_model,
    "SVM": svm_model,
}

classical_model_metrics = {}

for model_name, model in classical_model_candidates.items():
    preds = model.predict(X_test_tfidf)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="binary",
        zero_division=0,
    )
    classical_model_metrics[model_name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

classical_metrics_df = pd.DataFrame(classical_model_metrics).T.sort_values(
    by=["f1", "accuracy"],
    ascending=False,
)
print("\nClassical Model Comparison:")
print(classical_metrics_df.round(4).to_string())

best_classical_model_name = classical_metrics_df.index[0]
best_model = classical_model_candidates[best_classical_model_name]
print("Selected classical model:", best_classical_model_name)

# -----------------------------
# 12. SAVE MODEL
# -----------------------------
joblib.dump(best_model, "depression_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# -----------------------------
# 13. SIMPLE PREDICTION FUNCTION
# -----------------------------


def predict(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    pred = best_model.predict(vec)[0]
    prob = best_model.predict_proba(vec)[0]
    return pred, prob


# Example
print(predict("I feel very sad and lonely"))


# -----------------------------
# 14. OPTIMIZED TRANSFORMER IMPLEMENTATION (DISTILBERT)
# -----------------------------

# Install if needed:
# pip install torch transformers accelerate

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:
    raise ImportError(
        "Install 'torch', 'transformers', and 'accelerate' to run the transformer section."
    ) from exc


TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_OUTPUT_DIR = "distilbert_depression_model"
TRANSFORMER_RESULTS_DIR = "./transformer_results"
TRANSFORMER_LOGS_DIR = "./logs"
MAX_LENGTH = 128
LABEL_MAP = {0: "NO", 1: "YES"}

set_seed(42)
os.makedirs(TRANSFORMER_RESULTS_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_LOGS_DIR, exist_ok=True)

print("\nTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    print("GPU device:", torch.cuda.get_device_name(0))
    print("cuDNN benchmark:", getattr(torch.backends.cudnn, "benchmark", None))
else:
    print("Transformer training will run on CPU.")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.labels = list(labels)
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


# Prepare raw text (IMPORTANT: use original text, not heavily cleaned)
X_train_transformer = X_train_raw.tolist()
X_test_transformer = X_test_raw.tolist()

y_train_transformer = y_train.tolist()
y_test_transformer = y_test.tolist()

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, use_fast=True)
transformer_model = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_MODEL_NAME,
    num_labels=2,
    id2label=LABEL_MAP,
    label2id={label: idx for idx, label in LABEL_MAP.items()},
)
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)

# Create datasets
train_dataset = TextDataset(X_train_transformer, y_train_transformer, tokenizer)
test_dataset = TextDataset(X_test_transformer, y_test_transformer, tokenizer)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
use_cuda = torch.cuda.is_available()
dataloader_workers = 0 if os.name == "nt" else (2 if use_cuda else 0)
train_batch_size = 32 if use_cuda else 8
eval_batch_size = 64 if use_cuda else 16
training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
training_args_kwargs = {
    "output_dir": TRANSFORMER_RESULTS_DIR,
    "overwrite_output_dir": True,
    "learning_rate": 2e-5,
    "num_train_epochs": 2 if use_cuda else 1,
    "per_device_train_batch_size": train_batch_size,
    "per_device_eval_batch_size": eval_batch_size,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "logging_dir": TRANSFORMER_LOGS_DIR,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "report_to": "none",
    "dataloader_num_workers": dataloader_workers,
    "dataloader_pin_memory": use_cuda,
    "dataloader_persistent_workers": dataloader_workers > 0,
    "fp16": use_cuda,
    "gradient_accumulation_steps": 1,
    "group_by_length": True,
    "seed": 42,
    "data_seed": 42,
}

if "eval_strategy" in training_args_signature:
    training_args_kwargs["eval_strategy"] = "epoch"
else:
    training_args_kwargs["evaluation_strategy"] = "epoch"

training_args_kwargs = {
    key: value for key, value in training_args_kwargs.items() if key in training_args_signature
}

training_args = TrainingArguments(**training_args_kwargs)
print(
    "\nTransformer Training Configuration:\n"
    + pd.DataFrame(
        [
            {
                "device": "GPU" if use_cuda else "CPU",
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "dataloader_workers": dataloader_workers,
                "fp16": use_cuda,
            }
        ]
    ).to_string(index=False)
)

# -----------------------------
# METRICS FUNCTION
# -----------------------------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# -----------------------------
# TRAINER
# -----------------------------
trainer_kwargs = {
    "model": transformer_model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": test_dataset,
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
}

if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print(f"\nStarting transformer training on {'GPU' if use_cuda else 'CPU'}...")
train_output = trainer.train()
train_metrics = train_output.metrics

# -----------------------------
# EVALUATE MODEL
# -----------------------------
results = trainer.evaluate()


def format_metric_table(metric_dict, prefix):
    rows = []
    for key, value in metric_dict.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        metric_name = key[len(prefix) :] if key.startswith(prefix) else key
        rows.append(
            {
                "Metric": metric_name.replace("_", " ").title(),
                "Value": round(float(value), 4),
            }
        )
    return pd.DataFrame(rows)


train_metrics_df = format_metric_table(train_metrics, "train_")
eval_metrics_df = format_metric_table(results, "eval_")
key_eval_metrics = eval_metrics_df[
    eval_metrics_df["Metric"].isin(["Accuracy", "F1", "Precision", "Recall", "Loss"])
].reset_index(drop=True)

print("\nTransformer Training Summary")
print(train_metrics_df.to_string(index=False))
print("\nTransformer Evaluation Summary")
print(eval_metrics_df.to_string(index=False))
print("\nKey Evaluation Metrics")
print(key_eval_metrics.to_string(index=False))

# -----------------------------
# SAVE MODEL
# -----------------------------
trainer.save_model(TRANSFORMER_OUTPUT_DIR)
tokenizer.save_pretrained(TRANSFORMER_OUTPUT_DIR)

# -----------------------------
# PREDICTION FUNCTION (TRANSFORMER)
# -----------------------------


def transformer_predict(text):
    transformer_model.eval()
    device = transformer_model.device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = transformer_model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return pred, probs.cpu().numpy()[0]


def bert_predict(text):
    return transformer_predict(text)


# Example
print(transformer_predict("I feel completely hopeless and alone"))

# =============================================
# END OF SCRIPT
# =============================================
