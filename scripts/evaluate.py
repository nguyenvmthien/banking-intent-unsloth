"""
Evaluate IntentClassification on the Banking77 test set.
Supports zero_shot, finetuned, and all modes.
Uses batched inference for speed. LangSmith traces each prediction when
LANGSMITH_API_KEY is set.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from inference import IntentClassification


def run_evaluation(classifier: IntentClassification, df: pd.DataFrame,
                   label: str, batch_size: int = 8) -> float:
    """Run batched inference on all rows in df and print accuracy + classification report."""
    texts = df["text"].tolist()
    y_true = df["intent_name"].tolist()
    y_pred = []
    previews = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"[{label}]"):
        batch_texts = texts[start:start + batch_size]
        try:
            results = classifier.predict_batch(batch_texts)
            for i, result in enumerate(results):
                y_pred.append(result["label"])
                if start + i < 5:
                    previews.append(result)
        except Exception as e:
            for i, text in enumerate(batch_texts):
                y_pred.append("ERROR")
                if start + i < 5:
                    previews.append({"input": text, "raw_output": str(e), "label": "ERROR"})

    print("\nSample predictions:")
    for p in previews:
        print(f"  input      : {p['input'][:90]!r}")
        print(f"  raw_output : {p['raw_output']!r}")
        print(f"  label      : {p['label']!r}")
        print()

    acc = accuracy_score(y_true, y_pred)
    print("=" * 55)
    print(f"  {label}")
    print("=" * 55)
    print(f"  Samples    : {len(df)}")
    print(f"  Batch size : {batch_size}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 55)
    print(classification_report(y_true, y_pred, digits=4))
    return acc


def main():
    """Parse CLI args, load test data, run evaluation for one or all modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "finetuned", "all"],
                        default="finetuned")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from inference.yaml")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    config_path = os.path.join(repo_root, "configs", "inference.yaml")
    test_data_path = os.path.join(repo_root, "sample_data", "test.csv")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    batch_size = args.batch_size or config.get("batch_size", 8)

    df = pd.read_csv(test_data_path)
    modes = ["zero_shot", "finetuned"] if args.mode == "all" else [args.mode]

    mode_labels = {
        "zero_shot": "Zero-Shot (base model)",
        "finetuned": "Fine-Tuned (LoRA)",
    }

    results = {}
    for mode in modes:
        clf = IntentClassification(config_path, mode=mode)
        results[mode] = run_evaluation(clf, df, mode_labels[mode], batch_size=batch_size)

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        for mode, acc in results.items():
            print(f"  {mode_labels[mode]:<40} {acc:.4f}  ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
