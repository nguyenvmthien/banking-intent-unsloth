"""
Evaluate IntentClassification on the Banking77 test set.
Supports zero_shot, few_shot, finetuned, and all modes.
LangSmith traces each prediction when LANGSMITH_API_KEY is set.
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from inference import IntentClassification


def run_evaluation(classifier: IntentClassification, df: pd.DataFrame, label: str) -> float:
    y_true = df["intent_name"].tolist()
    y_pred = []
    previews = []

    for i, text in enumerate(tqdm(df["text"], desc=f"[{label}]")):
        try:
            result = classifier.predict(text)
            y_pred.append(result["label"])
            if i < 5:
                previews.append(result)
        except Exception as e:
            y_pred.append("ERROR")
            if i < 5:
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
    print(f"  Samples  : {len(df)}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 55)
    print(classification_report(y_true, y_pred, digits=4))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "finetuned", "all"],
                        default="finetuned")
    parser.add_argument("--few_shot_k", type=int, default=5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    config_path = os.path.join(repo_root, "configs", "inference.yaml")
    test_data_path = os.path.join(repo_root, "sample_data", "test.csv")

    df = pd.read_csv(test_data_path)
    modes = ["zero_shot", "few_shot", "finetuned"] if args.mode == "all" else [args.mode]

    mode_labels = {
        "zero_shot": "Zero-Shot (base model)",
        "few_shot": f"Few-Shot {args.few_shot_k}-shot (base model)",
        "finetuned": "Fine-Tuned (LoRA)",
    }

    results = {}
    for mode in modes:
        clf = IntentClassification(config_path, mode=mode, few_shot_k=args.few_shot_k)
        results[mode] = run_evaluation(clf, df, mode_labels[mode])

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        for mode, acc in results.items():
            print(f"  {mode_labels[mode]:<40} {acc:.4f}  ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
