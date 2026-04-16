import os
import argparse
import re
from difflib import get_close_matches
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from inference import IntentClassification


def normalize_intent_label(text):
    cleaned_text = str(text).strip().lower()
    cleaned_text = re.sub(r"[^a-z0-9]+", "_", cleaned_text)
    cleaned_text = re.sub(r"_+", "_", cleaned_text).strip("_")
    return cleaned_text


def map_to_known_label(prediction, known_labels):
    normalized_prediction = normalize_intent_label(prediction)
    normalized_labels = {normalize_intent_label(label): label for label in known_labels}

    if normalized_prediction in normalized_labels:
        return normalized_labels[normalized_prediction]

    for normalized_label, original_label in normalized_labels.items():
        if normalized_label in normalized_prediction or normalized_prediction in normalized_label:
            return original_label

    close_matches = get_close_matches(normalized_prediction, list(normalized_labels.keys()), n=1, cutoff=0.65)
    if close_matches:
        return normalized_labels[close_matches[0]]

    return normalized_prediction

def main():
    parser = argparse.ArgumentParser(description="Evaluate Intent Classification Model")
    parser.add_argument("--base", action="store_true", help="Evaluate the BASE model (zero-shot) instead of the fine-tuned one")
    args = parser.parse_args()

    # 1. Setup paths
    config_path = "../configs/inference.yaml"
    test_data_path = "../sample_data/test.csv"
    
    # Check if run from root or scripts folder
    if not os.path.exists(config_path):
        config_path = "configs/inference.yaml"
        test_data_path = "sample_data/test.csv"

    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        return

    # 2. Initialize Classifier
    mode_str = "BASE (Zero-shot)" if args.base else "FINE-TUNED"
    print(f"Initializing IntentClassification in {mode_str} mode...")
    try:
        classifier = IntentClassification(config_path, base_only=args.base)
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return

    # 3. Load Test Data
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    known_labels = sorted(df['intent_name'].dropna().unique().tolist())
    
    # 4. Run Inference
    print(f"Running evaluation on {len(df)} samples...")
    y_true = df['intent_name'].tolist()
    y_pred = []
    preview_rows = []
    
    for index, message in enumerate(tqdm(df['text'], desc=f"Evaluating {mode_str}")):
        try:
            prediction = classifier(message)
            mapped_prediction = map_to_known_label(prediction, known_labels)
            y_pred.append(mapped_prediction)
            if index < 5:
                preview_rows.append((message, prediction, mapped_prediction))
        except Exception as e:
            y_pred.append("ERROR")
            if index < 5:
                preview_rows.append((message, f"ERROR: {e}", "ERROR"))

    if preview_rows:
        print("\nSample predictions:")
        for message, raw_prediction, final_prediction in preview_rows:
            print(f"- text={message[:90]!r}")
            print(f"  raw={raw_prediction!r}")
            print(f"  mapped={final_prediction!r}")

    # 5. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"      RESULTS: {mode_str}      ")
    print("="*40)
    print(f"Total Samples: {len(df)}")
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*40)
    
    report = classification_report(y_true, y_pred, digits=4)
    print("\nDetailed Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
