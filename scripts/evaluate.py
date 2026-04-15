import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from inference import IntentClassification

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
    
    # 4. Run Inference
    print(f"Running evaluation on {len(df)} samples...")
    y_true = df['intent_name'].tolist()
    y_pred = []
    
    for message in tqdm(df['text'], desc=f"Evaluating {mode_str}"):
        try:
            prediction = classifier(message)
            y_pred.append(prediction)
        except Exception as e:
            y_pred.append("ERROR")

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
