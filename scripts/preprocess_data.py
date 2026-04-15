import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

def sample_dataset(df, test_size=0.1, random_state=42):
    """
    Stratified sampling to ensure all intent classes are still represented
    while reducing the dataset size to save compute resources.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for _, sample_index in split.split(df, df['label']):
        sampled_df = df.iloc[sample_index].reset_index(drop=True)
        return sampled_df

def main():
    print("Loading BANKING77 dataset from Hugging Face...")
    dataset = load_dataset("PolyAI/banking77")
    
    # Convert to pandas
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Get class names to map label ids to text for clear prompt engineering
    label_names = dataset['train'].features['label'].names
    train_df['intent_name'] = train_df['label'].apply(lambda x: label_names[x])
    test_df['intent_name'] = test_df['label'].apply(lambda x: label_names[x])
    
    print(f"Original Train size: {len(train_df)}")
    print(f"Original Test size: {len(test_df)}")
    
    # Using 100% of the data as requested
    sampled_train = train_df
    sampled_test = test_df
    
    print(f"Sampled Train size: {len(sampled_train)}")
    print(f"Sampled Test size: {len(sampled_test)}")
    
    # Ensure directory exists
    os.makedirs('../sample_data', exist_ok=True)
    
    # Keep only necessary columns: text, label, intent_name
    sampled_train = sampled_train[['text', 'label', 'intent_name']]
    sampled_test = sampled_test[['text', 'label', 'intent_name']]
    
    # Save to CSV
    train_csv_path = '../sample_data/train.csv'
    test_csv_path = '../sample_data/test.csv'
    
    sampled_train.to_csv(train_csv_path, index=False)
    sampled_test.to_csv(test_csv_path, index=False)
    
    print(f"Saved sampled train data to {train_csv_path}")
    print(f"Saved sampled test data to {test_csv_path}")

if __name__ == "__main__":
    main()
