import os
import re
import yaml
import pandas as pd
import torch
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer request.

### Input:
{}

### Response:
"""


def normalize_intent_label(text):
    cleaned_text = text.strip().lower()
    cleaned_text = re.sub(r"[^a-z0-9]+", "_", cleaned_text)
    cleaned_text = re.sub(r"_+", "_", cleaned_text).strip("_")
    return cleaned_text

class IntentClassification:
    def __init__(self, config_path, base_only=False):
        """
        Required: config_path must point to a configuration YAML file.
        base_only: If True, loads the original model without PEFT adapters.
        """
        if not config_path.endswith('.yaml'):
            raise ValueError(f"Requirement Error: config_path must be a .yaml file. Got: {config_path}")
            
        if not os.path.exists(config_path):
            # Try to resolve relative to root if script is run from different folder
            alt_path = os.path.join("..", config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_dir = os.path.dirname(os.path.abspath(config_path))
            
        # Use model_name for base comparison, or model_path for loaded adapters
        actual_model_path = self.config.get('model_name') if base_only else self.config.get('model_path')
        
        if not actual_model_path:
            raise KeyError(f"Config file must contain {'model_name' if base_only else 'model_path'}")
            
        print(f"Loading {'BASE' if base_only else 'FINE-TUNED'} model from {actual_model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = actual_model_path,
            max_seq_length = self.config['max_seq_length'],
            dtype = None,
            load_in_4bit = self.config['load_in_4bit'],
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.intent_labels = self._load_intent_labels()
        # Enable 2x faster native inference
        FastLanguageModel.for_inference(self.model)

    def _load_intent_labels(self):
        label_path = self.config.get('label_path', '../sample_data/train.csv')
        resolved_label_path = label_path if os.path.isabs(label_path) else os.path.normpath(
            os.path.join(self.config_dir, label_path)
        )

        if not os.path.exists(resolved_label_path):
            fallback_path = os.path.normpath(os.path.join(self.config_dir, '../sample_data/test.csv'))
            resolved_label_path = fallback_path if os.path.exists(fallback_path) else resolved_label_path

        if not os.path.exists(resolved_label_path):
            raise FileNotFoundError(f"Could not find label source CSV at {resolved_label_path}")

        label_frame = pd.read_csv(resolved_label_path)
        if 'intent_name' not in label_frame.columns:
            raise KeyError(f"Label source CSV must contain an 'intent_name' column: {resolved_label_path}")

        labels = label_frame['intent_name'].dropna().astype(str).unique().tolist()
        labels = sorted(labels)
        if not labels:
            raise ValueError(f"No intent labels found in {resolved_label_path}")
        return labels

    def _score_candidate_labels(self, prompt_text):
        prompt_tokens = self.tokenizer(
            prompt_text,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation = True,
            max_length = self.config['max_seq_length'],
        )["input_ids"].shape[-1]

        candidate_texts = [f"{prompt_text}{label}" for label in self.intent_labels]
        tokenized = self.tokenizer(
            candidate_texts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            add_special_tokens = False,
            max_length = self.config['max_seq_length'],
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_tokens = tokenized["input_ids"][:, 1:]
        token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

        candidate_lengths = tokenized["attention_mask"].sum(dim=1)
        scores = []
        for index, candidate_length in enumerate(candidate_lengths.tolist()):
            label_start = max(prompt_tokens - 1, 0)
            label_end = max(candidate_length - 1, label_start)
            scores.append(token_log_probs[index, label_start:label_end].sum().item())

        best_index = max(range(len(scores)), key=scores.__getitem__)
        return self.intent_labels[best_index], scores[best_index]
        
    def __call__(self, message):
        formatted_prompt = PROMPT_TEMPLATE.format(message)
        predicted_label, _ = self._score_candidate_labels(formatted_prompt)
        return predicted_label

if __name__ == "__main__":
    # Short usage example showing how the inference class is called
    print("--- Inference Example ---")
    
    # Path to the config file (or directly to the model directory)
    config_file = "../configs/inference.yaml"
    
    try:
        classifier = IntentClassification(config_file)
        
        test_message = "I lost my credit card yesterday, how do I order a new one?"
        print(f"\n[Input Message]: {test_message}")
        
        predicted_intent = classifier(test_message)
        print(f"[Predicted Intent]: {predicted_intent}")
        print("-------------------------")
    except Exception as e:
        print(f"Failed to run inference: {e}")
        print("Have you finished training and saving the model to outputs/checkpoint yet?")
