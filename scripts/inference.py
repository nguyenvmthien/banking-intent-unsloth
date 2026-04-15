import os
import yaml
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer request.

### Input:
{}

### Response:
"""

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
        # Enable 2x faster native inference
        FastLanguageModel.for_inference(self.model)
        
    def __call__(self, message):
        formatted_prompt = PROMPT_TEMPLATE.format(message)
        inputs = self.tokenizer(
            [formatted_prompt], return_tensors = "pt"
        ).to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens = self.config.get('max_new_tokens', 32),
            use_cache = True,
            pad_token_id = self.tokenizer.eos_token_id
        )
        
        # Decode and crop output
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # Since response contains the prompt, we extract just the completion part
        predicted_label = response.split("### Response:\n")[-1].strip()
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
