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
    def __init__(self, model_path):
        # Determine path to inference.yaml
        # For simplicity, if model_path points directly to the config, we load it.
        # Otherwise, load default configs/inference.yaml
        if model_path.endswith('.yaml'):
            with open(model_path, 'r') as f:
                self.config = yaml.safe_load(f)
            actual_model_path = self.config['model_path']
        else:
            # Fallback path if string is just directory
            actual_model_path = model_path
            # Provide sensible defaults
            self.config = {
                'max_seq_length': 256,
                'load_in_4bit': True,
                'max_new_tokens': 32
            }
            
        print(f"Loading model checkpoint from {actual_model_path}...")
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
