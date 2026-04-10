import os
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer request.

### Input:
{}

### Response:
{}"""

def main():
    # 1. Load config
    config_path = '../configs/train.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading model and tokenizer with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model_name'],
        max_seq_length = config['max_seq_length'],
        dtype = None,
        load_in_4bit = config['load_in_4bit'],
    )
    
    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora_r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = config['lora_alpha'],
        lora_dropout = config['lora_dropout'], 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = config.get('seed', 3407),
        use_rslora = False,
        loftq_config = None,
    )
    
    # 3. Prepare Dataset
    print(f"Loading data from {config['data_path']}...")
    df = pd.read_csv(config['data_path'])
    dataset = Dataset.from_pandas(df)
    
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        inputs = examples["text"]
        outputs = examples["intent_name"]
        texts = []
        for input_text, output_text in zip(inputs, outputs):
            text = PROMPT_TEMPLATE.format(input_text, output_text) + EOS_TOKEN
            texts.append(text)
        return { "formatted_text" : texts }
        
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    # 4. Configure Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "formatted_text",
        max_seq_length = config['max_seq_length'],
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config['batch_size'],
            gradient_accumulation_steps = config['gradient_accumulation_steps'],
            warmup_steps = 5,
            max_steps = config['max_steps'],
            learning_rate = config['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = config['optimizer'],
            weight_decay = config['weight_decay'],
            lr_scheduler_type = config['lr_scheduler_type'],
            seed = config.get('seed', 3407),
            output_dir = "outputs",
        ),
    )
    
    # 5. Train
    print("Starting training!")
    trainer_stats = trainer.train()
    
    # 6. Save checkpoint
    output_dir = config['output_dir']
    print(f"Saving model adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
