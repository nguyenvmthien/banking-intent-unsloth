"""
Fine-tune Qwen3-8B on Banking77 using Unsloth + SFTTrainer.

Resumption: if output_dir already contains a checkpoint-* folder,
training resumes from the latest one automatically.

Hub push: if hub_model_id is set in config, checkpoints are pushed to
HuggingFace Hub every hub_push_every_n_steps steps so they survive
Kaggle/Colab session death.
"""

import os
import glob
import yaml
import torch
import pandas as pd
from datasets import Dataset
from huggingface_hub import login as hf_login
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

INTENT_LABELS = [
    "Refund_not_showing_up", "activate_my_card", "age_limit",
    "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire",
    "card_acceptance", "card_arrival", "card_delivery_estimate",
    "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
]

LABELS_STR = "\n".join(f"- {lbl}" for lbl in INTENT_LABELS)

SYSTEM_PROMPT = (
    "You are a banking customer support intent classifier. "
    "Classify the user's message into exactly one of the following intent labels and "
    "respond with only that label, nothing else.\n\n"
    "Valid labels:\n"
    f"{LABELS_STR}"
)


def format_sample(tokenizer, user_text: str, label: str) -> str:
    """Format a single training example using the model's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": label},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the most recent checkpoint-N folder, or None."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


class HubPushCallback(TrainerCallback):
    """Push adapter to HF Hub every push_every steps."""

    def __init__(self, model, tokenizer, repo_id: str, push_every: int):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_id = repo_id
        self.push_every = push_every

    def on_step_end(self, args, state, control, **kwargs):
        """Push adapter weights to HF Hub at the configured step interval."""
        if state.global_step % self.push_every == 0:
            print(f"\n[HubPush] step {state.global_step} → pushing to {self.repo_id}")
            self.model.push_to_hub(self.repo_id, token=True)
            self.tokenizer.push_to_hub(self.repo_id, token=True)


def main():
    """Load config, build LoRA model, format dataset, and run SFT training."""
    config_path = "../configs/train.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # HF login — reads HF_TOKEN from env (set as Kaggle secret)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        print("Logged in to HuggingFace Hub.")
    else:
        print("HF_TOKEN not set — Hub push disabled.")

    output_dir = config["output_dir"]
    resume_checkpoint = find_latest_checkpoint(output_dir)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("No checkpoint found — starting fresh.")

    print(f"Loading model: {config['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume_checkpoint or config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 3407),
    )

    print(f"Loading data from {config['data_path']}...")
    df = pd.read_csv(config["data_path"])
    dataset = Dataset.from_pandas(df)

    def formatting_func(examples):
        return {
            "formatted_text": [
                format_sample(tokenizer, text, label)
                for text, label in zip(examples["text"], examples["intent_name"])
            ]
        }

    dataset = dataset.map(formatting_func, batched=True)

    hub_model_id = config.get("hub_model_id", "").strip()
    hub_push_callback = None
    if hub_model_id and hf_token:
        hub_push_callback = HubPushCallback(
            model, tokenizer, hub_model_id,
            push_every=config.get("hub_push_every_n_steps", 200),
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=[hub_push_callback] if hub_push_callback else None,
        args=SFTConfig(
            dataset_text_field="formatted_text",
            max_seq_length=config["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=10,
            num_train_epochs=config.get("num_train_epochs", 3),
            max_steps=config.get("max_steps", -1),
            learning_rate=config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim=config["optimizer"],
            weight_decay=config["weight_decay"],
            lr_scheduler_type=config["lr_scheduler_type"],
            seed=config.get("seed", 3407),
            output_dir=output_dir,
            save_steps=config.get("save_steps", 100),
            save_total_limit=config.get("save_total_limit", 3),
        ),
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_dir = config["output_dir"]
    print(f"Saving final adapter to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if hub_model_id and hf_token:
        print(f"Pushing final adapter to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id, token=True)
        tokenizer.push_to_hub(hub_model_id, token=True)

    print("Done.")


if __name__ == "__main__":
    main()
