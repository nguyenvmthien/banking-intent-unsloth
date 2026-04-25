"""
Banking intent classification using Qwen3-8B + Unsloth.

Inference strategy: direct generation via the model's official chat template,
then fuzzy-match the output to the nearest known Banking77 label.

Required interface (per assignment spec):

    class IntentClassification:
        def __init__(self, model_path): ...   # model_path = path to inference.yaml
        def __call__(self, message): ...      # returns predicted label string
"""

import os
import re
import yaml
import pandas as pd
import torch
from difflib import get_close_matches
from unsloth import FastLanguageModel

try:
    from langsmith import traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    def traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

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


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _match_label(generated: str) -> str:
    """Fuzzy-match raw model output to the nearest known Banking77 label."""
    norm_gen = _normalize(generated)
    norm_map = {_normalize(lbl): lbl for lbl in INTENT_LABELS}

    if norm_gen in norm_map:
        return norm_map[norm_gen]

    for norm_lbl, orig_lbl in norm_map.items():
        if norm_lbl in norm_gen or norm_gen in norm_lbl:
            return orig_lbl

    matches = get_close_matches(norm_gen, list(norm_map.keys()), n=1, cutoff=0.55)
    if matches:
        return norm_map[matches[0]]

    return generated


class IntentClassification:
    """
    Banking intent classifier — loads a Qwen3-8B (LoRA fine-tuned) model
    and classifies a customer message into one of 77 Banking77 intent labels.

    Supports three inference modes configured via inference.yaml:
      - finetuned  (default): LoRA-adapted model
      - zero_shot : base model, no examples
      - few_shot  : base model + k examples in system prompt

    Usage:
        clf = IntentClassification("configs/inference.yaml")
        label = clf("I lost my card, how do I order a new one?")
    """

    def __init__(self, model_path: str, mode: str = "finetuned",
                 few_shot_k: int = 5, few_shot_data_path: str = None):
        """
        Args:
            model_path: Path to inference.yaml config file.
            mode: One of "finetuned", "zero_shot", "few_shot".
            few_shot_k: Number of examples to use in few_shot mode.
            few_shot_data_path: Optional path to train.csv for few-shot sampling.
        """
        if mode not in ("zero_shot", "few_shot", "finetuned"):
            raise ValueError(f"Invalid mode: {mode!r}. Choose from: zero_shot, few_shot, finetuned")

        with open(model_path) as f:
            self.config = yaml.safe_load(f)

        self.config_dir = os.path.dirname(os.path.abspath(model_path))
        self.mode = mode
        self.few_shot_k = few_shot_k
        self.few_shot_examples: list[tuple[str, str]] = []

        checkpoint = (
            self.config["model_path"] if mode == "finetuned"
            else self.config["model_name"]
        )

        print(f"[{mode.upper()}] Loading model: {checkpoint}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=self.config["max_seq_length"],
            dtype=None,
            load_in_4bit=self.config["load_in_4bit"],
        )
        FastLanguageModel.for_inference(self.model)

        if mode == "few_shot":
            self._load_few_shot_examples(few_shot_data_path)

        self._setup_langsmith()

    def _load_few_shot_examples(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.normpath(
                os.path.join(self.config_dir, "../sample_data/train.csv")
            )
        df = pd.read_csv(data_path)
        sampled = (
            df.groupby("intent_name")
            .apply(lambda g: g.sample(1, random_state=42))
            .reset_index(drop=True)
            .sample(min(self.few_shot_k, df["intent_name"].nunique()), random_state=42)
        )
        self.few_shot_examples = list(zip(sampled["text"], sampled["intent_name"]))
        print(f"Loaded {len(self.few_shot_examples)} few-shot examples.")

    def _setup_langsmith(self):
        api_key = os.environ.get("LANGSMITH_API_KEY") or self.config.get("langsmith_api_key")
        if api_key and _LANGSMITH_AVAILABLE:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            project = self.config.get("langsmith_project", "banking-intent-v2")
            os.environ["LANGCHAIN_PROJECT"] = project
            print(f"LangSmith tracing enabled — project: {project}")
        else:
            print("LangSmith tracing disabled.")

    def _build_messages(self, user_text: str) -> list[dict]:
        system = SYSTEM_PROMPT
        if self.mode == "few_shot" and self.few_shot_examples:
            examples_str = "\n".join(
                f"User: {text}\nLabel: {label}"
                for text, label in self.few_shot_examples
            )
            system = system + "\n\nExamples:\n" + examples_str
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    @traceable(name="predict_intent")
    def predict(self, message: str) -> dict:
        """Run inference and return a dict with input, raw_output, and label."""
        messages = self._build_messages(message)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.get("max_new_tokens", 64),
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.shape[-1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Qwen3 thinking block is always generated internally; strip it from output
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        label = _match_label(raw_output)
        return {"input": message, "raw_output": raw_output, "label": label}

    def __call__(self, message: str) -> str:
        """Classify a single banking customer message. Returns a Banking77 label string."""
        return self.predict(message)["label"]


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "../configs/inference.yaml"
    test_message = "I lost my credit card yesterday, how do I order a new one?"

    for mode in ("zero_shot", "few_shot", "finetuned"):
        print(f"\n[{mode.upper()}]")
        try:
            clf = IntentClassification(config_file, mode=mode)
            result = clf.predict(test_message)
            print(f"  input      : {result['input']}")
            print(f"  raw_output : {result['raw_output']}")
            print(f"  label      : {result['label']}")
        except Exception as e:
            print(f"  ERROR: {e}")
