# Banking Intent Classification v2 — Qwen3-8B + Unsloth

Fine-tune **Qwen3-8B** (4-bit quantized via Unsloth) with LoRA adapters on the **Banking77** dataset to classify 77 banking customer intent categories.

Supports three inference modes: **zero-shot**, **few-shot**, and **fine-tuned**.

## Video Demonstration

[Video Demo](https://drive.google.com/your-drive-link-here)

---

## Project Structure

```
banking-intent-v2/
├── scripts/
│   ├── train.py              # Fine-tune Qwen3-8B with LoRA via Unsloth
│   ├── inference.py          # IntentClassification class
│   ├── evaluate.py           # Evaluate accuracy on test set
│   └── preprocess_data.py    # Download Banking77 from HuggingFace
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference configuration
├── sample_data/
│   ├── train.csv             # Training set (~10,000 samples)
│   └── test.csv              # Test set (~3,000 samples)
├── outputs/
│   └── checkpoint/           # Saved LoRA adapter after training
├── train_kaggle.ipynb        # End-to-end Kaggle notebook with resume support
├── train.sh
├── inference.sh
└── requirements.txt
```

---

## Setup

Requires a Linux GPU environment (Kaggle T4/P100 or Google Colab recommended).

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Data Preparation

```bash
cd scripts
python preprocess_data.py
```

Downloads Banking77 from HuggingFace (`PolyAI/banking77`) via direct parquet URLs and saves:
- `sample_data/train.csv` — 10,016 rows (`text`, `label`, `intent_name`)
- `sample_data/test.csv` — 3,084 rows

---

### 2. Fine-tuning

```bash
sh train.sh
# or
cd scripts && python train.py
```

The LoRA adapter is saved to `outputs/checkpoint/`. Training automatically resumes from the latest checkpoint if one exists.

#### Hyperparameters (`configs/train.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-8B` |
| Quantization | 4-bit (QLoRA) |
| Max sequence length | 2048 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q/k/v/o/gate/up/down proj |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Optimizer | `adamw_8bit` |
| Weight decay | 0.01 |
| Checkpoint save every | 100 steps |

---

### 3. Inference

The `IntentClassification` class in `scripts/inference.py` supports three modes:

| Mode | Model | Description |
|------|-------|-------------|
| `finetuned` (default) | LoRA adapter | Fine-tuned on Banking77 |
| `zero_shot` | Base Qwen3-8B | No examples in prompt |
| `few_shot` | Base Qwen3-8B | k examples injected into system prompt |

**Usage example:**

```python
from scripts.inference import IntentClassification

# model_path = path to inference.yaml config file
clf = IntentClassification("configs/inference.yaml")
label = clf("I lost my credit card, how do I order a replacement?")
print(label)  # e.g. "lost_or_stolen_card"
```

```bash
sh inference.sh
# or
cd scripts && python inference.py
```

Expected output:

```
[ZERO_SHOT]
  input      : 'I lost my credit card yesterday, how do I order a new one?'
  raw_output : 'lost_or_stolen_card'
  label      : 'lost_or_stolen_card'

[FEW_SHOT]
  ...

[FINETUNED]
  ...
```

---

### 4. Evaluation

```bash
cd scripts

# Fine-tuned model (default)
python evaluate.py

# Base model — zero-shot
python evaluate.py --mode zero_shot

# Base model — few-shot (5 examples)
python evaluate.py --mode few_shot --few_shot_k 5

# All three modes with comparison summary
python evaluate.py --mode all
```

---

### 5. Kaggle (recommended for GPU)

Open `train_kaggle.ipynb` on Kaggle. Before running:

1. Kaggle → **Add-ons → Secrets**: add `HF_TOKEN` (HuggingFace token with write access)
2. Notebook settings → **Accelerator**: GPU T4 x2 or P100
3. Notebook settings → **Internet**: On
4. Set `HF_REPO_ID` in the first cell to your HuggingFace repo

The notebook handles install → data download → training (with auto-resume if session dies) → sanity check inference.

---

### 6. LangSmith Tracing (optional)

Set `LANGSMITH_API_KEY` as an environment variable or add it to `configs/inference.yaml` under `langsmith_api_key`. Every `__call__` invocation will be traced to the project specified by `langsmith_project`.
