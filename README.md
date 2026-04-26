# Banking Intent Classification — Qwen3-8B + Unsloth

Fine-tune **Qwen3-8B** (4-bit quantized via Unsloth) with LoRA adapters on the **Banking77** dataset to classify 77 banking customer intent categories.

Supports two inference modes: **zero-shot** and **fine-tuned**.

## Video Demonstration

[Video Demo](https://drive.google.com/your-drive-link-here)

---

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── train.py              # Fine-tune Qwen3-8B with LoRA via Unsloth
│   ├── inference.py          # IntentClassification class
│   ├── evaluate.py           # Evaluate accuracy on test set
│   └── preprocess_data.py    # Download Banking77 from HuggingFace
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference configuration (set LANGSMITH_API_KEY here)
├── sample_data/
│   ├── train.csv             # Training set (~10,000 samples)
│   └── test.csv              # Test set (~3,000 samples)
├── outputs/
│   └── checkpoint/           # Saved LoRA adapter after training
├── train_kaggle.ipynb        # End-to-end Kaggle notebook (T4/P100)
├── train_colab.ipynb         # End-to-end Colab notebook (A100, Drive checkpoint)
├── train.sh                  # preprocess + train
├── inference.sh              # run inference
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
| Epochs | 1 |
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

The `IntentClassification` class in `scripts/inference.py` supports two modes:

| Mode | Model | `max_new_tokens` | Description |
|------|-------|-----------------|-------------|
| `finetuned` (default) | LoRA adapter | 32 | Fine-tuned on Banking77, outputs label directly |
| `zero_shot` | Base Qwen3-8B | 512 | No examples — model reasons via thinking block |

Batched inference is supported via `predict_batch(messages)` for faster evaluation.

**Usage example:**

```python
import sys
sys.path.insert(0, "scripts")
from inference import IntentClassification

clf = IntentClassification("configs/inference.yaml")  # finetuned (default)
label = clf("I lost my credit card, how do I order a replacement?")
print(label)  # e.g. "lost_or_stolen_card"

# Batch inference
results = clf.predict_batch(["I lost my card", "My PIN is blocked"])
```

```bash
cd scripts && python inference.py
```

Expected output:

```
[ZERO_SHOT]
  input      : 'I lost my credit card yesterday, how do I order a new one?'
  raw_output : 'lost_or_stolen_card'
  label      : 'lost_or_stolen_card'

[FINETUNED]
  input      : 'I lost my credit card yesterday, how do I order a new one?'
  raw_output : 'lost_or_stolen_card'
  label      : 'lost_or_stolen_card'
```

---

### 4. Evaluation

```bash
cd scripts

# Fine-tuned model (default)
python evaluate.py

# Base model — zero-shot
python evaluate.py --mode zero_shot

# Both modes with comparison summary
python evaluate.py --mode all

# Override batch size (default from inference.yaml)
python evaluate.py --mode all --batch_size 4
```

#### Results (A100 40GB)

| Mode | Dataset | Accuracy |
|------|---------|----------|
| Zero-Shot (base Qwen3-8B) | 200 stratified samples | 68.00% |
| Fine-Tuned (LoRA, 1 epoch) | 200 stratified samples | 90.00% |
| Fine-Tuned (LoRA, 1 epoch) | Full test set (3,080 samples) | **91.85%** |

---

### 5. Running on GPU

#### Kaggle (`train_kaggle.ipynb`)

1. Kaggle → **Add-ons → Secrets**: add `HF_TOKEN`
2. Notebook settings → **Accelerator**: GPU T4 x2 or P100
3. Notebook settings → **Internet**: On
4. Set `GITHUB_REPO_URL` and `HF_REPO_ID` in the first cell

#### Google Colab (`train_colab.ipynb`) — recommended for A100

1. Runtime → Change runtime type → **A100**
2. Colab left sidebar → 🔑 **Secrets**: add `HF_TOKEN`
3. Set `GITHUB_REPO_URL` and `HF_REPO_ID` in the first cell
4. Checkpoints are saved to Google Drive automatically — survives session restarts

Both notebooks handle: install → data download → training (auto-resume from checkpoint) → evaluate.

---

### 6. LangSmith Tracing (optional)

Set `LANGSMITH_API_KEY` as an environment variable or add it to `configs/inference.yaml` under `langsmith_api_key`. Every `__call__` invocation will be traced to the project specified by `langsmith_project`.
