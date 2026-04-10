# Fine-Tuning Intent Detection Model with BANKING77 using Unsloth

This project demonstrates how to fine-tune a large language model (Llama 3 8B) on the BANKING77 dataset using the Unsloth library, for robust intent classification. It completely fulfills the requirements of Project 2.

## Structure

```text
banking-intent-unsloth
|-- scripts
|   |-- train.py                 # Uses Unsloth to fine tune Llama-3
|   |-- inference.py             # Inference loop using the class
|   |-- preprocess_data.py       # Datset split logic
|-- configs
|   |-- train.yaml               # Hyperparameters for training
|   |-- inference.yaml           # Hyperparameters for inference
|-- sample_data
|   |-- train.csv                # Sampled training data
|   |-- test.csv                 # Sampled test data
|-- train.sh                     # Runner script
|-- inference.sh                 # Runner script
|-- requirements.txt             # Depencies
|-- README.md                    # Documentation
```

## Setup Instructions

### Environment Setup (Google Colab / Local Setup)

To use Unsloth correctly, you will usually need a Linux GPU environment or Google Colab. To install the dependencies, simply run:

```bash
pip install -r requirements.txt
```

*Note: For Google Colab, you can directly execute the provided bash scripts and they should work nicely with the standard T4 free tier VMs.*

### 1. Data Preparation

First, download and sample the `BANKING77` dataset. From the root directory, simply run:
```bash
cd scripts
python preprocess_data.py
```
This restricts the amount of data we fine-tune on to be within bounds for simple academic compute constraints.

### 2. Training the Model

The training script reads the parameters from `configs/train.yaml`. You can modify `batch_size`, `learning_rate` and `max_steps` to best fit your target configuration. By default, it will save the adapter weights to `outputs/checkpoint`.

Run training with:
```bash
sh train.sh
```
or 
```bash
cd scripts
python train.py
```

### 3. Inference

The task evaluates using an `IntentClassification` Python class. The implementation handles all the necessary Unsloth generation configurations seamlessly.

Run the test inference logic:
```bash
sh inference.sh
```

Because Unsloth uses Generative AI inference natively, it structures the input using a special instruction prompt template (found in `inference.py`) and allows the LLM to write out the explicit intent mapping.

## Demonstration Video

**TODO:** Add your demonstration video link here.
[Video Demo Link](https://drive.google.com/your-drive-link-here)
