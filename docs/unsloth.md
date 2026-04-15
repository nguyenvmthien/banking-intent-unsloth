# Technical Report: Banking Intent Classification with Llama-3 & Unsloth

## 1. Project Overview
This project implements a high-performance intent detection system for banking customer queries using the **BANKING77** dataset. We leverage **Llama-3 (8B)** and the **Unsloth** library to achieve efficient fine-tuning without compromising accuracy.

## 2. Technical Methodology

### 2.1 Unsloth Library
We chose Unsloth because it provides specialized kernels that make LLM fine-tuning **2x faster** and use **60% less memory**. Key features utilized:
- **4-bit Quantization (QLoRA)**: Reduces VRAM usage from ~32GB to ~5GB, allowing the model to run on a single Tesla T4 GPU (free tier Colab).
- **Native Inference**: Unsloth provides optimized inference kernels that are significantly faster than standard Hugging Face pipelines.

### 2.2 Fine-Tuning Strategy (LoRA)
Instead of updating all 8 billion parameters, we used **Low-Rank Adaptation (LoRA)**. 
- **Target Modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).
- **Rank (r)**: 16
- **Alpha**: 16
- **Dropout**: 0 (Optimal for Unsloth speed).

## 3. Dataset and Preprocessing
- **Source**: BANKING77 (10,003 training samples).
- **Format**: Converted to an instruction-following prompt:
  ```
  ### Instruction: Classify the intent of the following banking customer request.
  ### Input: [User query]
  ### Response: [Intent label]
  ```
- **Sampling**: Initially tested with 10%, scaled up to **100%** for the final submission to ensure maximum coverage of the 77 intent classes.

## 4. Training Configuration
The following hyperparameters were used in the final run:
- **Optimizer**: AdamW 8-bit (Paged)
- **Learning Rate**: 2e-4
- **Epochs**: 1
- **Batch Size**: 2 (with Gradient Accumulation = 4, Total BS = 8)
- **LR Scheduler**: Linear decay
- **Weight Decay**: 0.01

## 5. Evaluation and Comparison
Following the project requirements, we performed a comparison:
1.  **Zero-shot Performance**: The base Llama-3-8B model was evaluated. While it has good general knowledge, it often misses specific labels required by the BANKING77 taxonomy.
2.  **Fine-tuned Performance**: After 1 epoch of training on the full dataset, the model's accuracy on the 3,080 test samples showed a significant improvement, demonstrating the effectiveness of the fine-tuning process.

## 6. Conclusion
The combination of Llama-3 and Unsloth proves to be a robust solution for specialized NLP tasks like intent detection. This setup allows for rapid iteration and high-quality results even on limited computational resources.
