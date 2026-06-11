# Qwen2.5-7B-Instruct — CFPB Banking Complaint Categorisation

A domain-adapted large language model fine-tuned on the CFPB Consumer Complaint Database to automatically convert unstructured customer complaint narratives into structured ticket metadata for banking operations teams.

---

## The Problem

Financial institutions process thousands of customer complaints daily across mobile apps, websites, contact centres, email, and regulatory portals. These complaints arrive as free-form text — often incomplete, ambiguous, or written by customers who do not know which banking product or issue category applies to their situation.

The result is predictable: complaints get routed to the wrong team, require manual review and reassignment, and take longer to resolve than they should. Traditional classification models handle one label at a time and struggle with the nuanced language of consumer finance. A rule-based keyword system breaks down the moment a customer phrases something slightly differently.

This model addresses that by treating complaint categorisation as a **structured generation task** — the model reads the complaint narrative and produces all four required ticket fields in a single inference step.

---

## What the Model Does

Given a customer complaint narrative, the model outputs a structured JSON object containing:

```json
{
  "product":     "Checking or savings account",
  "sub_product": "Checking account",
  "issue":       "Unauthorized transactions or other transaction problem",
  "sub_issue":   "Debit card issue"
}
```

These four fields map directly to the CFPB Consumer Complaint taxonomy and can be consumed directly by complaint management systems, business rule engines, and routing workflows — no manual classification required.

### Example

**Input complaint:**
> *"I reported fraudulent transactions on my debit card and the bank reversed my provisional credit without explaining the investigation outcome. I have been trying to reach someone for three weeks and keep getting transferred."*

**Model output:**
```json
{
  "product":     "Checking or savings account",
  "sub_product": "Checking account",
  "issue":       "Unauthorized transactions or other transaction problem",
  "sub_issue":   "Debit card issue"
}
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | LoRA (Low-Rank Adaptation) via PEFT |
| Training hardware | AMD Instinct MI300X (192 GB VRAM) |
| Training backend | ROCm 7.2.4 / HIP |
| Model precision | bfloat16 |
| Task type | Structured JSON generation (causal LM) |
| Output format | JSON with 4 fields: product, sub_product, issue, sub_issue |

---

## Training Configuration

### LoRA Adapter

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | ~1% of total model parameters |

The base model weights are fully frozen. Only the LoRA adapter matrices are updated during training, making this efficient both in compute and storage — the saved adapter is significantly smaller than the full model.

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 5 (with early stopping, patience=3) |
| Batch size per device | 8 |
| Gradient accumulation steps | 4 |
| Effective batch size | 32 |
| Learning rate | 1e-4 |
| Optimiser | AdamW (PyTorch native) |
| LR scheduler | Linear |
| Precision | bf16 |
| Max sequence length | 1024 tokens |

Early stopping was applied with a patience of 3 evaluation checkpoints. Prior experiments on smaller model variants showed validation loss plateauing around epoch 2–3, so early stopping prevents wasted compute without sacrificing quality.

### Dataset

**Source:** CFPB Consumer Complaint Database (formatted as multi-turn chat JSONL)

**Splits used:**

| Split | Size |
|-------|------|
| Train | Full dataset (no cap) |
| Validation | 500 |
| Test | 500 |

**Sampling strategy:** Training data was sampled using proportional stratification by `product × issue` combination. This ensures that long-tail complaint categories — which would appear only once or twice in a random 500-sample draw — receive proportional representation. Without this, the model sees most issue labels fewer than 3 times, which is insufficient for reliable generation.

**Chat template:** Qwen's built-in `apply_chat_template` was used to format each example into a single training string with `<|im_start|>` / `<|im_end|>` special tokens. The assistant turn (the JSON output) was included in full — no generation prompt was added at training time.

---

## Inference with Constrained Decoding

At inference time, this model uses a **two-pass constrained decoding** approach:

1. **Pass 1** — Standard greedy decoding generates the JSON output.
2. **Pass 2** — Each field value is snapped to the nearest canonical CFPB label using TF-IDF cosine similarity (unigram + bigram features).

This matters because the CFPB taxonomy contains 80+ canonical issue strings with very similar phrasing. A model that generates *"unauthorized transaction"* when the canonical label is *"unauthorized transactions or other transaction problem"* would score zero on exact match — but is semantically correct. The constrained decoder corrects these surface-level mismatches without changing the underlying prediction.

Jaccard similarity was evaluated as an alternative snapping strategy but proved insufficient for near-duplicate labels (e.g., *"problem with fees"* vs *"other fee"*) where single-word differences produce high Jaccard overlap. TF-IDF on bigrams separates these reliably.

---

## Evaluation Results

Evaluated on 500 held-out test examples from the CFPB dataset. Baseline is the unmodified `Qwen2.5-7B-Instruct` base model with no fine-tuning.

### Primary Metrics — Structured JSON Extraction

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| Exact JSON Match | 0.0000 | 0.2280 | +0.2280 |
| Avg Field Accuracy | 0.0030 | 0.5925 | +0.5895 |
| Micro F1 | 0.0030 | 0.5925 | +0.5895 |
| Macro F1 | 0.0008 | 0.2395 | +0.2387 |
| Weighted F1 | 0.0059 | 0.5814 | +0.5755 |

**Per-field accuracy:**

| Field | Baseline | Fine-tuned | Δ |
|-------|----------|------------|---|
| product | 0.010 | **0.910** | +0.900 |
| sub_product | 0.002 | **0.628** | +0.626 |
| issue | 0.000 | **0.336** | +0.336 |
| sub_issue | 0.000 | **0.496** | +0.496 |

`product` accuracy of 91% is expected — the CFPB product taxonomy has around a dozen top-level categories and the model learns them well. `issue` at 33.6% reflects the genuine difficulty of the field: 80+ canonical strings with overlapping phrasing, many appearing infrequently even in the full training set.

### Secondary Metrics — Generative Quality

These metrics measure output fluency and n-gram overlap. They are secondary to the structured metrics above, but confirm the model is generating coherent, well-formed text.

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| ROUGE-1 | 0.4592 | 0.7035 | +0.2443 |
| ROUGE-2 | 0.2523 | 0.6049 | +0.3526 |
| ROUGE-L | 0.4258 | 0.6915 | +0.2657 |
| BLEU | 0.0003 | 0.1905 | +0.1902 |
| SacreBLEU | 19.77 | 65.07 | +45.30 |
| METEOR | 0.1133 | 0.6309 | +0.5176 |

The SacreBLEU jump from 19.77 to 65.07 and METEOR from 0.11 to 0.63 indicate the fine-tuned model is generating outputs that are not just structurally similar to references, but lexically aligned — which for this task means using the correct canonical CFPB terminology consistently.

---

## How to Use

### Load the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path    = "your-hf-username/qwen2.5-7b-cfpb-complaint-categorisation"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base      = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
```

### Run inference

```python
def categorise_complaint(complaint_text: str, model, tokenizer) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a banking complaint classification assistant. "
                "Given a consumer complaint narrative, extract the CFPB ticket fields "
                "as a JSON object with keys: product, sub_product, issue, sub_issue."
            ),
        },
        {
            "role": "user",
            "content": complaint_text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated  = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return generated


complaint = """
I reported fraudulent transactions on my debit card and the bank reversed
my provisional credit without explaining the investigation outcome.
"""

result = categorise_complaint(complaint, model, tokenizer)
print(result)
# {"product": "Checking or savings account", "sub_product": "Checking account",
#  "issue": "Unauthorized transactions or other transaction problem",
#  "sub_issue": "Debit card issue"}
```

---

## Dependencies

```
transformers==4.44.0
peft==0.12.0
accelerate==0.34.0
datasets==2.21.0
torch (ROCm-compatible build for AMD, or standard CUDA build)
scikit-learn
rouge-score
sacrebleu
nltk
```

---

## Limitations

- **CFPB taxonomy only.** The model is trained on and constrained to CFPB Consumer Complaint Database labels. It is not a general-purpose complaint classifier and should not be used with complaint taxonomies from other regulatory bodies or internal systems without retraining.
- **Issue field accuracy.** The `issue` field (33.6% accuracy) is the weakest link. The CFPB issue taxonomy contains 80+ canonical strings with overlapping phrasing. Expanding training data and further tuning the constrained decoder are the most direct paths to improvement.
- **English language only.** All training data is in English. Performance on non-English complaints is untested and likely poor.
- **Context length.** Complaints longer than 1024 tokens will be truncated. Most CFPB complaints are well within this limit, but very long narratives may lose relevant context.

---

## Intended Use

This model is intended for use by:
- Banking operations teams automating first-touch complaint categorisation
- Compliance teams processing regulatory complaint filings
- Contact centre platforms routing incoming complaints before agent assignment
- Research teams studying LLM adaptation for financial NLP tasks

It is not intended for consumer-facing deployment without human review of outputs, or for use in jurisdictions where automated complaint classification decisions have legal or regulatory implications without appropriate oversight.

---

## Training Infrastructure

Trained on an AMD Instinct MI300X GPU (192 GB HBM3 VRAM) running ROCm 7.2.4. The training stack is fully ROCm-native — `bitsandbytes` (CUDA-only) is not used. Model precision is bfloat16, which is the native compute type for the CDNA3 architecture.

---

## Citation

If you use this model in research or production, please cite the CFPB Consumer Complaint Database as the data source:

```
Consumer Financial Protection Bureau (CFPB)
Consumer Complaint Database
https://www.consumerfinance.gov/data-research/consumer-complaints/
```
