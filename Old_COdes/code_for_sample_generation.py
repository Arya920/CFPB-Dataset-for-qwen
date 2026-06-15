# ── DEMO EXAMPLE GENERATION ───────────────────────────────────────────────────
# Purpose : Find compelling before-vs-after examples for hackathon demo.
# Output  : best_demo_examples.json  (~10 examples where FT beats base clearly)
# Note    : This cell will be removed after the presentation.
# ─────────────────────────────────────────────────────────────────────────────

import json as _json_demo
from tqdm import tqdm as _tqdm_demo

DEMO_TEST_SIZE = 250          # number of test examples to score
DEMO_OUTPUT_N  = 10           # examples to save
DEMO_OUTPUT_PATH = f"{RESULTS_DIR}/best_demo_examples.json"

# We need the base (unfinetuned) model for comparison.
# Load it fresh so we don't disturb the fine-tuned model already in memory.
from transformers import AutoModelForCausalLM as _AutoModel
from peft import PeftModel as _PeftModel

print("Loading base model for demo comparison...")
_base_model = _AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
_base_model.eval()

# Fine-tuned model is already `model` in the current namespace.
_ft_model = model
_ft_model.eval()

# ── Step 1: Run inference on all 250 test examples ───────────────────────────

print(f"\nRunning inference on {DEMO_TEST_SIZE} test examples with base model...")
_demo_data = test_data[:DEMO_TEST_SIZE]

_base_preds = []
for ex in _tqdm_demo(_demo_data, desc="Base model"):
    _base_preds.append(predict(ex["messages"], _base_model))

print(f"\nRunning inference on {DEMO_TEST_SIZE} test examples with fine-tuned model...")
_ft_preds = []
for ex in _tqdm_demo(_demo_data, desc="Fine-tuned model"):
    _ft_preds.append(constrained_predict(ex["messages"], _ft_model, LABEL_VOCAB, SNAPPERS))

# ── Step 2: Compute per-example scores ───────────────────────────────────────

def _score_single(pred: str, ref: str) -> dict:
    """Compute evaluation scores for a single prediction/reference pair."""
    return compute_field_metrics([pred], [ref])


print("\nScoring all examples...")
_all_examples = []

for idx, (ex, base_pred, ft_pred) in enumerate(_tqdm_demo(
    zip(_demo_data, _base_preds, _ft_preds), total=len(_demo_data), desc="Scoring"
)):
    ref_str = extract_reference(ex["messages"])
    ref_parsed  = parse_json_output(ref_str)

    base_score = _score_single(base_pred, ref_str)
    ft_score   = _score_single(ft_pred,   ref_str)

    # Extract complaint text from the user turn
    complaint = next(
        (m["content"] for m in reversed(ex["messages"]) if m["role"] == "user"), ""
    )

    _all_examples.append({
        "idx"                  : idx,
        "complaint"            : complaint,
        "ground_truth"         : ref_parsed,
        "base_pred"            : parse_json_output(base_pred),
        "ft_pred"              : parse_json_output(ft_pred),
        "base_evaluation_score": base_score,
        "ft_evaluation_score"  : ft_score,
    })

# ── Step 3: Rank by improvement ──────────────────────────────────────────────

def _field_correct(pred_dict, gt_dict, field):
    return int(
        str(pred_dict.get(field, "")).strip().lower()
        ==
        str(gt_dict.get(field, "")).strip().lower()
    )


def _improvement_score(entry: dict) -> float:
    """
    Rank examples based on actual taxonomy correction.

    Higher score means:
    - More fields became correct after fine-tuning.
    - Product correction is heavily rewarded.
    """

    gt = entry["ground_truth"]
    bp = entry["base_pred"]
    fp = entry["ft_pred"]

    base_correct = sum([
        _field_correct(bp, gt, "product"),
        _field_correct(bp, gt, "sub_product"),
        _field_correct(bp, gt, "issue"),
        _field_correct(bp, gt, "sub_issue"),
    ])

    ft_correct = sum([
        _field_correct(fp, gt, "product"),
        _field_correct(fp, gt, "sub_product"),
        _field_correct(fp, gt, "issue"),
        _field_correct(fp, gt, "sub_issue"),
    ])

    improvement = ft_correct - base_correct

    product_bonus = (
        _field_correct(fp, gt, "product")
        -
        _field_correct(bp, gt, "product")
    ) * 5

    return improvement + product_bonus


print("\nRanking examples by improvement...")
_all_examples.sort(key=_improvement_score, reverse=True)

# ── Step 4: Save top examples ────────────────────────────────────────────────

def _build_improvement_summary(entry: dict):

    gt = entry["ground_truth"]
    bp = entry["base_pred"]
    fp = entry["ft_pred"]

    summary = {}

    for field in ["product", "sub_product", "issue", "sub_issue"]:

        base_correct = _field_correct(bp, gt, field)
        ft_correct = _field_correct(fp, gt, field)

        summary[field] = {
            "base_correct": base_correct,
            "ft_correct": ft_correct,
            "improved": ft_correct - base_correct
        }

    summary["base_total_correct"] = sum(
        summary[f]["base_correct"]
        for f in ["product", "sub_product", "issue", "sub_issue"]
    )

    summary["ft_total_correct"] = sum(
        summary[f]["ft_correct"]
        for f in ["product", "sub_product", "issue", "sub_issue"]
    )

    summary["total_improvement"] = (
        summary["ft_total_correct"]
        - summary["base_total_correct"]
    )

    return summary


_filtered_examples = []

for entry in _all_examples:

    imp = _build_improvement_summary(entry)

    if (
        imp["base_total_correct"] <= 1
        and
        imp["ft_total_correct"] >= 3
    ):
        _filtered_examples.append(entry)

if len(_filtered_examples) < DEMO_OUTPUT_N:
    _filtered_examples = _all_examples

_top_examples = []

for entry in _filtered_examples[:DEMO_OUTPUT_N]:

    _top_examples.append({
        "idx": entry["idx"],
        "complaint": entry["complaint"],
        "ground_truth": entry["ground_truth"],
        "base_prediction": entry["base_pred"],
        "ft_prediction": entry["ft_pred"],
        "base_metrics": entry["base_evaluation_score"],
        "ft_metrics": entry["ft_evaluation_score"],
        "improvement": _build_improvement_summary(entry),
    })

with open(DEMO_OUTPUT_PATH, "w") as _f:
    _json_demo.dump(_top_examples, _f, indent=2)

print(f"\nSaved {len(_top_examples)} demo examples → {DEMO_OUTPUT_PATH}")