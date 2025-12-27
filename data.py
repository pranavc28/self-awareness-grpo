"""Dataset loading and balancing for FEVER GRPO training."""
import random
from datasets import load_dataset

LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_fever_balanced(num_na: int = 500, num_pass: int = 250, num_fail: int = 250, seed: int = 42) -> list[dict]:
    """Load FEVER dataset with balanced sampling."""
    random.seed(seed)
    ds = load_dataset("fever", "v1.0", split="labelled_dev", trust_remote_code=True)
    
    by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in ds:
        label = ex.get("label")
        if label in by_label:
            by_label[label].append(ex)
    
    for lbl, count in [("NOT ENOUGH INFO", num_na), ("SUPPORTS", num_pass), ("REFUTES", num_fail)]:
        if len(by_label[lbl]) < count:
            raise ValueError(f"Not enough {lbl}: need {count}, have {len(by_label[lbl])}")
    
    sampled = []
    for lbl, count in [("NOT ENOUGH INFO", num_na), ("SUPPORTS", num_pass), ("REFUTES", num_fail)]:
        sampled.extend(random.sample(by_label[lbl], count))
    
    random.shuffle(sampled)
    
    result = []
    for ex in sampled:
        result.append({
            "id": ex["id"],
            "prompt": f"Claim: {ex['claim']}\n\nClassify this claim as PASS (supported), FAIL (refuted), or NA (not enough info).",
            "label": LABEL_MAP[ex["label"]],
            "original_label": ex["label"],
            "claim": ex["claim"],
        })
    return result

def build_prompt(example: dict) -> str:
    """Build the full prompt for a training example."""
    return f"""{example['prompt']}

Return EXACTLY in this format:
LABEL=<NA|PASS|FAIL>
CONF=<float 0 to 1 with 2 decimals>
RATIONALE=<one sentence or empty>
"""

