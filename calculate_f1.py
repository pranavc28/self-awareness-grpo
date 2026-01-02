"""Calculate F1 scores per outcome for FEVER classification results.

Works with outputs from both classify_fever.py and sample_grpo.py.
Supports both original FEVER labels (SUPPORTS, REFUTES, NOT ENOUGH INFO)
and simplified labels (PASS, FAIL, NA).
"""

import json
import sys
from collections import defaultdict
import re

# Simplified labels used by classify_fever.py and sample_grpo.py
LABELS = ["PASS", "FAIL", "NA"]

# Mapping from original FEVER labels to simplified labels
LABEL_MAP = {
    "SUPPORTS": "PASS",
    "REFUTES": "FAIL", 
    "NOT ENOUGH INFO": "NA",
    # Already simplified labels map to themselves
    "PASS": "PASS",
    "FAIL": "FAIL",
    "NA": "NA"
}


def normalize_label(label: str) -> str:
    """Normalize a label to the simplified format (PASS/FAIL/NA)."""
    return LABEL_MAP.get(label, label)


def load_results(path: str) -> dict:
    """Load results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def _validate_raw_response(text: str) -> tuple[str, float, bool]:
    """Validate LABEL/CONF formatting from `raw_response` using the *same* logic as training.

    This mirrors `train_grpo.py:parse_output`:
    - label defaults to None
    - conf defaults to 0.5
    - valid becomes False if LABEL is missing OR CONF is missing OR CONF isn't parseable
    """
    label, conf, valid = None, 0.5, True

    if not isinstance(text, str):
        return None, conf, False

    text = text.replace("<", "").replace(">", "")
    m = re.search(r"LABEL\s*=\s*(NA|PASS|FAIL)", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
    else:
        valid = False
    m = re.search(r"CONF\s*=\s*([\d.]+)", text)
    if m:
        try:
            conf = float(m.group(1))
            conf = max(0.0, min(1.0, conf))
        except:
            valid = False
    else:
        valid = False
    return label, conf, valid


def is_valid_result(r: dict) -> bool:
    """Determine whether a single result row should be included in metrics."""
    if "golden_label" not in r or "predicted_label" not in r:
        return False

    # Prefer explicit validity signal if present (sample_grpo.py emits this).
    if r.get("format_valid") is False:
        return False

    golden = normalize_label(r.get("golden_label"))
    predicted = normalize_label(r.get("predicted_label"))
    if golden not in LABELS or predicted not in LABELS:
        return False

    # If `format_valid` is absent but `raw_response` exists (sample_control.py),
    # validate the formatting from the raw response itself.
    if "format_valid" not in r and "raw_response" in r:
        _, _, valid = _validate_raw_response(r.get("raw_response", ""))
        if not valid:
            return False

    return True


def filter_valid_results(results: list[dict]) -> tuple[list[dict], int]:
    """Filter to only valid results and return (valid_results, num_skipped)."""
    valid_results: list[dict] = []
    skipped = 0
    for r in results:
        if is_valid_result(r):
            valid_results.append(r)
        else:
            skipped += 1
    return valid_results, skipped


def calculate_metrics(results: list[dict]) -> dict:
    """Calculate precision, recall, and F1 for each label."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    
    for r in results:
        # Normalize labels to simplified format
        golden = normalize_label(r["golden_label"])
        predicted = normalize_label(r["predicted_label"])
        
        if golden == predicted:
            tp[golden] += 1
        else:
            fp[predicted] += 1
            fn[golden] += 1
    
    metrics = {}
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] > 0 else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    
    return metrics


def calculate_and_print(data: dict, output_path: str = None) -> dict:
    """Calculate metrics, print them, and optionally save to file."""
    results = data["results"]
    valid_results, skipped = filter_valid_results(results)
    metrics = calculate_metrics(valid_results)
    
    # Calculate support (count per label)
    support = defaultdict(int)
    for r in valid_results:
        label = normalize_label(r["golden_label"])
        support[label] += 1
    
    correct = sum(1 for r in valid_results 
                  if normalize_label(r["golden_label"]) == normalize_label(r["predicted_label"]))
    accuracy = (correct / len(valid_results) * 100) if valid_results else 0.0
    
    # Get source info
    model = data.get("model", "Unknown")
    source = data.get("source", "Unknown")
    checkpoint = data.get("checkpoint", "")
    
    print(f"\n{'='*70}")
    print(f"FEVER F1 SCORES  |  Model: {model}  |  Source: {source}")
    if checkpoint and checkpoint != "base":
        print(f"Checkpoint: {checkpoint}")
    if skipped:
        print(f"Filtered out invalid results: {skipped}/{len(results)}")
    print(f"{'='*70}")
    print(f"\n{'Label':<10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'N':>6}")
    print("-" * 55)
    
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<10} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f} {support[label]:>6}")
    
    print("-" * 55)
    
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(LABELS)
    weighted_f1 = (sum(metrics[l]["f1"] * support[l] for l in LABELS) / len(valid_results)) if valid_results else 0.0
    
    print(f"\n{'Accuracy:':<15} {accuracy:>10.1f}%")
    print(f"{'Macro F1:':<15} {macro_f1:>10.4f}")
    print(f"{'Weighted F1:':<15} {weighted_f1:>10.4f}")
    if not valid_results:
        print("(No valid results after filtering; metrics are 0.0)")
    print()
    
    output_data = {
        "model": model,
        "source": source,
        "checkpoint": checkpoint,
        "num_total": len(results),
        "num_valid": len(valid_results),
        "num_invalid_skipped": skipped,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": metrics
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {output_path}")
    
    return output_data


def main():
    """Process one or both result files."""
    # Default files to process
    files_to_process = []
    
    if len(sys.argv) > 1:
        # User specified file(s)
        files_to_process = sys.argv[1:]
    else:
        # Process both default files if they exist
        import os
        if os.path.exists("fever_classification_results.json"):
            files_to_process.append("fever_classification_results.json")
        if os.path.exists("grpo_sample_results.json"):
            files_to_process.append("grpo_sample_results.json")
    
    if not files_to_process:
        print("No result files found. Run classify_fever.py or sample_grpo.py first.")
        return
    
    for input_path in files_to_process:
        try:
            data = load_results(input_path)
            
            # Determine output path based on source
            source = data.get("source", "unknown")
            if source == "classify_fever":
                output_path = "fever_f1_metrics.json"
            elif source == "sample_grpo":
                output_path = "grpo_f1_metrics.json"
            else:
                # Fallback based on input filename
                base = input_path.replace(".json", "")
                output_path = f"{base}_f1_metrics.json"
            
            calculate_and_print(data, output_path)
            
        except FileNotFoundError:
            print(f"File not found: {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    main()
