"""Calculate F1 scores per outcome for FEVER classification results.

Works with outputs from both classify_fever.py and sample_grpo.py.
Supports both original FEVER labels (SUPPORTS, REFUTES, NOT ENOUGH INFO)
and simplified labels (PASS, FAIL, NA).
"""

import json
import sys
from collections import defaultdict

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
    metrics = calculate_metrics(results)
    
    # Calculate support (count per label)
    support = defaultdict(int)
    for r in results:
        label = normalize_label(r["golden_label"])
        support[label] += 1
    
    correct = sum(1 for r in results 
                  if normalize_label(r["golden_label"]) == normalize_label(r["predicted_label"]))
    accuracy = correct / len(results) * 100
    
    # Get source info
    model = data.get("model", "Unknown")
    source = data.get("source", "Unknown")
    checkpoint = data.get("checkpoint", "")
    
    print(f"\n{'='*70}")
    print(f"FEVER F1 SCORES  |  Model: {model}  |  Source: {source}")
    if checkpoint and checkpoint != "base":
        print(f"Checkpoint: {checkpoint}")
    print(f"{'='*70}")
    print(f"\n{'Label':<10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'N':>6}")
    print("-" * 55)
    
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<10} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f} {support[label]:>6}")
    
    print("-" * 55)
    
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(LABELS)
    weighted_f1 = sum(metrics[l]["f1"] * support[l] for l in LABELS) / len(results)
    
    print(f"\n{'Accuracy:':<15} {accuracy:>10.1f}%")
    print(f"{'Macro F1:':<15} {macro_f1:>10.4f}")
    print(f"{'Weighted F1:':<15} {weighted_f1:>10.4f}")
    print()
    
    output_data = {
        "model": model,
        "source": source,
        "checkpoint": checkpoint,
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
