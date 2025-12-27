"""Calculate F1 scores per outcome for FEVER classification results."""

import json
from collections import defaultdict

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_results(path: str = "fever_classification_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def calculate_metrics(results: list[dict]) -> dict:
    """Calculate precision, recall, and F1 for each label."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    
    for r in results:
        golden, predicted = r["golden_label"], r["predicted_label"]
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


def main():
    data = load_results()
    results = data["results"]
    metrics = calculate_metrics(results)
    
    support = defaultdict(int)
    for r in results:
        support[r["golden_label"]] += 1
    
    correct = sum(1 for r in results if r["golden_label"] == r["predicted_label"])
    accuracy = correct / len(results) * 100
    
    print(f"\n{'='*65}")
    print(f"FEVER F1 SCORES  |  Model: {data.get('model', 'Unknown')}")
    print(f"{'='*65}")
    print(f"\n{'Label':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'N':>6}")
    print("-" * 65)
    
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<20} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f} {support[label]:>6}")
    
    print("-" * 65)
    
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(LABELS)
    weighted_f1 = sum(metrics[l]["f1"] * support[l] for l in LABELS) / len(results)
    
    print(f"\n{'Accuracy:':<20} {accuracy:>12.1f}%")
    print(f"{'Macro F1:':<20} {macro_f1:>12.4f}")
    print(f"{'Weighted F1:':<20} {weighted_f1:>12.4f}")
    print()
    
    with open("fever_f1_metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_label": metrics
        }, f, indent=2)
    
    print("Saved to fever_f1_metrics.json")


if __name__ == "__main__":
    main()
