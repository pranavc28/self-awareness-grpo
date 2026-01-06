"""Calibration and self-awareness analysis for GRPO vs baseline models.

Key metrics for self-awareness:
1. AUROC - Can confidence predict correctness?
2. Brier Score - Proper scoring rule for probabilistic predictions
3. Lower conf when wrong - Is model less confident when incorrect?
"""

import json
import sys
import re
import numpy as np
from typing import Optional

LABELS = ["PASS", "FAIL", "NA"]
LABEL_MAP = {"SUPPORTS": "PASS", "REFUTES": "FAIL", "NOT ENOUGH INFO": "NA",
             "PASS": "PASS", "FAIL": "FAIL", "NA": "NA"}


def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


def extract_confidence(result: dict) -> Optional[float]:
    """Extract confidence from result, handling both formats."""
    if "predicted_conf" in result:
        return result["predicted_conf"]
    
    # Try to extract from raw_response
    raw = result.get("raw_response", "")
    m = re.search(r"CONF\s*=\s*([\d.]+)", raw)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except:
            pass
    return None


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    
    results = []
    for r in data["results"]:
        golden = normalize_label(r.get("golden_label", ""))
        predicted = normalize_label(r.get("predicted_label", ""))
        conf = extract_confidence(r)
        
        if golden in LABELS and predicted in LABELS and conf is not None:
            results.append({
                "id": r["id"],
                "golden": golden,
                "predicted": predicted,
                "confidence": conf,
                "correct": golden == predicted
            })
    
    return results, data.get("source", path)


def compute_auroc(results: list[dict]) -> float:
    """AUROC for confidence predicting correctness. Higher = better self-awareness."""
    correct = np.array([r["correct"] for r in results])
    confs = np.array([r["confidence"] for r in results])
    
    n_pos = np.sum(correct)
    n_neg = len(correct) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Sort by confidence descending
    sorted_idx = np.argsort(-confs)
    sorted_correct = correct[sorted_idx]
    
    # Count inversions (Mann-Whitney U statistic)
    rank_sum = np.sum(np.where(sorted_correct)[0])
    u_stat = rank_sum - n_pos * (n_pos - 1) / 2
    auroc = 1 - (u_stat / (n_pos * n_neg))
    
    return float(auroc)


def compute_brier_score(results: list[dict]) -> float:
    """Brier score - lower is better. Measures both calibration and sharpness."""
    scores = []
    for r in results:
        target = 1.0 if r["correct"] else 0.0
        scores.append((r["confidence"] - target) ** 2)
    
    return float(np.mean(scores))


def compute_conf_when_wrong(results: list[dict]) -> float:
    """Mean confidence when prediction is incorrect."""
    wrong_confs = [r["confidence"] for r in results if not r["correct"]]
    if not wrong_confs:
        return 0.0
    return float(np.mean(wrong_confs))


def statistical_test_conf_when_wrong(results1: list[dict], results2: list[dict]) -> dict:
    """Test if confidence when wrong is significantly different."""
    from scipy import stats
    
    wrong1 = [r["confidence"] for r in results1 if not r["correct"]]
    wrong2 = [r["confidence"] for r in results2 if not r["correct"]]
    
    if len(wrong1) > 1 and len(wrong2) > 1:
        u_stat, p_value = stats.mannwhitneyu(wrong1, wrong2, alternative='two-sided')
    else:
        u_stat, p_value = 0, 1.0
    
    return {
        "baseline_conf_when_wrong": float(np.mean(wrong1)) if wrong1 else 0.0,
        "grpo_conf_when_wrong": float(np.mean(wrong2)) if wrong2 else 0.0,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }


def compare_models(path1: str, path2: str) -> dict:
    results1, name1 = load_results(path1)
    results2, name2 = load_results(path2)
    
    print(f"\n{'='*60}")
    print(f"SELF-AWARENESS ANALYSIS")
    print(f"{'='*60}")
    print(f"\nBaseline: {name1} ({len(results1)} examples)")
    print(f"GRPO:     {name2} ({len(results2)} examples)")
    
    # Accuracy for reference
    acc1 = np.mean([r["correct"] for r in results1])
    acc2 = np.mean([r["correct"] for r in results2])
    
    # Compute the 3 key metrics
    auroc1 = compute_auroc(results1)
    auroc2 = compute_auroc(results2)
    
    brier1 = compute_brier_score(results1)
    brier2 = compute_brier_score(results2)
    
    conf_wrong1 = compute_conf_when_wrong(results1)
    conf_wrong2 = compute_conf_when_wrong(results2)
    
    # Statistical test for conf when wrong
    test = statistical_test_conf_when_wrong(results1, results2)
    
    # Print results
    print(f"\n{'-'*60}")
    print("ACCURACY (for reference)")
    print(f"{'-'*60}")
    print(f"   Baseline: {acc1*100:.1f}%")
    print(f"   GRPO:     {acc2*100:.1f}%")
    
    print(f"\n{'-'*60}")
    print("1. AUROC (Confidence → Correctness)")
    print(f"   Higher = better (can confidence predict correctness?)")
    print(f"{'-'*60}")
    print(f"   Baseline: {auroc1:.4f}")
    print(f"   GRPO:     {auroc2:.4f}")
    auroc_winner = "GRPO" if auroc2 > auroc1 else "Baseline"
    auroc_diff = abs(auroc2 - auroc1)
    print(f"   Winner:   {auroc_winner} (+{auroc_diff:.4f})")
    
    print(f"\n{'-'*60}")
    print("2. BRIER SCORE")
    print(f"   Lower = better (measures calibration + sharpness)")
    print(f"{'-'*60}")
    print(f"   Baseline: {brier1:.4f}")
    print(f"   GRPO:     {brier2:.4f}")
    brier_winner = "GRPO" if brier2 < brier1 else "Baseline"
    brier_diff = abs(brier1 - brier2)
    print(f"   Winner:   {brier_winner} (-{brier_diff:.4f})")
    
    print(f"\n{'-'*60}")
    print("3. CONFIDENCE WHEN WRONG")
    print(f"   Lower = better (model should be uncertain when incorrect)")
    print(f"{'-'*60}")
    print(f"   Baseline: {conf_wrong1:.4f}")
    print(f"   GRPO:     {conf_wrong2:.4f}")
    conf_winner = "GRPO" if conf_wrong2 < conf_wrong1 else "Baseline"
    conf_diff = abs(conf_wrong1 - conf_wrong2)
    print(f"   Winner:   {conf_winner} (-{conf_diff:.4f})")
    print(f"   p-value:  {test['p_value']:.4f} {'(significant)' if test['significant'] else '(not significant)'}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    metrics = [
        ("AUROC", auroc2 > auroc1),
        ("Brier Score", brier2 < brier1),
        ("Conf When Wrong", conf_wrong2 < conf_wrong1),
    ]
    
    wins_grpo = sum(1 for _, grpo_wins in metrics if grpo_wins)
    wins_base = 3 - wins_grpo
    
    print(f"\n   {'Metric':<20} {'Winner':>10}")
    print(f"   {'-'*30}")
    for name, grpo_wins in metrics:
        winner = "GRPO" if grpo_wins else "Baseline"
        print(f"   {name:<20} {winner:>10}")
    
    print(f"\n   GRPO: {wins_grpo}/3  |  Baseline: {wins_base}/3")
    
    if wins_grpo > wins_base:
        print(f"\n   → GRPO shows better self-awareness")
    elif wins_base > wins_grpo:
        print(f"\n   → Baseline shows better self-awareness")
    else:
        print(f"\n   → Models are similar in self-awareness")
    
    print()
    
    return {
        "baseline": {
            "name": name1, "n": len(results1), "accuracy": float(acc1),
            "auroc": auroc1, "brier": brier1, "conf_when_wrong": conf_wrong1
        },
        "grpo": {
            "name": name2, "n": len(results2), "accuracy": float(acc2),
            "auroc": auroc2, "brier": brier2, "conf_when_wrong": conf_wrong2
        },
        "test_conf_when_wrong": test
    }


def main():
    try:
        from scipy import stats
    except ImportError:
        print("ERROR: pip install scipy")
        sys.exit(1)
    
    if len(sys.argv) == 3:
        result = compare_models(sys.argv[1], sys.argv[2])
        with open("calibration_analysis.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Saved detailed results to calibration_analysis.json")
    else:
        import os
        pairs = [
            ("fever_control_results.json", "fever_grpo_results.json"),
            ("vitaminc_control_results.json", "vitaminc_grpo_results.json"),
        ]
        for p1, p2 in pairs:
            if os.path.exists(p1) and os.path.exists(p2):
                compare_models(p1, p2)
                break
        else:
            print("Usage: python analyze_calibration.py <baseline.json> <grpo.json>")


if __name__ == "__main__":
    main()
