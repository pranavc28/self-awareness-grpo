"""Calibration and self-awareness analysis for GRPO vs baseline models.

Key metrics for self-awareness:
1. ECE (Expected Calibration Error) - Does confidence match accuracy?
2. Confidence separation - Is model more confident when correct?
3. Selective prediction - Does rejecting low-confidence improve accuracy?
4. Brier Score - Proper scoring rule for probabilistic predictions
5. AUROC - Can confidence predict correctness?
"""

import json
import sys
import re
import numpy as np
from collections import defaultdict
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


def compute_ece(results: list[dict], n_bins: int = 10) -> tuple[float, list]:
    """Expected Calibration Error - lower is better."""
    bins = [[] for _ in range(n_bins)]
    
    for r in results:
        bin_idx = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[bin_idx].append(r)
    
    ece = 0.0
    bin_stats = []
    n_total = len(results)
    
    for i, bin_results in enumerate(bins):
        if len(bin_results) == 0:
            bin_stats.append({"count": 0, "avg_conf": None, "accuracy": None})
            continue
        
        avg_conf = np.mean([r["confidence"] for r in bin_results])
        accuracy = np.mean([r["correct"] for r in bin_results])
        count = len(bin_results)
        
        ece += (count / n_total) * abs(accuracy - avg_conf)
        bin_stats.append({
            "bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}",
            "count": count,
            "avg_conf": float(avg_conf),
            "accuracy": float(accuracy),
            "gap": float(accuracy - avg_conf)
        })
    
    return float(ece), bin_stats


def compute_confidence_separation(results: list[dict]) -> dict:
    """Compare confidence when correct vs incorrect."""
    correct_confs = [r["confidence"] for r in results if r["correct"]]
    wrong_confs = [r["confidence"] for r in results if not r["correct"]]
    
    if not wrong_confs:
        return {"error": "No incorrect predictions"}
    
    return {
        "n_correct": len(correct_confs),
        "n_wrong": len(wrong_confs),
        "mean_conf_correct": float(np.mean(correct_confs)),
        "mean_conf_wrong": float(np.mean(wrong_confs)),
        "median_conf_correct": float(np.median(correct_confs)),
        "median_conf_wrong": float(np.median(wrong_confs)),
        "std_conf_correct": float(np.std(correct_confs)),
        "std_conf_wrong": float(np.std(wrong_confs)),
        "separation": float(np.mean(correct_confs) - np.mean(wrong_confs)),
        # Effect size (Cohen's d)
        "cohens_d": float((np.mean(correct_confs) - np.mean(wrong_confs)) / 
                         np.sqrt((np.var(correct_confs) + np.var(wrong_confs)) / 2))
    }


def compute_auroc(results: list[dict]) -> float:
    """AUROC for confidence predicting correctness. Higher = better self-awareness."""
    from scipy import stats
    
    correct = np.array([r["correct"] for r in results])
    confs = np.array([r["confidence"] for r in results])
    
    # Rank-based AUROC
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


def compute_selective_accuracy(results: list[dict], thresholds: list[float] = None) -> list[dict]:
    """Accuracy when rejecting predictions below confidence threshold."""
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    stats = []
    base_acc = np.mean([r["correct"] for r in results])
    
    for thresh in thresholds:
        kept = [r for r in results if r["confidence"] >= thresh]
        if len(kept) == 0:
            continue
        
        acc = np.mean([r["correct"] for r in kept])
        coverage = len(kept) / len(results)
        
        stats.append({
            "threshold": thresh,
            "accuracy": float(acc),
            "coverage": float(coverage),
            "n_kept": len(kept),
            "accuracy_gain": float(acc - base_acc)
        })
    
    return stats


def compute_brier_score(results: list[dict]) -> float:
    """Brier score - lower is better. Measures both calibration and sharpness."""
    scores = []
    for r in results:
        # For multi-class, we compute average over classes
        # Simplified: just check if confident prediction was correct
        target = 1.0 if r["correct"] else 0.0
        scores.append((r["confidence"] - target) ** 2)
    
    return float(np.mean(scores))


def compute_confidence_distribution(results: list[dict]) -> dict:
    """Statistics on confidence distribution."""
    confs = [r["confidence"] for r in results]
    
    return {
        "mean": float(np.mean(confs)),
        "median": float(np.median(confs)),
        "std": float(np.std(confs)),
        "min": float(np.min(confs)),
        "max": float(np.max(confs)),
        "p10": float(np.percentile(confs, 10)),
        "p25": float(np.percentile(confs, 25)),
        "p75": float(np.percentile(confs, 75)),
        "p90": float(np.percentile(confs, 90)),
        "pct_above_90": float(np.mean(np.array(confs) > 0.9) * 100),
        "pct_above_95": float(np.mean(np.array(confs) > 0.95) * 100),
        "pct_above_99": float(np.mean(np.array(confs) > 0.99) * 100),
    }


def statistical_test_separation(results1: list[dict], results2: list[dict]) -> dict:
    """Test if separation (conf_correct - conf_wrong) is significantly different."""
    from scipy import stats
    
    def get_separation_per_example(results):
        # For bootstrap: separation for each correct/incorrect pair
        correct_confs = [r["confidence"] for r in results if r["correct"]]
        wrong_confs = [r["confidence"] for r in results if not r["correct"]]
        return np.mean(correct_confs), np.mean(wrong_confs)
    
    sep1 = compute_confidence_separation(results1)
    sep2 = compute_confidence_separation(results2)
    
    # Mann-Whitney U test on confidence when wrong
    wrong1 = [r["confidence"] for r in results1 if not r["correct"]]
    wrong2 = [r["confidence"] for r in results2 if not r["correct"]]
    
    if len(wrong1) > 1 and len(wrong2) > 1:
        u_stat, p_value = stats.mannwhitneyu(wrong1, wrong2, alternative='two-sided')
    else:
        u_stat, p_value = 0, 1.0
    
    return {
        "baseline_separation": sep1["separation"],
        "grpo_separation": sep2["separation"],
        "baseline_conf_when_wrong": sep1["mean_conf_wrong"],
        "grpo_conf_when_wrong": sep2["mean_conf_wrong"],
        "p_value_conf_wrong": float(p_value),
        "grpo_less_confident_when_wrong": sep2["mean_conf_wrong"] < sep1["mean_conf_wrong"]
    }


def compare_models(path1: str, path2: str) -> dict:
    results1, name1 = load_results(path1)
    results2, name2 = load_results(path2)
    
    print(f"\n{'='*80}")
    print(f"SELF-AWARENESS / CALIBRATION ANALYSIS: {name1} vs {name2}")
    print(f"{'='*80}")
    print(f"\nLoaded: {name1}={len(results1)} examples, {name2}={len(results2)} examples")
    
    # Basic stats
    acc1 = np.mean([r["correct"] for r in results1])
    acc2 = np.mean([r["correct"] for r in results2])
    
    # Compute all metrics
    ece1, bins1 = compute_ece(results1)
    ece2, bins2 = compute_ece(results2)
    
    sep1 = compute_confidence_separation(results1)
    sep2 = compute_confidence_separation(results2)
    
    auroc1 = compute_auroc(results1)
    auroc2 = compute_auroc(results2)
    
    brier1 = compute_brier_score(results1)
    brier2 = compute_brier_score(results2)
    
    dist1 = compute_confidence_distribution(results1)
    dist2 = compute_confidence_distribution(results2)
    
    sel1 = compute_selective_accuracy(results1)
    sel2 = compute_selective_accuracy(results2)
    
    # Print results
    print(f"\n{'-'*80}")
    print("1. ACCURACY (for reference)")
    print(f"{'-'*80}")
    print(f"   {name1}: {acc1*100:.1f}%")
    print(f"   {name2}: {acc2*100:.1f}%")
    
    print(f"\n{'-'*80}")
    print("2. CALIBRATION (ECE) - Lower is better")
    print(f"{'-'*80}")
    print(f"   {name1}: ECE = {ece1:.4f}")
    print(f"   {name2}: ECE = {ece2:.4f}")
    ece_winner = "GRPO" if ece2 < ece1 else "Baseline" if ece1 < ece2 else "Tie"
    print(f"   Winner: {ece_winner} (improvement: {(ece1-ece2)*100:.2f}% absolute)")
    
    print(f"\n   Calibration by confidence bin (GRPO):")
    print(f"   {'Bin':<12} {'Count':>8} {'Avg Conf':>10} {'Accuracy':>10} {'Gap':>10}")
    for b in bins2:
        if b["count"] > 0:
            gap_str = f"{b['gap']:+.3f}"
            print(f"   {b['bin']:<12} {b['count']:>8} {b['avg_conf']:>10.3f} {b['accuracy']:>10.3f} {gap_str:>10}")
    
    print(f"\n{'-'*80}")
    print("3. CONFIDENCE SEPARATION - Does model know when it's wrong?")
    print(f"{'-'*80}")
    print(f"   A self-aware model should be LESS confident when WRONG.")
    print()
    print(f"   {'Metric':<30} {'Baseline':>12} {'GRPO':>12} {'Better?':>10}")
    print(f"   {'-'*64}")
    print(f"   {'Mean conf (correct)':<30} {sep1['mean_conf_correct']:>12.4f} {sep2['mean_conf_correct']:>12.4f}")
    print(f"   {'Mean conf (wrong)':<30} {sep1['mean_conf_wrong']:>12.4f} {sep2['mean_conf_wrong']:>12.4f} {'GRPO' if sep2['mean_conf_wrong'] < sep1['mean_conf_wrong'] else 'Base'}")
    print(f"   {'Separation (higher=better)':<30} {sep1['separation']:>12.4f} {sep2['separation']:>12.4f} {'GRPO' if sep2['separation'] > sep1['separation'] else 'Base'}")
    print(f"   {'Cohens d (effect size)':<30} {sep1['cohens_d']:>12.4f} {sep2['cohens_d']:>12.4f} {'GRPO' if sep2['cohens_d'] > sep1['cohens_d'] else 'Base'}")
    
    print(f"\n{'-'*80}")
    print("4. AUROC (Confidence -> Correctness) - Higher is better")
    print(f"{'-'*80}")
    print(f"   Can confidence predict if the answer is correct?")
    print(f"   {name1}: AUROC = {auroc1:.4f}")
    print(f"   {name2}: AUROC = {auroc2:.4f}")
    auroc_winner = "GRPO" if auroc2 > auroc1 else "Baseline" if auroc1 > auroc2 else "Tie"
    print(f"   Winner: {auroc_winner}")
    
    print(f"\n{'-'*80}")
    print("5. BRIER SCORE - Lower is better")
    print(f"{'-'*80}")
    print(f"   {name1}: Brier = {brier1:.4f}")
    print(f"   {name2}: Brier = {brier2:.4f}")
    brier_winner = "GRPO" if brier2 < brier1 else "Baseline" if brier1 < brier2 else "Tie"
    print(f"   Winner: {brier_winner}")
    
    print(f"\n{'-'*80}")
    print("6. CONFIDENCE DISTRIBUTION")
    print(f"{'-'*80}")
    print(f"   {'Statistic':<20} {'Baseline':>12} {'GRPO':>12}")
    print(f"   {'-'*44}")
    print(f"   {'Mean':<20} {dist1['mean']:>12.4f} {dist2['mean']:>12.4f}")
    print(f"   {'Median':<20} {dist1['median']:>12.4f} {dist2['median']:>12.4f}")
    print(f"   {'Std Dev':<20} {dist1['std']:>12.4f} {dist2['std']:>12.4f}")
    print(f"   {'% > 0.90':<20} {dist1['pct_above_90']:>11.1f}% {dist2['pct_above_90']:>11.1f}%")
    print(f"   {'% > 0.95':<20} {dist1['pct_above_95']:>11.1f}% {dist2['pct_above_95']:>11.1f}%")
    print(f"   {'% > 0.99':<20} {dist1['pct_above_99']:>11.1f}% {dist2['pct_above_99']:>11.1f}%")
    
    print(f"\n{'-'*80}")
    print("7. SELECTIVE PREDICTION (reject low confidence)")
    print(f"{'-'*80}")
    print(f"   Accuracy gain when only keeping high-confidence predictions:")
    print(f"   {'Threshold':<12} {'Base Acc':>10} {'GRPO Acc':>10} {'Base Cov':>10} {'GRPO Cov':>10}")
    print(f"   {'-'*52}")
    for s1, s2 in zip(sel1, sel2):
        print(f"   >={s1['threshold']:<11} {s1['accuracy']*100:>9.1f}% {s2['accuracy']*100:>9.1f}% {s1['coverage']*100:>9.1f}% {s2['coverage']*100:>9.1f}%")
    
    # Statistical test
    test = statistical_test_separation(results1, results2)
    
    print(f"\n{'-'*80}")
    print("8. STATISTICAL TEST: Confidence when wrong")
    print(f"{'-'*80}")
    print(f"   Baseline mean conf when wrong: {test['baseline_conf_when_wrong']:.4f}")
    print(f"   GRPO mean conf when wrong:     {test['grpo_conf_when_wrong']:.4f}")
    print(f"   Mann-Whitney p-value:          {test['p_value_conf_wrong']:.4f}")
    if test['p_value_conf_wrong'] < 0.05:
        print(f"   Significant difference in confidence when wrong (p < 0.05)")
    else:
        print(f"   No significant difference in confidence when wrong")
    
    # Summary
    print(f"\n{'='*80}")
    print("SELF-AWARENESS SUMMARY")
    print(f"{'='*80}")
    
    wins_grpo = 0
    wins_base = 0
    
    metrics = [
        ("ECE (calibration)", ece2 < ece1),
        ("Confidence separation", sep2['separation'] > sep1['separation']),
        ("AUROC (conf->correct)", auroc2 > auroc1),
        ("Brier score", brier2 < brier1),
        ("Lower conf when wrong", sep2['mean_conf_wrong'] < sep1['mean_conf_wrong']),
    ]
    
    print("\n   Metric                         Winner")
    print("   " + "-"*45)
    for name, grpo_wins in metrics:
        winner = "GRPO" if grpo_wins else "Baseline"
        print(f"   {name:<30}   {winner}")
        if grpo_wins:
            wins_grpo += 1
        else:
            wins_base += 1
    
    print(f"\n   GRPO wins: {wins_grpo}/5, Baseline wins: {wins_base}/5")
    
    if wins_grpo > wins_base:
        print(f"\n   GRPO shows better self-awareness on most metrics!")
    elif wins_base > wins_grpo:
        print(f"\n   Baseline shows better self-awareness on most metrics.")
    else:
        print(f"\n   Models are similar in self-awareness.")
    
    print()
    
    return {
        "baseline": {
            "name": name1, "n": len(results1), "accuracy": acc1,
            "ece": ece1, "auroc": auroc1, "brier": brier1,
            "separation": sep1, "distribution": dist1
        },
        "grpo": {
            "name": name2, "n": len(results2), "accuracy": acc2,
            "ece": ece2, "auroc": auroc2, "brier": brier2,
            "separation": sep2, "distribution": dist2
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

