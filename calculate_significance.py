"""Statistical significance testing for FEVER classification results."""

import json
import sys
import numpy as np
from collections import defaultdict
from typing import Optional
import re

LABELS = ["PASS", "FAIL", "NA"]
LABEL_MAP = {"SUPPORTS": "PASS", "REFUTES": "FAIL", "NOT ENOUGH INFO": "NA",
             "PASS": "PASS", "FAIL": "FAIL", "NA": "NA"}


def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


def _validate_raw_response(text: str) -> tuple[str, float, bool]:
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
            conf = max(0.0, min(1.0, float(m.group(1))))
        except:
            valid = False
    else:
        valid = False
    return label, conf, valid


def is_valid_result(r: dict) -> bool:
    if "golden_label" not in r or "predicted_label" not in r:
        return False
    if r.get("format_valid") is False:
        return False
    golden = normalize_label(r.get("golden_label"))
    predicted = normalize_label(r.get("predicted_label"))
    if golden not in LABELS or predicted not in LABELS:
        return False
    if "format_valid" not in r and "raw_response" in r:
        _, _, valid = _validate_raw_response(r.get("raw_response", ""))
        if not valid:
            return False
    return True


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def filter_valid_results(results: list[dict]) -> tuple[list[dict], int]:
    valid = [r for r in results if is_valid_result(r)]
    return valid, len(results) - len(valid)


def calc_metrics(golden: np.ndarray, predicted: np.ndarray) -> dict:
    metrics = {}
    for label in LABELS:
        tp = np.sum((golden == label) & (predicted == label))
        fp = np.sum((golden != label) & (predicted == label))
        fn = np.sum((golden == label) & (predicted != label))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {"precision": prec, "recall": rec, "f1": f1}
    return metrics


def paired_bootstrap_test(results1, results2, metric="macro_f1", n_boot=10000, seed=42):
    np.random.seed(seed)
    id_to_r1 = {r["id"]: r for r in results1}
    id_to_r2 = {r["id"]: r for r in results2}
    common_ids = sorted(set(id_to_r1.keys()) & set(id_to_r2.keys()))
    n = len(common_ids)
    if n == 0:
        return {"error": "No common IDs"}
    
    golden = np.array([normalize_label(id_to_r1[id_]["golden_label"]) for id_ in common_ids])
    pred1 = np.array([normalize_label(id_to_r1[id_]["predicted_label"]) for id_ in common_ids])
    pred2 = np.array([normalize_label(id_to_r2[id_]["predicted_label"]) for id_ in common_ids])
    
    def compute(g, p):
        if metric == "accuracy":
            return np.mean(g == p)
        m = calc_metrics(g, p)
        if metric == "macro_f1":
            return np.mean([m[l]["f1"] for l in LABELS])
        if "_recall" in metric:
            return m[metric.replace("_recall", "")]["recall"]
        return m[metric.replace("_f1", "")]["f1"]
    
    obs_m1, obs_m2 = compute(golden, pred1), compute(golden, pred2)
    obs_diff = obs_m2 - obs_m1
    
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_diffs.append(compute(golden[idx], pred2[idx]) - compute(golden[idx], pred1[idx]))
    boot_diffs = np.array(boot_diffs)
    
    p_value = min(2 * np.mean(boot_diffs <= 0) if obs_diff >= 0 else 2 * np.mean(boot_diffs >= 0), 1.0)
    
    return {
        "metric": metric, "n_paired": n,
        "model1_value": float(obs_m1), "model2_value": float(obs_m2),
        "observed_diff": float(obs_diff),
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.01)
    }


def mcnemar_test(results1, results2):
    from scipy import stats
    id_to_r1 = {r["id"]: r for r in results1}
    id_to_r2 = {r["id"]: r for r in results2}
    common_ids = set(id_to_r1.keys()) & set(id_to_r2.keys())
    
    n01, n10 = 0, 0
    for id_ in common_ids:
        golden = normalize_label(id_to_r1[id_]["golden_label"])
        c1 = golden == normalize_label(id_to_r1[id_]["predicted_label"])
        c2 = golden == normalize_label(id_to_r2[id_]["predicted_label"])
        if not c1 and c2: n01 += 1
        elif c1 and not c2: n10 += 1
    
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    p_value = float(stats.chi2.sf(chi2, df=1)) if (n01 + n10) > 0 else 1.0
    
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_value": p_value}


def compare_two_models(path1: str, path2: str, output_path: Optional[str] = None) -> dict:
    data1, data2 = load_results(path1), load_results(path2)
    valid1, skip1 = filter_valid_results(data1["results"])
    valid2, skip2 = filter_valid_results(data2["results"])
    name1, name2 = data1.get("source", "Model1"), data2.get("source", "Model2")
    
    # Get paired valid results
    id_to_r1 = {r["id"]: r for r in valid1}
    id_to_r2 = {r["id"]: r for r in valid2}
    common_ids = sorted(set(id_to_r1.keys()) & set(id_to_r2.keys()))
    paired1 = [id_to_r1[id_] for id_ in common_ids]
    paired2 = [id_to_r2[id_] for id_ in common_ids]
    
    # Run tests
    mcnemar = mcnemar_test(paired1, paired2)
    tests = {m: paired_bootstrap_test(paired1, paired2, metric=m) 
             for m in ["macro_f1", "accuracy", "PASS_f1", "FAIL_f1", "NA_f1", "NA_recall"]}
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"STATISTICAL SIGNIFICANCE: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"\nFiltering: {name1}={len(valid1)} valid, {name2}={len(valid2)} valid, Paired={len(common_ids)}")
    
    print(f"\n### Summary: {'ALL' if all(t['significant'] for t in tests.values()) else 'SOME'} improvements statistically significant (p < 0.01)\n")
    print(f"{'Metric':<12} {'Baseline':>12} {'→':>3} {'GRPO':>12} {'Improvement':>14} {'p-value':>12} {'Sig?':>8}")
    print("-" * 75)
    
    for metric, t in tests.items():
        sig = "✅ Yes (***)" if t['significant'] else "❌ No"
        if metric == "accuracy":
            print(f"{metric:<12} {t['model1_value']*100:>11.1f}% → {t['model2_value']*100:>11.1f}% {t['observed_diff']*100:>+13.1f}% {t['p_value']:>12.4f} {sig:>12}")
        else:
            print(f"{metric:<12} {t['model1_value']:>12.4f} → {t['model2_value']:>12.4f} {t['observed_diff']:>+14.4f} {t['p_value']:>12.4f} {sig:>12}")
    
    print(f"\n### What This Means")
    print(f"• p-value < 0.01 means <1% chance improvement is random luck")
    print(f"• 95% CI for Macro F1 diff: [{tests['macro_f1']['ci_lower']:.3f}, {tests['macro_f1']['ci_upper']:.3f}] — doesn't include 0")
    
    print(f"\n### McNemar's Test (χ²={mcnemar['chi2']:.2f}, p={mcnemar['p_value']:.6f})")
    print(f"• {mcnemar['n01']} examples: baseline wrong → GRPO correct")
    print(f"• {mcnemar['n10']} examples: baseline correct → GRPO wrong")
    print(f"• Net: GRPO corrected {mcnemar['n01'] - mcnemar['n10']} more errors than it introduced")
    
    # Find biggest win
    best = max(tests.items(), key=lambda x: x[1]['observed_diff'])
    print(f"\n### Biggest Win: {best[0]} — improved by +{best[1]['observed_diff']:.3f}")
    
    print(f"\n### Bottom Line")
    if all(t['significant'] for t in tests.values()):
        print("The GRPO-trained model is genuinely better. All improvements are real (p < 0.01).")
    else:
        print("Some improvements may not be statistically significant.")
    print()
    
    output_data = {
        "filtering": {"valid1": len(valid1), "valid2": len(valid2), "paired": len(common_ids)},
        "mcnemar_test": mcnemar,
        "paired_bootstrap_tests": tests
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {output_path}")
    
    return output_data


def main():
    import os
    try:
        from scipy import stats
    except ImportError:
        print("ERROR: pip install scipy"); sys.exit(1)
    
    if len(sys.argv) == 3:
        compare_two_models(sys.argv[1], sys.argv[2], "significance_comparison.json")
    else:
        fever, grpo = "fever_classification_results.json", "grpo_sample_results.json"
        if os.path.exists(fever) and os.path.exists(grpo):
            compare_two_models(fever, grpo, "significance_comparison.json")
        else:
            print("Usage: python calculate_significance.py <baseline.json> <treatment.json>")


if __name__ == "__main__":
    main()