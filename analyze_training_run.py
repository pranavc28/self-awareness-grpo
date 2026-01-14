"""Comprehensive analysis of GRPO training metrics.

This script provides a detailed, readable summary of training progression,
identifying issues with reward components and suggesting improvements.
"""
import json
import numpy as np
from collections import defaultdict

def load_metrics(filepath="training_metrics.json"):
    """Load training metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def compute_stats(values):
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "trend": 0}
    arr = np.array(values)
    # Trend: slope of linear regression (positive = improving, negative = degrading)
    if len(arr) > 1:
        x = np.arange(len(arr))
        trend = np.polyfit(x, arr, 1)[0]
    else:
        trend = 0
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "trend": float(trend)
    }

def analyze_training(metrics):
    """Perform comprehensive analysis of training metrics."""
    
    # Extract time series for each metric
    steps = [m["step"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]
    
    # Cumulative reward components
    cum_components = defaultdict(list)
    for m in metrics:
        for comp, val in m["cumulative"].items():
            cum_components[comp].append(val)
    
    # By-outcome reward components
    outcome_components = {
        label: defaultdict(list)
        for label in ["NA", "PASS", "FAIL"]
    }
    for m in metrics:
        for label in ["NA", "PASS", "FAIL"]:
            for comp, val in m["by_outcome"][label].items():
                outcome_components[label][comp].append(val)
    
    # Divide into phases
    n = len(metrics)
    phase_size = n // 4
    phases = {
        "early": (0, phase_size),
        "mid_early": (phase_size, 2*phase_size),
        "mid_late": (2*phase_size, 3*phase_size),
        "late": (3*phase_size, n)
    }
    
    return {
        "total_steps": n,
        "accuracy": compute_stats(accuracy),
        "cumulative_components": {
            comp: compute_stats(vals)
            for comp, vals in cum_components.items()
        },
        "outcome_components": {
            label: {
                comp: compute_stats(vals)
                for comp, vals in comps.items()
            }
            for label, comps in outcome_components.items()
        },
        "phases": {
            phase: {
                "accuracy": compute_stats(accuracy[start:end]),
                "total_reward": compute_stats(cum_components["total"][start:end])
            }
            for phase, (start, end) in phases.items()
        },
        "raw": {
            "accuracy": accuracy,
            "cumulative": dict(cum_components),
            "by_outcome": {
                label: dict(comps)
                for label, comps in outcome_components.items()
            }
        }
    }

def format_trend(trend):
    """Format trend value with direction indicator."""
    if trend > 0.001:
        return f"↑ {trend:.4f}"
    elif trend < -0.001:
        return f"↓ {trend:.4f}"
    else:
        return f"→ {trend:.4f}"

def generate_report(analysis):
    """Generate a human-readable report of training analysis."""
    
    report = []
    report.append("=" * 80)
    report.append("GRPO TRAINING ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Overall summary
    report.append("\n📊 OVERALL SUMMARY")
    report.append("-" * 40)
    report.append(f"Total training steps: {analysis['total_steps']}")
    acc = analysis['accuracy']
    report.append(f"Accuracy: {acc['mean']:.3f} ± {acc['std']:.3f} (range: {acc['min']:.3f} - {acc['max']:.3f})")
    report.append(f"Accuracy trend: {format_trend(acc['trend'])}")
    
    # Phase progression
    report.append("\n📈 TRAINING PROGRESSION BY PHASE")
    report.append("-" * 40)
    for phase, data in analysis['phases'].items():
        phase_acc = data['accuracy']
        phase_rew = data['total_reward']
        report.append(f"\n{phase.upper().replace('_', ' ')}:")
        report.append(f"  Accuracy: {phase_acc['mean']:.3f} ± {phase_acc['std']:.3f}")
        report.append(f"  Total Reward: {phase_rew['mean']:.3f} ± {phase_rew['std']:.3f}")
    
    # Reward component analysis
    report.append("\n🎯 REWARD COMPONENT ANALYSIS")
    report.append("-" * 40)
    
    cum = analysis['cumulative_components']
    component_order = ['r_correct', 'r_false_na', 'r_explore', 'r_format', 'total']
    
    report.append("\nCumulative (averaged across all labels):")
    for comp in component_order:
        if comp in cum:
            stats = cum[comp]
            status = "✅" if stats['mean'] >= 0 or comp in ['r_cls'] else "⚠️"
            if comp == 'r_false_na' and stats['mean'] < -0.3:
                status = "🔴"  # Critical issue
            elif comp == 'r_correct' and stats['mean'] < 1.0:
                status = "⚠️"
            report.append(f"  {status} {comp:20s}: {stats['mean']:+.4f} ± {stats['std']:.4f} | trend: {format_trend(stats['trend'])}")
    
    # Per-outcome breakdown
    report.append("\n📋 PER-LABEL BREAKDOWN")
    report.append("-" * 40)
    
    for label in ["NA", "PASS", "FAIL"]:
        report.append(f"\n{label}:")
        label_data = analysis['outcome_components'][label]
        for comp in component_order:
            if comp in label_data:
                stats = label_data[comp]
                report.append(f"  {comp:20s}: {stats['mean']:+.4f} ± {stats['std']:.4f}")
    
    # Key issues identification
    report.append("\n" + "=" * 80)
    report.append("🔍 KEY ISSUES IDENTIFIED")
    report.append("=" * 80)
    
    issues = []
    
    # Check accuracy stagnation
    if abs(acc['trend']) < 0.0005:
        issues.append({
            "severity": "HIGH",
            "issue": "Accuracy is not improving (trend ≈ 0)",
            "cause": "Learning signal may be too weak or reward components are canceling out",
            "recommendation": "Increase correct_pos/neg ratio, reduce auxiliary penalties"
        })
    
    # Check NA dominance
    na_correct = analysis['outcome_components']['NA']['r_correct']['mean']
    pass_correct = analysis['outcome_components']['PASS']['r_correct']['mean']
    fail_correct = analysis['outcome_components']['FAIL']['r_correct']['mean']
    
    if na_correct > 1.5 and (pass_correct < 0 or fail_correct < 0):
        issues.append({
            "severity": "HIGH",
            "issue": "Model heavily biased toward NA label",
            "cause": "NA is easiest to predict correctly, model defaults to it",
            "recommendation": "Reduce NA examples or increase false_na penalty weight"
        })
    
    # Check false_na penalty activation
    false_na = cum['r_false_na']['mean']
    if false_na < -0.4:
        issues.append({
            "severity": "MEDIUM",
            "issue": f"High false NA rate (avg penalty: {false_na:.3f})",
            "cause": "Model predicts NA when evidence exists for PASS/FAIL",
            "recommendation": "Consider increasing lambda_false_na or curriculum learning"
        })
    
    # Check r_cls impact
    r_cls = cum['r_cls']['mean']
    if r_cls < -0.15:
        issues.append({
            "severity": "MEDIUM",
            "issue": f"Poor calibration (r_cls = {r_cls:.3f})",
            "cause": "Model's probability distribution doesn't align with gold labels",
            "recommendation": "Increase alpha_logp to strengthen calibration signal"
        })
    
    # Check entropy collapse
    r_entropy = cum['r_entropy']['mean']
    if r_entropy < 0.003 and analysis['accuracy']['mean'] < 0.75:
        issues.append({
            "severity": "LOW",
            "issue": f"Low entropy bonus contribution ({r_entropy:.5f})",
            "cause": "Model may have collapsed to deterministic predictions too early",
            "recommendation": "Increase entropy_bonus coefficient for more exploration"
        })
    
    # Print issues
    for issue in issues:
        severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[issue["severity"]]
        report.append(f"\n{severity_icon} [{issue['severity']}] {issue['issue']}")
        report.append(f"   Cause: {issue['cause']}")
        report.append(f"   Fix: {issue['recommendation']}")
    
    # Recommendations summary
    report.append("\n" + "=" * 80)
    report.append("💡 RECOMMENDED PARAMETER CHANGES")
    report.append("=" * 80)
    
    recommendations = []
    
    if false_na < -0.4 or (pass_correct < 0 and fail_correct < 0):
        recommendations.append("1. REDUCE NA BIAS:")
        recommendations.append("   - Change dataset balance: reduce NA ratio (try 300 NA, 350 PASS, 350 FAIL)")
        recommendations.append("   - Increase lambda_false_na from 2.0 → 3.0")
        recommendations.append("")
    
    if abs(acc['trend']) < 0.0005:
        recommendations.append("2. STRENGTHEN LEARNING SIGNAL:")
        recommendations.append("   - Increase correct_pos from 2.0 → 3.0")
        recommendations.append("   - Increase correct_neg from 1.0 → 1.5 (harder penalty for mistakes)")
        recommendations.append("   - Reduce alpha_logp from 0.10 → 0.05 (less noisy signal)")
        recommendations.append("")
    
    if r_entropy < 0.003:
        recommendations.append("3. IMPROVE EXPLORATION:")
        recommendations.append("   - Increase entropy_bonus from 0.01 → 0.05")
        recommendations.append("   - Increase temperature from 0.2 → 0.3 initially")
        recommendations.append("")
    
    recommendations.append("4. SIMPLIFY REWARD FUNCTION (recommended for cleaner signal):")
    recommendations.append("   - Consider removing r_inconsistent (adds noise, rarely fires)")
    recommendations.append("   - Consider removing r_entropy (model already learns format quickly)")
    recommendations.append("   - Focus on: r_correct, r_false_na, r_format only")
    
    report.extend(recommendations)
    
    return "\n".join(report)

def main():
    """Main analysis function."""
    print("Loading training metrics...")
    metrics = load_metrics()
    
    print("Analyzing training run...")
    analysis = analyze_training(metrics)
    
    print("\n" + generate_report(analysis))
    
    # Save analysis to JSON
    analysis_output = {k: v for k, v in analysis.items() if k != 'raw'}
    with open("training_analysis_detailed.json", "w") as f:
        json.dump(analysis_output, f, indent=2)
    print("\n\nDetailed analysis saved to training_analysis_detailed.json")

if __name__ == "__main__":
    main()

