"""Plot training metrics from GRPO training.

Generates visualizations for:
1. Per-component rewards over training steps (cumulative)
2. Per-component rewards by outcome (NA, PASS, FAIL)
3. Accuracy over training steps
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(path: str = "training_metrics.json") -> list[dict]:
    """Load training metrics from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def smooth_values(values: list, window: int) -> tuple[list, list]:
    """Apply moving average smoothing to values.
    
    Returns:
        tuple: (smoothed_values, valid_indices) - smoothed data and corresponding step indices
    """
    if len(values) < window:
        return values, list(range(len(values)))
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # The valid indices start at window-1 (center of first window)
    start_idx = window // 2
    indices = list(range(start_idx, start_idx + len(smoothed)))
    return smoothed.tolist(), indices


def plot_cumulative_rewards(metrics: list[dict], output_path: str = "rewards_cumulative.png"):
    """Plot cumulative rewards per component over training steps.
    
    Shows both raw per-step values (with low opacity) and smoothed trend lines.
    """
    steps = [m["step"] for m in metrics]
    desired = ["r_correct", "r_cls", "r_cal", "r_conf", "r_false_na", "r_align", "r_inconsistent", "r_format", "r_entropy", "total"]
    # Backward compatible: only plot keys that exist in the metrics file.
    available = set(metrics[0].get("cumulative", {}).keys()) if metrics else set()
    components = [c for c in desired if c in available]
    palette = {
        "r_correct": "#27ae60",
        "r_cls": "#2ecc71",
        "r_cal": "#3498db",
        "r_conf": "#3498db",
        "r_false_na": "#e74c3c",
        "r_align": "#9b59b6",
        "r_inconsistent": "#34495e",
        "r_format": "#f39c12",
        "r_entropy": "#1abc9c",
        "total": "#1a1a2e",
    }
    
    # Calculate smoothing window (adaptive based on data size)
    window = min(20, max(5, len(metrics) // 8)) if len(metrics) > 10 else 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot individual components (everything except total, if present)
    comps_wo_total = [c for c in components if c != "total"]
    for comp in comps_wo_total:
        color = palette[comp]
        values = [m["cumulative"][comp] for m in metrics]
        
        # Plot raw values with low opacity
        ax1.plot(steps, values, color=color, linewidth=1, alpha=0.25)
        
        # Plot smoothed trend line
        if len(values) > window:
            smoothed, smooth_indices = smooth_values(values, window)
            smooth_steps = [steps[i] for i in smooth_indices if i < len(steps)]
            ax1.plot(smooth_steps[:len(smoothed)], smoothed, label=comp, color=color, linewidth=2.5)
        else:
            ax1.plot(steps, values, label=comp, color=color, linewidth=2.5)
    
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Reward Value", fontsize=12)
    ax1.set_title(f"Reward Components Over Training (Smoothed, window={window})", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot total reward (if present)
    if "total" in available:
        total_values = [m["cumulative"]["total"] for m in metrics]
        
        # Plot raw values with low opacity and fill
        ax2.plot(steps, total_values, color=palette["total"], linewidth=1, alpha=0.3)
        ax2.fill_between(steps, total_values, alpha=0.15, color=palette["total"])
        
        # Plot smoothed trend line
        if len(total_values) > window:
            smoothed, smooth_indices = smooth_values(total_values, window)
            smooth_steps = [steps[i] for i in smooth_indices if i < len(steps)]
            ax2.plot(smooth_steps[:len(smoothed)], smoothed, label=f"Smoothed (window={window})", 
                     color="#e74c3c", linewidth=2.5)
            ax2.legend(loc="best", fontsize=10)
    
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Total Reward", fontsize=12)
    ax2.set_title("Total Reward Over Training", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved cumulative rewards plot to {output_path}")
    plt.close()


def plot_rewards_by_outcome(metrics: list[dict], output_path: str = "rewards_by_outcome.png"):
    """Plot rewards per component for each outcome (NA, PASS, FAIL) with smoothing."""
    steps = [m["step"] for m in metrics]
    desired = ["r_correct", "r_cls", "r_cal", "r_conf", "r_false_na", "r_align", "r_inconsistent", "r_format", "r_entropy", "total"]
    available = set(metrics[0].get("by_outcome", {}).get("NA", {}).keys()) if metrics else set()
    components = [c for c in desired if c in available]
    outcomes = ["NA", "PASS", "FAIL"]
    outcome_colors = {"NA": "#e74c3c", "PASS": "#2ecc71", "FAIL": "#3498db"}
    
    # Calculate smoothing window
    window = min(20, max(5, len(metrics) // 8)) if len(metrics) > 10 else 1
    
    # Grid size depends on component count
    n = max(len(components), 1)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    
    for idx, comp in enumerate(components):
        ax = axes[idx]
        for outcome in outcomes:
            values = [m["by_outcome"][outcome][comp] for m in metrics]
            color = outcome_colors[outcome]
            
            # Plot raw with low opacity
            ax.plot(steps, values, color=color, linewidth=1, alpha=0.2)
            
            # Plot smoothed
            if len(values) > window:
                smoothed, smooth_indices = smooth_values(values, window)
                smooth_steps = [steps[i] for i in smooth_indices if i < len(steps)]
                ax.plot(smooth_steps[:len(smoothed)], smoothed, label=outcome, color=color, linewidth=2)
            else:
                ax.plot(steps, values, label=outcome, color=color, linewidth=2)
        
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(f"{comp}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Hide unused subplots
    for j in range(len(components), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Reward Components by Outcome (Smoothed, window={window})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved rewards by outcome plot to {output_path}")
    plt.close()


def plot_accuracy(metrics: list[dict], output_path: str = "accuracy.png"):
    """Plot accuracy over training steps."""
    steps = [m["step"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps, accuracy, color="#1a1a2e", linewidth=2.5)
    ax.fill_between(steps, accuracy, alpha=0.3, color="#1a1a2e")
    
    # Add smoothed trend line
    if len(accuracy) > 10:
        window = min(20, len(accuracy) // 5)
        smoothed = np.convolve(accuracy, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        ax.plot(smooth_steps, smoothed, color="#e74c3c", linewidth=2, linestyle="--", 
                label=f"Smoothed (window={window})")
        ax.legend(loc="best")
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy Over Training", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved accuracy plot to {output_path}")
    plt.close()


def plot_component_contribution(metrics: list[dict], output_path: str = "component_contribution.png"):
    """Plot smoothed component contributions showing how each component contributes to total reward."""
    steps = [m["step"] for m in metrics]
    desired = ["r_correct", "r_cls", "r_cal", "r_conf", "r_false_na", "r_align", "r_inconsistent", "r_format", "r_entropy"]
    available = set(metrics[0].get("cumulative", {}).keys()) if metrics else set()
    components = [c for c in desired if c in available]
    palette = {
        "r_correct": "#27ae60",
        "r_cls": "#2ecc71",
        "r_cal": "#3498db",
        "r_conf": "#3498db",
        "r_false_na": "#e74c3c",
        "r_align": "#9b59b6",
        "r_inconsistent": "#34495e",
        "r_format": "#f39c12",
        "r_entropy": "#1abc9c",
    }
    
    # Calculate smoothing window
    window = min(20, max(5, len(metrics) // 8)) if len(metrics) > 10 else 1
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get component values and plot with smoothing
    for comp in components:
        color = palette[comp]
        values = [m["cumulative"][comp] for m in metrics]
        
        # Plot raw with low opacity
        ax.plot(steps, values, color=color, linewidth=1, alpha=0.2)
        
        # Plot smoothed
        if len(values) > window:
            smoothed, smooth_indices = smooth_values(values, window)
            smooth_steps = [steps[i] for i in smooth_indices if i < len(steps)]
            ax.plot(smooth_steps[:len(smoothed)], smoothed, label=comp, color=color, linewidth=2.5)
        else:
            ax.plot(steps, values, label=comp, color=color, linewidth=2.5)
    
    # Add total line with smoothing
    total = [m["cumulative"]["total"] for m in metrics]
    ax.plot(steps, total, color="#1a1a2e", linewidth=1, alpha=0.2)
    if len(total) > window:
        smoothed, smooth_indices = smooth_values(total, window)
        smooth_steps = [steps[i] for i in smooth_indices if i < len(steps)]
        ax.plot(smooth_steps[:len(smoothed)], smoothed, label="total", color="#1a1a2e", linewidth=3, linestyle="--")
    else:
        ax.plot(steps, total, label="total", color="#1a1a2e", linewidth=3, linestyle="--")
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Reward Value", fontsize=12)
    ax.set_title(f"Reward Component Contributions (Smoothed, window={window})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved component contribution plot to {output_path}")
    plt.close()


def main():
    """Generate all plots from training metrics."""
    metrics_path = "training_metrics.json"
    
    if not Path(metrics_path).exists():
        print(f"Error: {metrics_path} not found. Run training first.")
        return
    
    print(f"Loading metrics from {metrics_path}...")
    metrics = load_metrics(metrics_path)
    print(f"Loaded {len(metrics)} training steps")
    
    # Generate all plots
    plot_cumulative_rewards(metrics)
    plot_rewards_by_outcome(metrics)
    plot_accuracy(metrics)
    plot_component_contribution(metrics)
    
    print("\nAll plots generated successfully!")
    
    # Print summary statistics
    if metrics:
        last = metrics[-1]
        first = metrics[0]
        print(f"\n=== Training Summary ===")
        print(f"Steps: {first['step']} → {last['step']}")
        print(f"Accuracy: {first['accuracy']:.3f} → {last['accuracy']:.3f}")
        print(f"\nFinal Cumulative Rewards:")
        for comp in ["r_correct", "r_cls", "r_cal", "r_conf", "r_false_na", "r_align", "r_inconsistent", "r_format", "r_entropy", "total"]:
            if comp in last.get("cumulative", {}):
                print(f"  {comp}: {last['cumulative'][comp]:.4f}")


if __name__ == "__main__":
    main()

