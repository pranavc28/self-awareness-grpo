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


def plot_cumulative_rewards(metrics: list[dict], output_path: str = "rewards_cumulative.png"):
    """Plot cumulative rewards per component over training steps."""
    steps = [m["step"] for m in metrics]
    components = ["r_cls", "r_conf", "r_false_na", "r_align", "r_format", "total"]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#1a1a2e"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot individual components
    for comp, color in zip(components[:-1], colors[:-1]):
        values = [m["cumulative"][comp] for m in metrics]
        ax1.plot(steps, values, label=comp, color=color, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Reward Value", fontsize=12)
    ax1.set_title("Reward Components Over Training (Cumulative)", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot total reward
    total_values = [m["cumulative"]["total"] for m in metrics]
    ax2.plot(steps, total_values, label="Total Reward", color=colors[-1], linewidth=2.5)
    ax2.fill_between(steps, total_values, alpha=0.3, color=colors[-1])
    
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
    """Plot rewards per component for each outcome (NA, PASS, FAIL)."""
    steps = [m["step"] for m in metrics]
    components = ["r_cls", "r_conf", "r_false_na", "r_align", "r_format", "total"]
    outcomes = ["NA", "PASS", "FAIL"]
    outcome_colors = {"NA": "#e74c3c", "PASS": "#2ecc71", "FAIL": "#3498db"}
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, comp in enumerate(components):
        ax = axes[idx]
        for outcome in outcomes:
            values = [m["by_outcome"][outcome][comp] for m in metrics]
            ax.plot(steps, values, label=outcome, color=outcome_colors[outcome], linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(f"{comp}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle("Reward Components by Outcome Over Training", fontsize=14, fontweight="bold", y=1.02)
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
    """Plot stacked area chart showing how each component contributes to total reward."""
    steps = [m["step"] for m in metrics]
    components = ["r_cls", "r_conf", "r_false_na", "r_align", "r_format"]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get component values
    data = {comp: [m["cumulative"][comp] for m in metrics] for comp in components}
    
    # For stacked area, we need to separate positive and negative contributions
    # Let's just plot lines with markers for clarity
    for comp, color in zip(components, colors):
        ax.plot(steps, data[comp], label=comp, color=color, linewidth=2, alpha=0.8)
    
    # Add total line
    total = [m["cumulative"]["total"] for m in metrics]
    ax.plot(steps, total, label="total", color="#1a1a2e", linewidth=3, linestyle="--")
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Reward Value", fontsize=12)
    ax.set_title("Reward Component Contributions Over Training", fontsize=14, fontweight="bold")
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
        for comp in ["r_cls", "r_conf", "r_false_na", "r_align", "r_format", "total"]:
            print(f"  {comp}: {last['cumulative'][comp]:.4f}")


if __name__ == "__main__":
    main()

