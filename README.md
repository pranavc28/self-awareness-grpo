# Self-Aware GRPO: Confidence-Calibrated Fact Verification

This project trains a language model to produce well-calibrated confidence scores for fact verification using Group Relative Policy Optimization (GRPO). The model learns to express high confidence when correct and low confidence when uncertain, going beyond simple accuracy optimization to achieve genuine self-awareness about its own predictions.

## Summary

This repository implements GRPO training for fact verification across three datasets: FEVER (Wikipedia claims), VitaminC (Wikipedia revisions), and ClimateFEVER (climate science claims). The training uses a multi-component reward function that jointly optimizes for classification accuracy and confidence calibration—penalizing overconfident wrong predictions and underconfident correct ones. Evaluation scripts compare GRPO-trained checkpoints against the base model using statistical significance tests (paired bootstrap) and calibration metrics (AUROC, Brier score, confidence variance). Results show significant improvements on in-domain data (+13.5% accuracy, +29% NA recall on FEVER) with calibration gains measured by increased confidence discrimination.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** This project requires access to the Tinker API for model training and inference.

## Data Preparation

Download and prepare the datasets:

```bash
# FEVER dataset (~6GB Wikipedia dump)
python extract_fever_dataset.py

# VitaminC dataset
python extract_vitaminc_dataset.py

# ClimateFEVER dataset
python extract_climatefever_dataset.py
```

## Training

Run GRPO training with the mixed dataset (recommended):

```bash
python train_grpo.py
```

This trains for 160 steps with:
- Batch size: 32 examples × 8 samples = 256 completions/step
- LoRA rank: 32
- Learning rate: 2e-6
- Warmup: 50 steps for calibration terms

Training outputs:
- `training_metrics.json` - Step-by-step reward components
- Checkpoints saved to Tinker storage

## Evaluation

### Sample from trained model

```bash
# Evaluate on FEVER
python sample_grpo.py fever <checkpoint_path>

# Evaluate on VitaminC
python sample_grpo.py vitaminc <checkpoint_path>

# Evaluate on ClimateFEVER
python sample_grpo.py climatefever <checkpoint_path>
```

### Sample from base model (control)

```bash
python sample_control.py fever
python sample_control.py vitaminc
python sample_control.py climatefever
```

### Statistical Analysis

```bash
# Calibration analysis (AUROC, Brier, Confidence Std)
python analyze_calibration.py <control_results.json> <grpo_results.json>

# Significance testing (paired bootstrap)
python calculate_significance.py <control_results.json> <grpo_results.json>
```

### Visualize Training

```bash
python plot_training_metrics.py
```

Generates: `accuracy.png`, `rewards_cumulative.png`, `rewards_by_outcome.png`, `component_contribution.png`

## Project Structure

```
├── train_grpo.py              # Main GRPO training loop
├── sample_grpo.py             # Evaluate GRPO checkpoint
├── sample_control.py          # Evaluate base model
├── analyze_calibration.py     # Calibration metrics
├── calculate_significance.py  # Statistical significance tests
├── plot_training_metrics.py   # Training visualization
├── extract_fever_dataset.py   # FEVER data loader
├── extract_vitaminc_dataset.py
├── extract_climatefever_dataset.py
├── fever dataset training/    # Results from FEVER-only training
├── mixed dataset training/    # Results from mixed training
└── requirements.txt
```

## Output Format

The model outputs structured predictions:

```
LABEL=PASS|FAIL|NA
CONF=0.XX
RATIONALE=
```

Where:
- `PASS` = Evidence supports the claim
- `FAIL` = Evidence contradicts the claim  
- `NA` = Insufficient evidence
- `CONF` = Confidence score (0.00-1.00)

