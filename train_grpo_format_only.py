"""GRPO training with FORMAT-ONLY reward for baseline comparison.

This trains a model using GRPO but with ONLY format compliance reward.
No accuracy reward, no class weights, no exploration bonus - just format.

This gives you a fair baseline to compare against your full GRPO reward function.
The model will learn to output clean labels (PASS/FAIL/NA) but won't be
optimized for correctness or calibration.

Comparison:
- Format-only baseline: Learns output format, random accuracy
- Full GRPO: Learns format + accuracy + calibration

Usage:
    python train_grpo_format_only.py
"""
import asyncio
import json
import random
import re
import numpy as np
from dataclasses import dataclass
from functools import wraps

import tinker
from tinker import types

from extract_fever_dataset import WikiReader, download_wiki_pages
from extract_vitaminc_dataset import download_vitaminc
from extract_climatefever_dataset import download_climatefever

MODEL = "openai/gpt-oss-20b"
SEED = 42
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 10.0


def async_retry(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
    """Decorator for async functions that retries on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), RETRY_MAX_DELAY)
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = delay * jitter
                        print(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


# Global WikiReader instance
_wiki_reader: WikiReader | None = None


def get_wiki_reader() -> WikiReader:
    """Get or create the global WikiReader instance."""
    global _wiki_reader
    if _wiki_reader is None:
        print("Initializing WikiReader...")
        wiki_dir = download_wiki_pages()
        _wiki_reader = WikiReader(wiki_dir)
    return _wiki_reader


@dataclass
class Config:
    """Configuration for FORMAT-ONLY GRPO training.
    
    This is a minimal config that ONLY rewards format compliance.
    """
    batch_size: int = 32
    group_size: int = 8
    max_tokens: int = 28000
    temperature: float = 0.3
    lr: float = 5e-6
    steps: int = 100  # Fewer steps needed for format-only
    weight_decay: float = 0.01
    
    # FORMAT-ONLY REWARD CONFIG
    # Only reward for correct format, nothing else
    format_reward: float = 1.0      # Reward for valid format
    format_penalty: float = -1.0    # Penalty for invalid format
    
    # KL regularization
    beta_kl: float = 0.02


def load_mixed_dataset(
    fever_na: int = 120, fever_pass: int = 120, fever_fail: int = 100,
    vitaminc_na: int = 120, vitaminc_pass: int = 120, vitaminc_fail: int = 100,
    climatefever_na: int = 120, climatefever_pass: int = 120, climatefever_fail: int = 100,
):
    """Load a mixed dataset from FEVER, VitaminC, and ClimateFEVER."""
    random.seed(SEED)
    sampled = []
    
    # Load FEVER examples
    print("Loading FEVER examples...")
    with open("fever_data/fever_dev.json", "r") as f:
        fever_examples = json.load(f)
    fever_by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in fever_examples:
        if ex.get("label") in fever_by_label:
            fever_by_label[ex["label"]].append(ex)
    
    for lbl, count in [("NOT ENOUGH INFO", fever_na), ("SUPPORTS", fever_pass), ("REFUTES", fever_fail)]:
        selected = random.sample(fever_by_label[lbl], min(count, len(fever_by_label[lbl])))
        for ex in selected:
            sampled.append({
                "prompt": ex["claim"],
                "label": LABEL_MAP[ex["label"]],
                "evidence": ex.get("evidence", []),
                "source": "fever"
            })
    print(f"  Added {fever_na + fever_pass + fever_fail} FEVER examples")
    
    # Load VitaminC examples
    print("Loading VitaminC examples...")
    vitaminc_examples = download_vitaminc("vitaminc_data", split="dev")
    vitaminc_by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in vitaminc_examples:
        if ex.get("label") in vitaminc_by_label:
            vitaminc_by_label[ex["label"]].append(ex)
    
    for lbl, count in [("NOT ENOUGH INFO", vitaminc_na), ("SUPPORTS", vitaminc_pass), ("REFUTES", vitaminc_fail)]:
        selected = random.sample(vitaminc_by_label[lbl], min(count, len(vitaminc_by_label[lbl])))
        for ex in selected:
            sampled.append({
                "prompt": ex["claim"],
                "label": LABEL_MAP[ex["label"]],
                "evidence": ex.get("evidence", ""),
                "source": "vitaminc"
            })
    print(f"  Added {vitaminc_na + vitaminc_pass + vitaminc_fail} VitaminC examples")
    
    # Load ClimateFEVER examples
    print("Loading ClimateFEVER examples...")
    climatefever_examples = download_climatefever("climatefever_data")
    climatefever_by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in climatefever_examples:
        if ex.get("label") in climatefever_by_label:
            climatefever_by_label[ex["label"]].append(ex)
    
    for lbl, count in [("NOT ENOUGH INFO", climatefever_na), ("SUPPORTS", climatefever_pass), ("REFUTES", climatefever_fail)]:
        available = len(climatefever_by_label[lbl])
        actual_count = min(count, available)
        if actual_count < count:
            print(f"  Warning: Only {available} {lbl} examples available in ClimateFEVER, using all")
        selected = random.sample(climatefever_by_label[lbl], actual_count)
        for ex in selected:
            sampled.append({
                "prompt": ex["claim"],
                "label": LABEL_MAP[ex["label"]],
                "evidence": ex.get("evidence", ""),
                "source": "climatefever"
            })
    print(f"  Added {climatefever_na + climatefever_pass + climatefever_fail} ClimateFEVER examples")
    
    random.shuffle(sampled)
    
    print(f"\nTotal mixed dataset: {len(sampled)} examples")
    print(f"  NA: {sum(1 for ex in sampled if ex['label'] == 'NA')}")
    print(f"  PASS: {sum(1 for ex in sampled if ex['label'] == 'PASS')}")
    print(f"  FAIL: {sum(1 for ex in sampled if ex['label'] == 'FAIL')}")
    
    return sampled


def build_prompt(example):
    """Build the same prompt format as the full GRPO training."""
    wiki_reader = get_wiki_reader()
    evidence_texts = wiki_reader.get_evidence_text(example.get("evidence", []))
    
    if evidence_texts:
        evidence = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence = "No specific evidence provided."
    
    return f"""Fact verification task. Classify claims based on evidence.

Labels:
- PASS = Evidence supports the claim
- FAIL = Evidence contradicts the claim
- NA = Insufficient or unclear evidence

Examples:
Claim: The sky is blue.
Evidence: The sky appears blue due to Rayleigh scattering.
LABEL=PASS

Claim: Water boils at 50°C.
Evidence: Water boils at 100°C at sea level.
LABEL=FAIL

Claim: The Earth is flat.
Evidence: Earth is an oblate spheroid with a circumference of about 40,075 km.
LABEL=FAIL

Claim: Humans have 206 bones.
Evidence: Adult humans typically have 206 bones in their body.
LABEL=PASS

Claim: Aliens built the pyramids.
Evidence: The Great Pyramid was built around 2560 BC.
LABEL=NA

Now classify this claim:

Claim: {example['prompt']}

Evidence:
{evidence}

Think: Does the evidence clearly support or contradict this specific claim? If unclear and confident in being unclear, say NA.
LABEL="""


def parse_output(text):
    """Parse model output to extract label and format validity."""
    text = re.sub(r'<\|[^|]*\|>', '', text).strip()
    text_upper = text.upper()
    
    # Check for exact LABEL=X format (most desirable)
    label_exact = re.match(r'^LABEL\s*=\s*(NA|PASS|FAIL)\s*$', text_upper)
    if label_exact:
        return label_exact.group(1), True
    
    # Check for exact bare label (also acceptable)
    if text_upper in ("NA", "PASS", "FAIL"):
        return text_upper, True
    
    # Fallback: find LABEL=X anywhere in text (penalize)
    label_match = re.search(r'LABEL\s*=\s*(NA|PASS|FAIL)', text_upper)
    if label_match:
        return label_match.group(1), False
    
    # Last resort: find bare label anywhere
    bare_match = re.search(r'\b(NA|PASS|FAIL)\b', text_upper)
    if bare_match:
        return bare_match.group(1), False
    
    return None, False


def compute_format_only_reward(pred_label: str | None, format_valid: bool, cfg: Config):
    """
    FORMAT-ONLY reward function.
    
    This ONLY rewards format compliance. No accuracy, no class weights, nothing else.
    The model learns to output clean labels but not which label is correct.
    """
    # If format is valid, give reward. Otherwise, penalty.
    if format_valid and pred_label is not None:
        reward = cfg.format_reward
    else:
        reward = cfg.format_penalty
    
    return {
        "total": reward,
        "r_format": reward,
    }


@async_retry()
async def sample_completions(sampling_client, tokenizer, prompt_text, cfg: Config):
    """Sample multiple completions from the model for GRPO advantage estimation."""
    tokens = tokenizer.encode(prompt_text)
    model_input = types.ModelInput.from_ints(tokens)
    params = types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=["\n"]
    )
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=cfg.group_size,
        sampling_params=params
    )
    samples = []
    for seq in result.sequences:
        text = tokenizer.decode(seq.tokens)
        token_lps = seq.logprobs if hasattr(seq, "logprobs") else []
        if token_lps is None:
            token_lps = []
        if hasattr(token_lps, "tolist"):
            token_lps = token_lps.tolist()
        samples.append({"text": text, "tokens": seq.tokens, "token_logprobs": token_lps})
    return samples


async def run_training():
    """
    FORMAT-ONLY GRPO training loop.
    
    This trains the model to output clean labels (PASS/FAIL/NA) but does NOT
    optimize for correctness. Use this as a baseline to compare against
    your full GRPO reward function.
    """
    cfg = Config()
    print("=" * 60)
    print("FORMAT-ONLY GRPO BASELINE TRAINING")
    print("=" * 60)
    print("This trains ONLY for format compliance, NOT accuracy.")
    print("Use this checkpoint as a baseline to compare against full GRPO.")
    print("=" * 60)
    
    print(f"\nLoading dataset...")
    dataset = load_mixed_dataset()
    print(f"Loaded {len(dataset)} examples")
    
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="format_only_init")
    
    print("\nRunning sanity check...")
    test_batch = dataset[:5]
    sanity_tasks = [
        sample_completions(sampling_client, tokenizer, build_prompt(ex), cfg)
        for ex in test_batch
    ]
    sanity_results = await asyncio.gather(*sanity_tasks)
    for ex, samples in zip(test_batch, sanity_results):
        for s in samples[:2]:
            print(f"RAW: {repr(s['text'])}")
            label, valid = parse_output(s["text"])
            print(f"  Gold={ex['label']} Pred={label} Valid={valid}")
    
    metrics_history = []
    
    print(f"\nStarting FORMAT-ONLY GRPO training for {cfg.steps} steps...")
    print(f"Config: format_reward={cfg.format_reward}, format_penalty={cfg.format_penalty}")
    print(f"        temp={cfg.temperature}, lr={cfg.lr}\n")
    
    for step in range(cfg.steps):
        batch_idx = (step * cfg.batch_size) % len(dataset)
        batch = dataset[batch_idx:batch_idx + cfg.batch_size]
        if len(batch) < cfg.batch_size:
            batch = batch + dataset[:cfg.batch_size - len(batch)]

        prompts = [build_prompt(ex) for ex in batch]
        
        try:
            sampling_tasks = [
                sample_completions(sampling_client, tokenizer, prompt, cfg)
                for prompt in prompts
            ]
            batch_samples = await asyncio.gather(*sampling_tasks)
        except Exception as e:
            print(f"WARNING: Sampling failed at step {step}: {e}")
            continue
        
        all_data = []
        step_format_valid = 0
        step_format_total = 0
        step_correct = 0  # Track accuracy even though we don't reward it
        
        for ex, prompt, samples in zip(batch, prompts, batch_samples):
            rewards = []

            for s in samples:
                pred_label, valid = parse_output(s["text"])
                reward_breakdown = compute_format_only_reward(pred_label, valid, cfg)
                rewards.append(reward_breakdown["total"])
                
                step_format_total += 1
                if valid:
                    step_format_valid += 1
                if pred_label == ex["label"]:
                    step_correct += 1
            
            baseline = np.mean(rewards)
            advantages = [(r - baseline) for r in rewards]
            if len(advantages) > 1:
                std = np.std(advantages) + 1e-8
                advantages = [(a / std) for a in advantages]
            
            for i, s in enumerate(samples):
                full_tokens = tokenizer.encode(prompt) + list(s["tokens"])
                prompt_len = len(tokenizer.encode(prompt))
                target_tokens = full_tokens[1:]
                adv_per_token = [advantages[i]] * len(s["tokens"])
                adv_full = [0.0] * (prompt_len - 1) + adv_per_token
                adv_full = adv_full[:len(target_tokens)]

                token_lps = s.get("token_logprobs", []) or []
                old_logprobs_full = [0.0] * (prompt_len - 1) + list(token_lps)
                old_logprobs_full = old_logprobs_full[:len(target_tokens)]
                
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": target_tokens,
                        "logprobs": old_logprobs_full,
                        "advantages": adv_full,
                    }
                )
                all_data.append(datum)
        
        if all_data:
            fwd_future = await training_client.forward_backward_async(all_data, loss_fn="ppo")
            opt_future = await training_client.optim_step_async(
                types.AdamParams(
                    learning_rate=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
            )
            await fwd_future
            await opt_future
        
        format_rate = step_format_valid / max(step_format_total, 1)
        accuracy = step_correct / max(step_format_total, 1)
        
        step_metrics = {
            "step": step,
            "format_rate": format_rate,
            "accuracy": accuracy,  # Not rewarded, just tracked
        }
        metrics_history.append(step_metrics)
        
        print(f"Step {step:3d}: Format={format_rate:.3f} | Accuracy={accuracy:.3f} (not rewarded)")
        
        if step % 25 == 0 and step > 0:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"format_only_step_{step}")
            with open("training_metrics_format_only.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
    
    with open("training_metrics_format_only.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nSaved training metrics to training_metrics_format_only.json")
    
    final_path = (await training_client.save_weights_for_sampler_async(name="grpo-format-only-baseline")).result().path
    print(f"\n" + "=" * 60)
    print("FORMAT-ONLY BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoint: {final_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python sample_grpo.py fever '{final_path}'")
    print(f"  python sample_grpo.py vitaminc '{final_path}'")
    print(f"\nCompare against your full GRPO checkpoint to see the effect")
    print(f"of your reward function on accuracy and calibration.")
    print("=" * 60)
    return final_path


if __name__ == "__main__":
    asyncio.run(run_training())

