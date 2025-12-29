"""GRPO training for FEVER fact verification with self-aware confidence calibration.

This module implements Group Relative Policy Optimization (GRPO) for training a model
to perform fact verification on the FEVER dataset. The training incorporates:
- Classification accuracy rewards
- Confidence calibration penalties/bonuses  
- False NA prevention
- Format compliance checking

The reward function is designed to encourage:
1. Correct classification (via weighted log-probability rewards)
2. High confidence on correct predictions, low confidence on incorrect ones
3. Avoiding false "Not Enough Info" predictions when evidence exists
4. Alignment between stated confidence and model's internal probability
"""
import asyncio, json, random, re, numpy as np
from dataclasses import dataclass
from functools import wraps
import tinker
from tinker import types

from extract_fever_dataset import WikiReader, download_wiki_pages

MODEL = "moonshotai/Kimi-K2-Thinking"
SEED = 42
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 10.0  # seconds


def async_retry(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
    """
    Decorator for async functions that retries on failure with exponential backoff.
    
    Uses jitter to avoid thundering herd when multiple requests retry simultaneously.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (doubles each attempt)
    """
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
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), RETRY_MAX_DELAY)
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = delay * jitter
                        print(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
            # All retries exhausted
            raise last_exception
        return wrapper
    return decorator

# Global WikiReader instance for evidence lookup
_wiki_reader: WikiReader | None = None


def get_wiki_reader() -> WikiReader:
    """
    Get or create the global WikiReader instance.
    
    Lazily initializes the WikiReader on first call to avoid loading the full
    Wikipedia index unless needed. The WikiReader provides access to evidence
    sentences from the FEVER Wikipedia dump.
    
    Returns:
        WikiReader: Initialized reader for looking up evidence texts.
    """
    global _wiki_reader
    if _wiki_reader is None:
        print("Initializing WikiReader...")
        wiki_dir = download_wiki_pages()
        _wiki_reader = WikiReader(wiki_dir)
    return _wiki_reader


@dataclass
class Config:
    """
    Configuration for GRPO training hyperparameters.
    
    Reward function parameters:
        w_na: Weight for NOT ENOUGH INFO classification reward (higher to emphasize)
        w_pass: Weight for SUPPORTS classification reward
        w_fail: Weight for REFUTES classification reward
        gamma: Bonus coefficient for confidence when prediction is correct
        delta: Penalty coefficient for confidence when prediction is incorrect
        lambda_false_na: Penalty for incorrectly predicting NA when gold is SUPPORTS/REFUTES
        eta_align: Penalty coefficient for confidence-probability misalignment
        format_penalty: Penalty for invalid output format
        beta_kl: KL divergence regularization weight
    
    Training parameters:
        batch_size: Number of examples per training step
        group_size: Number of samples per example for GRPO advantage estimation
        max_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature for exploration
        lr: Learning rate for AdamW optimizer
        steps: Total training steps
    """
    batch_size: int = 64  # 64 examples × 4 samples = 256 completions per step
    group_size: int = 4
    max_tokens: int = 96
    temperature: float = 0.8
    lr: float = 2e-6
    steps: int = 500
    w_na: float = 2.0
    w_pass: float = 1.0
    w_fail: float = 1.0
    gamma: float = 0.25
    delta: float = 1.5
    lambda_false_na: float = 1.5
    eta_align: float = 0.5
    beta_kl: float = 0.02

_NOT_ENOUGH_INFO_COUNT = 500
_SUPPORTS_COUNT = 250
_REFUTES_COUNT = 250

def load_dataset(not_enough_info_count: int = _NOT_ENOUGH_INFO_COUNT, supports_count: int = _SUPPORTS_COUNT, refutes_count: int = _REFUTES_COUNT):
    """
    Load and sample a balanced subset of the FEVER development set.
    
    Creates a balanced dataset with:
    - not_enough_info_count NOT ENOUGH INFO examples (mapped to NA)
    - supports_count SUPPORTS examples (mapped to PASS)  
    - refutes_count REFUTES examples (mapped to FAIL)
    
    Each example includes the original evidence field for WikiReader lookup,
    which is used during prompt construction to provide supporting context.
    
    Returns:
        list[dict]: List of examples with keys:
            - prompt: The claim text
            - label: Mapped label (NA/PASS/FAIL)
            - evidence: Original evidence list for WikiReader lookup
    
    Raises:
        ValueError: If insufficient examples exist for any label category.
    """
    random.seed(SEED)
    with open("fever_data/fever_dev.json", "r") as f:
        examples = json.load(f)
    by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in examples:
        if ex.get("label") in by_label:
            by_label[ex["label"]].append(ex)
    sampled = []
    for lbl, count in [("NOT ENOUGH INFO", not_enough_info_count), ("SUPPORTS", supports_count), ("REFUTES", refutes_count)]:
        sampled.extend(random.sample(by_label[lbl], count))
    random.shuffle(sampled)
    return [{"prompt": ex["claim"], "label": LABEL_MAP[ex["label"]], "evidence": ex.get("evidence", [])} for ex in sampled]


def build_prompt(example):
    """
    Build the classification prompt with claim and evidence context.
    
    Constructs a prompt that includes:
    1. The claim to be verified
    2. Retrieved evidence sentences from Wikipedia (via WikiReader)
    3. Instructions for outputting LABEL, CONF, and RATIONALE
    
    The evidence helps the model make informed predictions. For NA cases,
    evidence may be empty or insufficient, which the model should recognize.
    This affects the reward function: the model should express lower confidence
    when evidence is weak and avoid false NA predictions when evidence exists.
    
    Args:
        example: Dict with 'prompt' (claim text) and 'evidence' (evidence list)
    
    Returns:
        str: Formatted prompt string for the model.
    """
    wiki_reader = get_wiki_reader()
    evidence_texts = wiki_reader.get_evidence_text(example.get("evidence", []))
    
    if evidence_texts:
        evidence = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence = "No specific evidence provided."
    
    return f"""You are a fact verification expert. Classify whether a claim is supported, refuted, or has insufficient evidence.

Claim: {example['prompt']}

You have additional evidence to consider in case you need it:
{evidence}

Classify into exactly one category:
- PASS: Evidence supports the claim (SUPPORTS)
- FAIL: Evidence contradicts the claim (REFUTES)
- NA: Insufficient evidence (NOT ENOUGH INFO)

Respond with EXACTLY this format:
LABEL=<NA|PASS|FAIL>
CONF=<float 0-1, 2 decimals>
RATIONALE=<one sentence or empty>
"""

def parse_output(text):
    """
    Parse model output to extract label, confidence, and format validity.
    
    Extracts structured output from the model's response text. The expected format:
        LABEL=<NA|PASS|FAIL>
        CONF=<float 0-1>
        RATIONALE=<text>
    
    Format validity affects the reward function via format_penalty. Invalid outputs
    receive a penalty to encourage the model to follow the specified format.
    
    Args:
        text: Raw model output string.
    
    Returns:
        tuple: (label, conf, valid) where:
            - label: Extracted label (NA/PASS/FAIL) or None if missing
            - conf: Confidence float clamped to [0,1], default 0.5 if missing
            - valid: Boolean indicating if output format was correct
    """
    label, conf, valid = None, 0.5, True
    m = re.search(r"LABEL\s*=\s*(NA|PASS|FAIL)", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
    else:
        valid = False
    m = re.search(r"CONF\s*=\s*([\d.]+)", text)
    if m:
        try:
            conf = float(m.group(1))
            conf = max(0.0, min(1.0, conf))
        except:
            valid = False
    else:
        valid = False
    return label, conf, valid

def compute_probs_from_logits(logprobs_dict):
    """
    Convert log-probabilities to normalized probabilities using softmax.
    
    Uses the log-sum-exp trick for numerical stability. The resulting probabilities
    are used in the reward function to:
    1. Compute r_cls: Weighted log-probability of the gold label
    2. Determine y_hat: The model's most likely prediction
    3. Compute r_align: Penalize mismatch between stated confidence and p_hat
    
    Args:
        logprobs_dict: Dict mapping labels (NA/PASS/FAIL) to their log-probabilities.
    
    Returns:
        dict: Normalized probability distribution over labels summing to 1.
    """
    max_lp = max(logprobs_dict.values())
    exp_sum = sum(np.exp(lp - max_lp) for lp in logprobs_dict.values())
    probs = {k: np.exp(v - max_lp) / exp_sum for k, v in logprobs_dict.items()}
    return probs

def compute_reward(gold, pred_conf, probs, cfg: Config, format_valid):
    """
    Compute the total reward for a single model prediction.
    
    The reward function combines five components to encourage self-aware,
    well-calibrated fact verification:
    
    1. r_cls (Classification Reward):
       w[gold] * log(p_gold)
       Weighted log-probability of the correct label. Higher weights (w_na=2.0)
       for NA encourage the model to properly identify insufficient evidence.
    
    2. r_conf (Confidence Reward/Penalty):
       - If correct (y_hat == gold): +gamma * max(0, conf - 0.5)
         Rewards high confidence on correct predictions.
       - If incorrect: -delta * max(0, conf - 0.5)  
         Penalizes overconfidence on wrong predictions.
       This teaches calibration: be confident when right, uncertain when wrong.
    
    3. r_false_na (False NA Penalty):
       -lambda_false_na if gold != NA and y_hat == NA
       Penalizes predicting "not enough info" when evidence exists.
       Prevents the model from taking the easy way out.
    
    4. r_align (Alignment Penalty):
       -eta_align * |pred_conf - p_hat|
       Penalizes mismatch between stated confidence and model's internal
       probability. Encourages honest confidence reporting.
    
    5. r_format (Format Penalty):
       -format_penalty if output format is invalid
       Encourages adherence to the LABEL/CONF/RATIONALE format.
    
    Args:
        gold: Ground truth label (NA/PASS/FAIL).
        pred_conf: Stated confidence from parsed output [0,1].
        probs: Probability distribution from probe_label_logprobs.
        cfg: Config with reward function hyperparameters.
        format_valid: Whether the output format was valid.
    
    Returns:
        dict: Reward breakdown with keys:
            - total: Sum of all components
            - r_cls: Classification reward
            - r_conf: Confidence reward/penalty
            - r_false_na: False NA penalty
            - r_align: Alignment penalty
            - r_format: Format penalty
    """
    eps = 1e-10
    w_map = {"NA": cfg.w_na, "PASS": cfg.w_pass, "FAIL": cfg.w_fail}
    p_gold = probs.get(gold, eps)
    r_cls = w_map[gold] * np.log(p_gold + eps)
    y_hat = max(probs, key=probs.get)
    p_hat = probs[y_hat]
    if y_hat == gold:
        r_conf = cfg.gamma * max(0.0, pred_conf - 0.5)
    else:
        r_conf = -cfg.delta * max(0.0, pred_conf - 0.5)
    r_false_na = -cfg.lambda_false_na if (gold != "NA" and y_hat == "NA") else 0.0
    r_align = -cfg.eta_align * abs(pred_conf - p_hat)
    r_format = 0.0
    total = r_cls + r_conf + r_false_na + r_align + r_format
    return {
        "total": total,
        "r_cls": r_cls,
        "r_conf": r_conf,
        "r_false_na": r_false_na,
        "r_align": r_align,
        "r_format": r_format,
    }

@async_retry()
async def probe_batch_logprobs(training_client, tokenizer, prompts):
    """
    Probe the model's log-probabilities for all labels across a batch of prompts.
    
    This is highly optimized: it sends all probe Datums for the entire batch 
    in a single forward_backward call, maximizing GPU throughput.
    
    Args:
        training_client: Tinker training client.
        tokenizer: Tokenizer instance.
        prompts: List of prompt strings for the batch.
        
    Returns:
        list[dict]: A list (one per prompt) of label-to-logprob mappings.
    """
    all_probe_data = []
    all_label_info = [] # List of (label, start, end) per prompt
    
    for prompt_text in prompts:
        prefix = prompt_text + "LABEL="
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_len = len(prefix_tokens)
        
        prompt_labels = []
        for label in ["NA", "PASS", "FAIL"]:
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            full_tokens = prefix_tokens + label_tokens
            
            input_tokens = full_tokens[:-1]
            target_tokens = np.array(full_tokens[1:], dtype=np.int64)
            weights = np.zeros(len(target_tokens), dtype=np.float32)
            
            label_start_idx = prefix_len - 1
            label_end_idx = label_start_idx + len(label_tokens)
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights,
                }
            )
            all_probe_data.append(datum)
            prompt_labels.append((label, label_start_idx, label_end_idx))
        all_label_info.append(prompt_labels)
    
    # Run a single batch forward_backward for all 3 * batch_size datums
    fwd_result = await training_client.forward_backward_async(all_probe_data, loss_fn="cross_entropy")
    fwd_output = fwd_result.result() if hasattr(fwd_result, 'result') else fwd_result
    
    results = []
    datum_idx = 0
    for prompt_labels in all_label_info:
        logprobs = {}
        for label, start_idx, end_idx in prompt_labels:
            output_logprobs = fwd_output.loss_fn_outputs[datum_idx]['logprobs']
            if hasattr(output_logprobs, 'tolist'):
                output_logprobs = output_logprobs.tolist()
            
            label_logprobs = output_logprobs[start_idx:end_idx]
            logprobs[label] = sum(label_logprobs) if label_logprobs else -10.0
            datum_idx += 1
        results.append(logprobs)
        
    return results

@async_retry()
async def sample_completions(sampling_client, tokenizer, prompt_text, cfg: Config):
    """
    Sample multiple completions from the model for GRPO advantage estimation.
    
    GRPO (Group Relative Policy Optimization) requires multiple samples per prompt
    to estimate relative advantages. This function generates cfg.group_size samples
    with temperature-based exploration to discover diverse completion strategies.
    
    Each sample is parsed (via parse_output) to extract label, confidence, and
    format validity. Rewards are computed for each sample, and advantages are
    calculated relative to the group mean. This enables learning which sampled
    behaviors are better/worse than the baseline.
    
    Args:
        sampling_client: Tinker sampling client for generation.
        tokenizer: Tokenizer for prompt encoding and response decoding.
        prompt_text: The full prompt string (claim + evidence + instructions).
        cfg: Config with sampling parameters (max_tokens, temperature, group_size).
    
    Returns:
        list[dict]: List of samples, each containing:
            - text: Decoded response text
            - tokens: Raw token IDs
            - logprob: Sum of token log-probabilities (for PPO loss)
    """
    tokens = tokenizer.encode(prompt_text)
    model_input = types.ModelInput.from_ints(tokens)
    params = types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=["\n\n", "RATIONALE="]
    )
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=cfg.group_size,
        sampling_params=params
    )
    samples = []
    for seq in result.sequences:
        text = tokenizer.decode(seq.tokens)
        lps = seq.logprobs if hasattr(seq, 'logprobs') else []
        total_lp = sum(lps) if lps else 0.0
        samples.append({"text": text, "tokens": seq.tokens, "logprob": total_lp})
    return samples

async def run_training():
    """
    Main GRPO training loop for FEVER fact verification.
    
    Training Process:
    1. Load balanced FEVER dataset (500 NA, 250 PASS, 250 FAIL)
    2. Initialize LoRA training client and sampling client
    3. For each training step:
       a. Sample a batch of examples
       b. For each example (in parallel):
          - Generate group_size completions with temperature sampling
          - Probe label log-probabilities for reward computation
          - Compute rewards using the multi-component reward function
          - Calculate advantages relative to group mean
       c. Perform PPO-style forward-backward pass with advantages
       d. Update model weights with AdamW optimizer
    4. Periodically checkpoint and log metrics
    
    Reward Function Summary (see compute_reward for details):
    - r_cls: Weighted log-probability of correct label
    - r_conf: Confidence bonus/penalty based on correctness
    - r_false_na: Penalty for false "not enough info" predictions
    - r_align: Penalty for confidence-probability mismatch
    - r_format: Penalty for invalid output format
    
    Returns:
        str: Path to the final saved checkpoint.
    """
    cfg = Config()
    print(f"Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} examples")
    
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="init")
    
    print("Running sanity check...")
    test_batch = dataset[:5]
    sanity_tasks = [
        sample_completions(sampling_client, tokenizer, build_prompt(ex), cfg)
        for ex in test_batch
    ]
    sanity_results = await asyncio.gather(*sanity_tasks)
    for ex, samples in zip(test_batch, sanity_results):
        for s in samples[:2]:
            label, conf, valid = parse_output(s["text"])
            print(f"  Gold={ex['label']} Pred={label} Conf={conf:.2f} Valid={valid}")
    
    metrics_history = []
    reward_components = ["total", "r_cls", "r_conf", "r_false_na", "r_align", "r_format"]
    
    print(f"\nStarting GRPO training for {cfg.steps} steps...")
    for step in range(cfg.steps):
        batch_idx = (step * cfg.batch_size) % len(dataset)
        batch = dataset[batch_idx:batch_idx + cfg.batch_size]
        if len(batch) < cfg.batch_size:
            batch = batch + dataset[:cfg.batch_size - len(batch)]

        # 1. Build prompts for the whole batch
        prompts = [build_prompt(ex) for ex in batch]
        
        # 2. Launch all sampling and probing tasks concurrently
        # We launch batch_size sampling tasks (each doing group_size samples)
        # and ONE big probing task for the entire batch.
        try:
            sampling_tasks = [
                sample_completions(sampling_client, tokenizer, prompt, cfg)
                for prompt in prompts
            ]
            probing_task = probe_batch_logprobs(training_client, tokenizer, prompts)
            
            # Wait for everything in this batch to complete
            all_batch_results = await asyncio.gather(*sampling_tasks, probing_task)
            
            # Unpack results: first N are samples, last is the batch logprobs
            batch_samples = all_batch_results[:-1]
            batch_label_lps = all_batch_results[-1]
        except Exception as e:
            print(f"WARNING: Batch processing failed at step {step}: {e}")
            continue
        
        all_data = []
        step_rewards_by_outcome = {
            label: {comp: [] for comp in reward_components}
            for label in ["NA", "PASS", "FAIL"]
        }
        step_rewards_cumulative = {comp: [] for comp in reward_components}
        correct, total = 0, 0
        
        # 3. Process each example in the batch
        for ex, prompt, samples, label_lps in zip(batch, prompts, batch_samples, batch_label_lps):
            probs = compute_probs_from_logits(label_lps)
            rewards = []
            for s in samples:
                pred_label, pred_conf, valid = parse_output(s["text"])
                reward_breakdown = compute_reward(ex["label"], pred_conf, probs, cfg, valid)
                rewards.append(reward_breakdown["total"])
                
                for comp in reward_components:
                    step_rewards_by_outcome[ex["label"]][comp].append(reward_breakdown[comp])
                    step_rewards_cumulative[comp].append(reward_breakdown[comp])
                
                if pred_label == ex["label"]:
                    correct += 1
                total += 1
            
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
                
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": target_tokens,
                        "logprobs": [s["logprob"] / max(len(s["tokens"]), 1)] * len(target_tokens),
                        "advantages": adv_full,
                    }
                )
                all_data.append(datum)
        
        if all_data:
            fwd_future = await training_client.forward_backward_async(all_data, loss_fn="ppo")
            opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=cfg.lr))
            await fwd_future
            await opt_future
        
        acc = correct / max(total, 1)
        step_metrics = {
            "step": step,
            "accuracy": acc,
            "by_outcome": {},
            "cumulative": {}
        }
        
        for label in ["NA", "PASS", "FAIL"]:
            step_metrics["by_outcome"][label] = {
                comp: float(np.mean(step_rewards_by_outcome[label][comp])) 
                      if step_rewards_by_outcome[label][comp] else 0.0
                for comp in reward_components
            }
        
        step_metrics["cumulative"] = {
            comp: float(np.mean(step_rewards_cumulative[comp])) 
                  if step_rewards_cumulative[comp] else 0.0
            for comp in reward_components
        }
        
        metrics_history.append(step_metrics)
        
        if step % 10 == 0:
            cum = step_metrics["cumulative"]
            print(f"Step {step}: Acc={acc:.3f} | "
                  f"r_cls={cum['r_cls']:.3f} r_conf={cum['r_conf']:.3f} "
                  f"r_false_na={cum['r_false_na']:.3f} r_align={cum['r_align']:.3f} "
                  f"r_format={cum['r_format']:.3f} | total={cum['total']:.3f}")
        
        if step % 50 == 0 and step > 0:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"step_{step}")
            with open("training_metrics.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
    
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved training metrics to training_metrics.json")
    
    final_path = (await training_client.save_weights_for_sampler_async(name="self-aware-grpo-test")).result().path
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    return final_path

if __name__ == "__main__":
    asyncio.run(run_training())

