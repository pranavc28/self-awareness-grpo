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
from extract_vitaminc_dataset import download_vitaminc
from extract_climatefever_dataset import download_climatefever

MODEL = "openai/gpt-oss-20b"
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
    
    OPTIMIZED V6_EXPLORE_MAX CONFIG - Based on reward curve analysis:
    
    Problem solved: NA Collapse
    - Previous config had NA advantage of +6.7 (model always predicted NA)
    - New config has NA advantage of -0.6 (balanced incentives)
    
    Key changes:
    1. Gentler incorrect penalty (0.5 vs 2.0) - makes exploration less risky
    2. Massive exploration bonus (4.0 vs 0.5) - immediate reward for trying PASS/FAIL
    3. Reduced NA weight (0.3 vs 0.8) - breaks the "safe harbor" of defaulting to NA
    4. Lower false_na penalty (2.0 vs 4.0) - avoids double-punishing PASS/FAIL attempts
    
    Expected improvements:
    - Mean PASS reward: -3.8 → +2.6
    - Mean FAIL reward: -5.2 → +1.2
    - Class balance std: 3.22 → 0.66
    
    Reward function parameters:
        correct_pos: Reward for correct label emission
        correct_neg: Penalty for incorrect label emission (kept low for exploration)
        lambda_false_na: Penalty for predicting NA when gold is PASS/FAIL
        format_penalty: Penalty for invalid output format
        class_weight_*: Per-class reward multipliers (NA < PASS = FAIL)
        exploration_bonus: Bonus for attempting PASS/FAIL predictions
    
    Training parameters:
        batch_size: Number of examples per training step
        group_size: Number of samples per example for GRPO advantage estimation
        max_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature (0.7 for exploration)
        lr: Learning rate for AdamW optimizer
        steps: Total training steps
    """
    batch_size: int = 32
    group_size: int = 8
    max_tokens: int = 28000
    # Moderate temperature for diversity (helps explore PASS/FAIL)
    temperature: float = 0.3
    # Lower LR for stability (high variance in your run)
    lr: float = 5e-6
    steps: int = 200
    # Regularization
    weight_decay: float = 0.01
    
    # OPTIMIZED REWARD CONFIG (V6_EXPLORE_MAX) - Based on reward curve analysis
    # Key insight: Lower penalties for trying + massive exploration bonus breaks NA collapse
    # See recommended_config.json for analysis details
    correct_pos: float = 4.0       # Reward for correct prediction (was 3.0)
    correct_neg: float = 0.5       # Penalty for incorrect - MUCH gentler (was 2.0, 75% reduction)
    class_weight_na: float = 0.15  # VERY LOW - make NA unattractive (was 0.3)
    class_weight_pass: float = 2.5 # Higher PASS incentive (was 2.0)
    class_weight_fail: float = 3.5 # HIGHEST weight - FAIL is hardest to learn (was 2.5)
    lambda_false_na: float = 0.5   # MINIMAL - stop punishing exploration (was 2.0)
    exploration_bonus: float = 8.0 # HUGE - make trying PASS/FAIL irresistible (was 4.0)
    
    # Format compliance
    format_penalty: float = 0.25
    
    # KL regularization (for PPO stability)
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

    # Save training IDs to a separate JSON file for future reference
    training_ids = [ex["id"] for ex in sampled]
    with open("training_ids.json", "w") as f:
        json.dump(training_ids, f, indent=2)
    print(f"Saved {len(training_ids)} training IDs to training_ids.json")

    return [{"prompt": ex["claim"], "label": LABEL_MAP[ex["label"]], "evidence": ex.get("evidence", [])} for ex in sampled]


def load_mixed_dataset(
    # IMPROVED: More balanced dataset to combat NA class dominance
    # Original had 200-200-200 NA and 70-70-70 for PASS/FAIL (NA = 59%)
    # New: 120-120-120 gives roughly equal representation (NA = 35%)
    fever_na: int = 100, fever_pass: int = 100, fever_fail: int = 80,
    vitaminc_na: int = 180, vitaminc_pass: int = 180, vitaminc_fail: int = 150,
    climatefever_na: int = 180, climatefever_pass: int = 180, climatefever_fail: int = 150,
):
    """
    Load a mixed dataset from FEVER, VitaminC, and ClimateFEVER for multi-domain training.
    
    This enables learning generalizable self-awareness patterns across different
    fact verification domains: general Wikipedia facts (FEVER), Wikipedia revisions
    (VitaminC), and climate science claims (ClimateFEVER).
    
    Args:
        fever_na/pass/fail: Number of each label from FEVER
        vitaminc_na/pass/fail: Number of each label from VitaminC
        climatefever_na/pass/fail: Number of each label from ClimateFEVER
    
    Returns:
        list[dict]: Mixed dataset with keys:
            - prompt: The claim text
            - label: Mapped label (NA/PASS/FAIL)
            - evidence: Evidence list (FEVER) or string (VitaminC/ClimateFEVER)
            - source: Dataset source identifier (fever/vitaminc/climatefever)
    """
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
                "evidence": ex.get("evidence", []),  # List format for WikiReader
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
                "evidence": ex.get("evidence", ""),  # String format directly
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
                "evidence": ex.get("evidence", ""),  # String format directly
                "source": "climatefever"
            })
    print(f"  Added {climatefever_na + climatefever_pass + climatefever_fail} ClimateFEVER examples")
    
    random.shuffle(sampled)
    
    # Save training IDs
    training_ids = [f"{ex['source']}_{i}" for i, ex in enumerate(sampled)]
    with open("training_ids_mixed.json", "w") as f:
        json.dump(training_ids, f, indent=2)
    
    print(f"\nTotal mixed dataset: {len(sampled)} examples")
    print(f"  NA: {sum(1 for ex in sampled if ex['label'] == 'NA')}")
    print(f"  PASS: {sum(1 for ex in sampled if ex['label'] == 'PASS')}")
    print(f"  FAIL: {sum(1 for ex in sampled if ex['label'] == 'FAIL')}")
    
    return sampled


def build_prompt(example):
    """
    Build a self-aware classification prompt with claim and evidence context.
    
    SELF-AWARENESS PROMPTING TECHNIQUES USED:
    1. Evidence Evaluation First - Model assesses evidence quality before deciding
    2. Epistemic Humility Cues - Explicit encouragement to recognize uncertainty
    3. Confidence Self-Reflection - "How confident should you be?"
    4. Appropriate NA Usage - "Better to say NA when genuinely uncertain than to guess"
    
    This approach is based on research showing that:
    - Chain-of-Thought prompting improves reasoning by breaking down steps
    - Self-reflection prompts lead to more calibrated confidence
    - Epistemic humility cues help models recognize knowledge limits
    
    The prompt guides the model to:
    1. Consider if evidence directly addresses the claim
    2. Assess evidence sufficiency and relevance
    3. Calibrate confidence before making a decision
    4. Choose NA only when genuinely uncertain, not as a default
    
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
    
    # Self-aware prompt with few-shot examples for format compliance
    # Key: Examples teach format, instructions encourage calibrated confidence
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
    """
    Parse model output to extract label and format validity.
    
    Extracts the label from the model's response text. The expected format is
    LABEL=X where X is NA, PASS, or FAIL.
    
    Format validity affects the reward function via format_penalty. Invalid outputs
    receive a penalty to encourage the model to follow the specified format.
    
    Args:
        text: Raw model output string.
    
    Returns:
        tuple: (label, valid) where:
            - label: Extracted label (NA/PASS/FAIL) or None if missing
            - valid: Boolean indicating if output format was correct
    """
    # Strip special tokens that might be appended (e.g., <|end|>, <|return|>)
    text = re.sub(r'<\|[^|]*\|>', '', text).strip()
    text_upper = text.upper()
    
    # Check for exact LABEL=X format (most desirable - mark as valid)
    label_exact = re.match(r'^LABEL\s*=\s*(NA|PASS|FAIL)\s*$', text_upper)
    if label_exact:
        return label_exact.group(1), True
    
    # Check for exact bare label (also acceptable as valid)
    if text_upper in ("NA", "PASS", "FAIL"):
        return text_upper, True
    
    # Fallback: find LABEL=X anywhere in text (penalize - model wrote extra text)
    label_match = re.search(r'LABEL\s*=\s*(NA|PASS|FAIL)', text_upper)
    if label_match:
        return label_match.group(1), False
    
    # Last resort: find bare label anywhere (heavily penalize)
    bare_match = re.search(r'\b(NA|PASS|FAIL)\b', text_upper)
    if bare_match:
        return bare_match.group(1), False
    
    return None, False

# NOTE: compute_probs_from_logits and probe_batch_logprobs removed in simplified version
# These were used for r_cls and r_inconsistent which added noise without improving accuracy

def compute_reward(
    gold: str,
    pred_label: str | None,
    cfg: Config,
    format_valid: bool,
    evidence_provided: bool,
):
    """
    SIMPLIFIED reward function with cleaner learning signal.
    
    REMOVED (too noisy, didn't improve accuracy):
    - r_cls (log-probability calibration)
    - r_inconsistent (argmax mismatch - rarely fires)
    - r_entropy (exploration bonus - model learns format quickly)
    
    KEPT and IMPROVED:
    1. r_correct (Accuracy Reward):
       Class-weighted reward/penalty for correct classification.
       PASS/FAIL get 1.5x weight to combat NA bias.
    
    2. r_false_na (False NA Penalty):
       Strong penalty when predicting NA for PASS/FAIL examples.
       Increased to 3.5 and scaled by class weight.
    
    3. r_format (Format Penalty):
       Penalty for invalid output format.
    
    Args:
        gold: Ground truth label (NA/PASS/FAIL).
        pred_label: Label parsed from the sampled completion.
        cfg: Config with reward function hyperparameters.
        format_valid: Whether the output format was valid.
        evidence_provided: Whether the example includes evidence.
    
    Returns:
        dict: Reward breakdown with keys:
            - total: Sum of all components
            - r_correct: Class-weighted accuracy reward
            - r_false_na: False NA penalty
            - r_format: Format penalty
    """
    pred_label = (pred_label or "NA").upper()
    if pred_label not in ("NA", "PASS", "FAIL"):
        pred_label = "NA"
        format_valid = False

    # Get class weight for the gold label (PASS/FAIL get 1.5x bonus)
    class_weights = {
        "NA": cfg.class_weight_na,
        "PASS": cfg.class_weight_pass,
        "FAIL": cfg.class_weight_fail,
    }
    weight = class_weights.get(gold, 1.0)

    # r_correct: Class-weighted accuracy reward
    if pred_label == gold:
        r_correct = cfg.correct_pos * weight
    else:
        r_correct = -cfg.correct_neg * weight

    # r_false_na: Strong penalty for predicting NA when evidence exists and gold != NA
    r_false_na = 0.0
    if evidence_provided and gold != "NA" and pred_label == "NA":
        # Scale penalty by class weight (punish more for missing PASS/FAIL)
        r_false_na = -cfg.lambda_false_na * weight

    # r_explore: Small bonus for predicting PASS/FAIL (helps break NA collapse)
    # This encourages the model to explore non-NA predictions
    r_explore = 0.0
    if pred_label in ("PASS", "FAIL"):
        r_explore = cfg.exploration_bonus

    # r_format: Format compliance penalty
    r_format = -cfg.format_penalty if not format_valid else 0.0

    total = r_correct + r_false_na + r_explore + r_format
    return {
        "total": total,
        "r_correct": r_correct,
        "r_false_na": r_false_na,
        "r_explore": r_explore,
        "r_format": r_format,
    }


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
        stop=["\n"]  # Stop at newline
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
        # Ensure python list for downstream padding/concat
        if hasattr(token_lps, "tolist"):
            token_lps = token_lps.tolist()
        samples.append({"text": text, "tokens": seq.tokens, "token_logprobs": token_lps})
    return samples

async def run_training():
    """
    IMPROVED GRPO training loop for FEVER fact verification.
    
    Key improvements based on training analysis:
    1. Simplified 3-component reward (removed noisy r_cls, r_inconsistent, r_entropy)
    2. Class-weighted rewards (PASS/FAIL get 1.5x bonus to combat NA bias)
    3. Balanced dataset (equal NA/PASS, slightly less FAIL)
    4. Higher temperature (0.4) for better exploration
    5. Stronger correct/incorrect signal (3.0 / 2.0 vs 2.0 / 1.0)
    6. Removed probing step (faster training)
    
    Training Process:
    1. Load balanced mixed dataset (~1020 examples: 360 NA, 360 PASS, 300 FAIL)
    2. Initialize LoRA training client and sampling client
    3. For each training step:
       a. Sample a batch of examples
       b. For each example (in parallel):
          - Generate group_size completions with temperature sampling
          - Compute simplified rewards (r_correct, r_false_na, r_format)
          - Calculate advantages relative to group mean
       c. Perform PPO-style forward-backward pass with advantages
       d. Update model weights with AdamW optimizer
    4. Periodically checkpoint and log metrics (includes per-class accuracy)
    
    SIMPLIFIED Reward Function:
    - r_correct: Class-weighted accuracy reward/penalty (main signal)
    - r_false_na: Strong penalty for predicting NA when gold is PASS/FAIL  
    - r_format: Penalty for invalid output format
    
    Returns:
        str: Path to the final saved checkpoint.
    """
    cfg = Config()
    print(f"Loading dataset...")
    dataset = load_mixed_dataset()
    print(f"Loaded {len(dataset)} examples")
    
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="init2")
    
    print("Running sanity check...")
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
    # SIMPLIFIED + r_explore to break NA collapse
    reward_components = ["total", "r_correct", "r_false_na", "r_explore", "r_format"]

    print(f"\nStarting GRPO training for {cfg.steps} steps...")
    print(f"Config: correct_pos={cfg.correct_pos}, correct_neg={cfg.correct_neg}, "
          f"lambda_false_na={cfg.lambda_false_na}, exploration_bonus={cfg.exploration_bonus}")
    print(f"        temp={cfg.temperature}, lr={cfg.lr}")
    print(f"Class weights: NA={cfg.class_weight_na}, PASS={cfg.class_weight_pass}, FAIL={cfg.class_weight_fail}\n")
    
    for step in range(cfg.steps):
        batch_idx = (step * cfg.batch_size) % len(dataset)
        batch = dataset[batch_idx:batch_idx + cfg.batch_size]
        if len(batch) < cfg.batch_size:
            batch = batch + dataset[:cfg.batch_size - len(batch)]

        # 1. Build prompts for the whole batch
        prompts = [build_prompt(ex) for ex in batch]
        
        # 2. Sample completions (removed probing - no longer needed for simplified reward)
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
        step_rewards_by_outcome = {
            label: {comp: [] for comp in reward_components}
            for label in ["NA", "PASS", "FAIL"]
        }
        step_rewards_cumulative = {comp: [] for comp in reward_components}
        
        # Track per-class accuracy for better monitoring
        class_correct = {"NA": 0, "PASS": 0, "FAIL": 0}
        class_total = {"NA": 0, "PASS": 0, "FAIL": 0}
        
        # 3. Process each example in the batch
        for ex, prompt, samples in zip(batch, prompts, batch_samples):
            evidence_provided = bool(ex.get("evidence"))
            rewards = []

            for s in samples:
                pred_label, valid = parse_output(s["text"])
                reward_breakdown = compute_reward(ex["label"], pred_label, cfg, valid, evidence_provided)
                rewards.append(reward_breakdown["total"])
                
                for comp in reward_components:
                    step_rewards_by_outcome[ex["label"]][comp].append(reward_breakdown[comp])
                    step_rewards_cumulative[comp].append(reward_breakdown[comp])
                
                # Track per-class accuracy
                class_total[ex["label"]] += 1
                if pred_label == ex["label"]:
                    class_correct[ex["label"]] += 1
            
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

                # Old-policy per-token logprobs are required for stable PPO.
                # Pad prompt positions with 0; only completion tokens get logprobs.
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
            # AdamW with L2 regularization (weight decay)
            opt_future = await training_client.optim_step_async(
                types.AdamParams(
                    learning_rate=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
            )
            await fwd_future
            await opt_future
        
        # Compute per-class accuracies
        acc_na = class_correct["NA"] / max(class_total["NA"], 1)
        acc_pass = class_correct["PASS"] / max(class_total["PASS"], 1)
        acc_fail = class_correct["FAIL"] / max(class_total["FAIL"], 1)
        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        acc = total_correct / max(total_samples, 1)
        
        step_metrics = {
            "step": step,
            "accuracy": acc,
            "accuracy_na": acc_na,
            "accuracy_pass": acc_pass,
            "accuracy_fail": acc_fail,
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
        
        cum = step_metrics["cumulative"]
        # IMPROVED: Show per-class accuracy breakdown and exploration bonus
        print(f"Step {step:3d}: Acc={acc:.3f} (NA:{acc_na:.2f} PASS:{acc_pass:.2f} FAIL:{acc_fail:.2f}) | "
              f"r_correct={cum['r_correct']:+.2f} r_false_na={cum['r_false_na']:+.2f} "
              f"r_explore={cum['r_explore']:+.2f} | total={cum['total']:+.2f}")
        
        if step % 50 == 0 and step > 0:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"step_{step}")
            with open("training_metrics.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
    
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved training metrics to training_metrics.json")
    
    final_path = (await training_client.save_weights_for_sampler_async(name="self-aware-grpo-mixed-regularization-v5")).result().path
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    return final_path

if __name__ == "__main__":
    asyncio.run(run_training())
