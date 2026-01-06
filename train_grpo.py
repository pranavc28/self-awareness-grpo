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
    
    Reward function parameters:
        correct_pos: Reward for correct label emission (parsed output)
        correct_neg: Penalty for incorrect label emission (parsed output)
        alpha_logp: Weight on log(p_gold) shaping term from probed label distribution
        eta_cal: Log-scoring calibration weight (penalizes overconfidence when wrong)
        eta_align: Confidence-vs-probed-probability alignment weight
        lambda_false_na: Penalty for emitting NA when evidence exists and gold is PASS/FAIL
        lambda_inconsistent: Penalty when emitted label != probed argmax label (discourages reward hacking)
        format_penalty: Penalty for invalid output format
        beta_kl: KL divergence regularization weight (if supported by PPO impl)
    
    Training parameters:
        batch_size: Number of examples per training step
        group_size: Number of samples per example for GRPO advantage estimation
        max_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature for exploration
        lr: Learning rate for AdamW optimizer
        steps: Total training steps
    """
    # Keep total samples/step roughly constant while improving within-group signal:
    # 32 examples × 8 samples = 256 completions per step (same as 64×4).
    batch_size: int = 32
    group_size: int = 8
    max_tokens: int = 512
    temperature: float = 0.2
    lr: float = 2e-6
    steps: int = 160
    # Increase correctness reward so it dominates early learning.
    correct_pos: float = 2.0
    correct_neg: float = 1.0
    alpha_logp: float = 0.10
    # These are the *end* values; we ramp up from 0 over warmup_steps.
    # Higher values for better calibration (previously 0.10/0.15)
    eta_cal: float = 0.50
    eta_align: float = 0.30
    warmup_steps: int = 50
    lambda_false_na: float = 0.5
    lambda_inconsistent: float = 0.25
    format_penalty: float = 0.25
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
    fever_na: int = 200, fever_pass: int = 70, fever_fail: int = 70,
    vitaminc_na: int = 200, vitaminc_pass: int = 70, vitaminc_fail: int = 70,
    climatefever_na: int = 200, climatefever_pass: int = 70, climatefever_fail: int = 70,
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
    Build a generalizable classification prompt with claim and evidence context.
    
    Uses Natural Language Inference (NLI) framing for better cross-dataset
    generalization. The prompt focuses on logical relationships between
    evidence and claim rather than dataset-specific terminology.
    
    Key design principles from fact verification literature:
    1. NLI framing (entailment/contradiction/neutral) generalizes better
    2. First-principles logical reasoning over dataset-specific patterns
    3. Clear abstention criteria based on evidence sufficiency
    4. No dataset-specific terminology hints
    
    Args:
        example: Dict with 'prompt' (claim text) and 'evidence' (evidence list or string)
                 For FEVER: evidence is a list for WikiReader lookup
                 For VitaminC/ClimateFEVER: evidence is a direct string
    
    Returns:
        str: Formatted prompt string for the model.
    """
    evidence_raw = example.get("evidence", [])
    source = example.get("source", "fever")
    
    # Handle different evidence formats based on source
    if source == "fever" and isinstance(evidence_raw, list):
        # FEVER uses WikiReader for evidence lookup
        wiki_reader = get_wiki_reader()
        evidence_texts = wiki_reader.get_evidence_text(evidence_raw)
    elif isinstance(evidence_raw, str) and evidence_raw:
        # VitaminC/ClimateFEVER provide evidence as string directly
        evidence_texts = [evidence_raw]
    else:
        evidence_texts = []
    
    if evidence_texts:
        evidence = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence = "No relevant evidence available."
    
    return f"""You are an expert at Natural Language Inference. Your task is to determine the logical relationship between evidence and a claim.

CLAIM: {example['prompt']}

EVIDENCE:
{evidence}

Determine the relationship between the evidence and the claim:

- PASS: The evidence ENTAILS the claim. The claim logically follows from the evidence. If the evidence is true, the claim must be true.
- FAIL: The evidence CONTRADICTS the claim. The claim is logically inconsistent with the evidence. If the evidence is true, the claim must be false.
- NA: The evidence is NEUTRAL. The evidence neither entails nor contradicts the claim. The claim could be true or false independent of this evidence.

Decision guidelines:
1. Focus ONLY on what the evidence explicitly states or directly implies.
2. Choose PASS only if the evidence provides clear support for the claim being true.
3. Choose FAIL only if the evidence provides clear support for the claim being false.
4. Choose NA when the evidence is unrelated, insufficient, or does not bear on the claim's truth value.
5. Do not use external knowledge beyond what is stated in the evidence.

<Output Format>
The LABEL should be one of PASS, FAIL, or NA based on the decision guidelines.

CONF is your probability estimate (0.00 to 1.00) that your chosen LABEL is correct. Calibrate carefully:
- 0.90-0.99: Evidence is explicit and unambiguous. You are highly certain.
- 0.75-0.89: Evidence strongly suggests the answer but requires minor inference.
- 0.60-0.74: Evidence is relevant but the conclusion requires interpretation or has some ambiguity.
- 0.50-0.59: Borderline case. Evidence is weak, indirect, or you are genuinely uncertain.

IMPORTANT: Do NOT default to high confidence. Use lower confidence when:
- The evidence requires multiple inferential steps
- Key details are missing or ambiguous  
- The claim uses vague language ("only", "always", "some")
- You are choosing NA due to insufficient evidence

Return the LABEL, CONF, and RATIONALE each on a separate line. The RATIONALE should be an empty string.

Return only 3 lines. Your output should be in the following format:
LABEL=PASS
CONF=0.75
RATIONALE=

Now begin your response. Do not write anything before LABEL.
</Output Format>
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
    text = text.replace("<", "").replace(">", "")
    m = re.search(r"LABEL\s*[:=]\s*(NA|PASS|FAIL)", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
    else:
        valid = False
    m = re.search(r"CONF\s*[:=]\s*([\d.]+)", text)
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

def compute_reward(
    gold: str,
    pred_label: str | None,
    pred_conf: float,
    probs: dict,
    cfg: Config,
    format_valid: bool,
    evidence_provided: bool,
):
    """
    Compute the total reward for a single model prediction.
    
    The reward function combines multiple components to encourage self-aware,
    well-calibrated fact verification:
    
    1. r_correct (Accuracy Reward):
       Reward/penalty for whether the *emitted* label matches gold.
       This is the PRIMARY learning signal for classification accuracy.
    
    2. r_cls (Shaping via probed logits):
       alpha_logp * log(p_gold) to softly encourage the probed distribution
       to put probability mass on the gold label.
    
    3. r_cal (Calibration via log scoring):
       eta_cal * log(conf) if correct, eta_cal * log(1-conf) if incorrect.
       Log scoring HEAVILY penalizes overconfidence when wrong (log(0.01) ≈ -4.6)
       while only mildly penalizing high confidence when correct (log(0.99) ≈ -0.01).
       This prevents the model from always outputting 0.99.
    
    4. r_false_na (False NA Penalty):
       -lambda_false_na if evidence exists, gold != NA, and the model emits NA.
       Prevents the model from taking the easy way out.
    
    5. r_align (Alignment Penalty):
       -eta_align * |conf - p_pred|, where p_pred is the probed probability
       of the *emitted* label. Encourages consistency between reported confidence
       and the model's own label probabilities.
    
    6. r_inconsistent (Consistency Penalty):
       -lambda_inconsistent if emitted label != probed argmax label.
       Discourages reward hacking where the model emits one label but internally
       prefers another.

    7. r_format (Format Penalty):
       -format_penalty if output format is invalid.
    
    Args:
        gold: Ground truth label (NA/PASS/FAIL).
        pred_label: Label parsed from the sampled completion.
        pred_conf: Confidence parsed from the sampled completion [0,1].
        probs: Probability distribution from probe_label_logprobs.
        cfg: Config with reward function hyperparameters.
        format_valid: Whether the output format was valid.
        evidence_provided: Whether the FEVER example includes evidence annotations.
    
    Returns:
        dict: Reward breakdown with keys:
            - total: Sum of all components
            - r_correct: Accuracy reward (+1 correct, -0.5 incorrect)
            - r_cls: Classification calibration (scaled log-prob)
            - r_cal: Proper-scoring calibration term
            - r_false_na: False NA penalty
            - r_align: Alignment penalty
            - r_inconsistent: Consistency penalty
            - r_format: Format penalty
    """
    eps = 1e-10
    pred_label = (pred_label or "NA").upper()
    if pred_label not in ("NA", "PASS", "FAIL"):
        pred_label = "NA"
        format_valid = False

    y_hat = max(probs, key=probs.get)
    p_gold = probs.get(gold, eps)
    p_pred = probs.get(pred_label, eps)

    # PRIMARY SIGNAL: reward the *emitted* label (this is what your metrics measure).
    r_correct = cfg.correct_pos if pred_label == gold else -cfg.correct_neg

    # Shaping: encourage the probed distribution to put mass on the gold label
    r_cls = cfg.alpha_logp * float(np.log(p_gold + eps))

    # Warm up calibration/alignment to avoid overpowering the accuracy signal early.
    # (Used by run_training via optional overrides; defaults to cfg values.)
    #
    # Calibration: LOG SCORING (proper scoring rule that heavily penalizes overconfidence)
    # - When correct: reward log(conf) → small penalty for high conf
    # - When wrong: reward log(1-conf) → HUGE penalty for high conf when wrong
    # This prevents the model from always outputting 0.99
    eta_cal = getattr(cfg, "eta_cal", 0.0)
    if pred_label == gold:
        r_cal = eta_cal * float(np.log(pred_conf + eps))  # log(0.99) ≈ -0.01
    else:
        r_cal = eta_cal * float(np.log(1 - pred_conf + eps))  # log(0.01) ≈ -4.6!

    # Alignment: stated confidence should match probed probability of the *emitted* label
    eta_align = getattr(cfg, "eta_align", 0.0)
    r_align = -eta_align * float(abs(pred_conf - p_pred))

    # Penalize NA when evidence exists and gold is not NA (prevents NA collapse)
    r_false_na = 0.0
    if evidence_provided and gold != "NA" and pred_label == "NA":
        r_false_na = -cfg.lambda_false_na

    # Penalize inconsistency between what the model emits and what its probed argmax says
    r_inconsistent = -cfg.lambda_inconsistent if pred_label != y_hat else 0.0

    # Format penalty
    r_format = -cfg.format_penalty if not format_valid else 0.0

    total = r_correct + r_cls + r_cal + r_false_na + r_align + r_inconsistent + r_format
    return {
        "total": total,
        "r_correct": r_correct,
        "r_cls": r_cls,
        "r_cal": r_cal,
        "r_false_na": r_false_na,
        "r_align": r_align,
        "r_inconsistent": r_inconsistent,
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
        # If the prompt already ends with LABEL= (we do this to improve compliance),
        # don't append it again.
        prefix = prompt_text if prompt_text.rstrip().endswith("LABEL=") else (prompt_text + "LABEL=")
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
        stop=["\nRATIONALE="]
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
    - r_correct: Reward/penalty for emitted label correctness
    - r_cls: Shaping via probed log-probability of the gold label
    - r_cal: Proper-scoring calibration term for confidence
    - r_false_na: Penalty for emitting NA when evidence exists and gold is PASS/FAIL
    - r_align: Penalty for confidence mismatch vs probed probability of emitted label
    - r_inconsistent: Penalty when emitted label != probed argmax label
    - r_format: Penalty for invalid output format
    
    Returns:
        str: Path to the final saved checkpoint.
    """
    cfg = Config()
    print(f"Loading mixed dataset from FEVER, VitaminC, and ClimateFEVER...")
    # 200 NA + 50 PASS + 50 FAIL from each dataset = 900 total examples
    dataset = load_mixed_dataset(
        fever_na=200, fever_pass=70, fever_fail=70,
        vitaminc_na=200, vitaminc_pass=70, vitaminc_fail=70,
        climatefever_na=200, climatefever_pass=70, climatefever_fail=70,
    )
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
            label, conf, valid = parse_output(s["text"])
            print(f"  Gold={ex['label']} Pred={label} Conf={conf:.2f} Valid={valid}")
    
    metrics_history = []
    reward_components = ["total", "r_correct", "r_cls", "r_cal", "r_false_na", "r_align", "r_inconsistent", "r_format"]

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
            evidence_provided = bool(ex.get("evidence"))
            rewards = []
            # Ramp calibration/alignment from 0 -> configured value over warmup.
            warmup_steps = max(int(getattr(cfg, "warmup_steps", 0) or 0), 0)
            warm = 1.0 if warmup_steps == 0 else min(1.0, (step + 1) / warmup_steps)
            eta_cal_saved, eta_align_saved = cfg.eta_cal, cfg.eta_align
            cfg.eta_cal = eta_cal_saved * warm
            cfg.eta_align = eta_align_saved * warm

            for s in samples:
                pred_label, pred_conf, valid = parse_output(s["text"])
                reward_breakdown = compute_reward(ex["label"], pred_label, pred_conf, probs, cfg, valid, evidence_provided)
                rewards.append(reward_breakdown["total"])
                
                for comp in reward_components:
                    step_rewards_by_outcome[ex["label"]][comp].append(reward_breakdown[comp])
                    step_rewards_cumulative[comp].append(reward_breakdown[comp])
                
                if pred_label == ex["label"]:
                    correct += 1
                total += 1

            # Restore config (avoid surprising future callers)
            cfg.eta_cal, cfg.eta_align = eta_cal_saved, eta_align_saved
            
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
        
        cum = step_metrics["cumulative"]
        print(f"Step {step}: Acc={acc:.3f} | "
            f"r_correct={cum['r_correct']:.3f} r_cls={cum['r_cls']:.3f} "
            f"r_cal={cum['r_cal']:.3f} r_false_na={cum['r_false_na']:.3f} "
            f"r_align={cum['r_align']:.3f} r_incon={cum['r_inconsistent']:.3f} "
            f"r_format={cum['r_format']:.3f} "
            f"| total={cum['total']:.3f}")
        
        if step % 50 == 0 and step > 0:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"step_{step}")
            with open("training_metrics.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
    
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved training metrics to training_metrics.json")
    
    final_path = (await training_client.save_weights_for_sampler_async(name="self-aware-grpo-mixed-v1")).result().path
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    return final_path

if __name__ == "__main__":
    asyncio.run(run_training())

