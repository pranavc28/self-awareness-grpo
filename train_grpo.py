"""GRPO training for FEVER fact verification with self-aware confidence calibration."""
import asyncio, json, random, re, numpy as np
from dataclasses import dataclass
import tinker
from tinker import types

MODEL = "moonshotai/Kimi-K2-Thinking"
SEED = 42
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}
REV_MAP = {v: k for k, v in LABEL_MAP.items()}

@dataclass
class Config:
    batch_size: int = 8
    group_size: int = 8
    max_tokens: int = 96
    temperature: float = 0.8
    lr: float = 2e-6
    steps: int = 500
    w_na: float = 2.0
    w_pass: float = 1.0
    w_fail: float = 1.0
    gamma: float = 0.25
    delta: float = 2.0
    lambda_false_na: float = 1.5
    eta_align: float = 0.5
    format_penalty: float = 2.0
    beta_kl: float = 0.02

def load_dataset():
    random.seed(SEED)
    with open("fever_data/fever_dev.json", "r") as f:
        examples = json.load(f)
    by_label = {"NOT ENOUGH INFO": [], "SUPPORTS": [], "REFUTES": []}
    for ex in examples:
        if ex.get("label") in by_label:
            by_label[ex["label"]].append(ex)
    for lbl, lst in by_label.items():
        if lbl == "NOT ENOUGH INFO" and len(lst) < 500:
            raise ValueError(f"Not enough {lbl}: {len(lst)}")
        if lbl != "NOT ENOUGH INFO" and len(lst) < 250:
            raise ValueError(f"Not enough {lbl}: {len(lst)}")
    random.shuffle(by_label["NOT ENOUGH INFO"])
    random.shuffle(by_label["SUPPORTS"])
    random.shuffle(by_label["REFUTES"])
    sampled = by_label["NOT ENOUGH INFO"][:500] + by_label["SUPPORTS"][:250] + by_label["REFUTES"][:250]
    random.shuffle(sampled)
    return [{"prompt": ex["claim"], "label": LABEL_MAP[ex["label"]]} for ex in sampled]

def build_prompt(example):
    return f"""Claim: {example['prompt']}

Classify as NA (not enough info), PASS (supports), or FAIL (refutes).
Return EXACTLY:
LABEL=<NA|PASS|FAIL>
CONF=<float 0-1, 2 decimals>
RATIONALE=<one sentence or empty>
"""

def parse_output(text):
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
    eps = 1e-10
    max_lp = max(logprobs_dict.values())
    exp_sum = sum(np.exp(lp - max_lp) for lp in logprobs_dict.values())
    probs = {k: np.exp(v - max_lp) / exp_sum for k, v in logprobs_dict.items()}
    return probs

def compute_reward(gold, pred_label, pred_conf, probs, cfg: Config, format_valid):
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
    r_format = 0.0 if format_valid else -cfg.format_penalty
    return r_cls + r_conf + r_false_na + r_align + r_format

async def probe_label_logprobs(sampling_client, tokenizer, prompt_text, renderer=None):
    logprobs = {}
    for label in ["NA", "PASS", "FAIL"]:
        full_text = prompt_text + f"LABEL={label}\n"
        tokens = tokenizer.encode(full_text)
        model_input = types.ModelInput.from_ints(tokens)
        try:
            result = await sampling_client.compute_logprobs_async(model_input)
            lp_list = result.result() if hasattr(result, 'result') else result
            if hasattr(lp_list, 'logprobs'):
                lp_list = lp_list.logprobs
            valid_lps = [lp for lp in lp_list if lp is not None]
            logprobs[label] = sum(valid_lps[-5:]) if valid_lps else -10.0
        except Exception as e:
            logprobs[label] = -10.0
    return logprobs

async def sample_completions(sampling_client, tokenizer, prompt_text, cfg: Config):
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
    for ex in test_batch:
        prompt = build_prompt(ex)
        samples = await sample_completions(sampling_client, tokenizer, prompt, cfg)
        for s in samples[:2]:
            label, conf, valid = parse_output(s["text"])
            print(f"  Gold={ex['label']} Pred={label} Conf={conf:.2f} Valid={valid}")
    
    print(f"\nStarting GRPO training for {cfg.steps} steps...")
    for step in range(cfg.steps):
        batch_idx = (step * cfg.batch_size) % len(dataset)
        batch = dataset[batch_idx:batch_idx + cfg.batch_size]
        if len(batch) < cfg.batch_size:
            batch = batch + dataset[:cfg.batch_size - len(batch)]
        
        all_data, all_advantages = [], []
        step_rewards = {"NA": [], "PASS": [], "FAIL": []}
        correct, total = 0, 0
        
        for ex in batch:
            prompt = build_prompt(ex)
            samples = await sample_completions(sampling_client, tokenizer, prompt, cfg)
            label_lps = await probe_label_logprobs(sampling_client, tokenizer, prompt)
            probs = compute_probs_from_logits(label_lps)
            
            rewards = []
            for s in samples:
                pred_label, pred_conf, valid = parse_output(s["text"])
                r = compute_reward(ex["label"], pred_label, pred_conf, probs, cfg, valid)
                rewards.append(r)
                step_rewards[ex["label"]].append(r)
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
                weights = [0.0] * (prompt_len - 1) + [1.0] * len(s["tokens"])
                weights = weights[:len(target_tokens)]
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
        
        if step % 10 == 0:
            avg_r = {k: np.mean(v) if v else 0 for k, v in step_rewards.items()}
            acc = correct / max(total, 1)
            print(f"Step {step}: Acc={acc:.3f} R_NA={avg_r['NA']:.3f} R_PASS={avg_r['PASS']:.3f} R_FAIL={avg_r['FAIL']:.3f}")
        
        if step % 50 == 0 and step > 0:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"step_{step}")
    
    final_path = (await training_client.save_weights_for_sampler_async(name="final")).result().path
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    return final_path

if __name__ == "__main__":
    asyncio.run(run_training())

