"""Sample from trained GRPO checkpoint on FEVER examples.

IMPORTANT: This uses the SAME output format as `train_grpo.py` (LABEL/CONF/RATIONALE)
to avoid evaluation/prompt mismatch, which can otherwise make GRPO look worse than it is.
"""
import asyncio
import json
import re
import sys

import tinker
from tinker import types

from extract_fever_dataset import get_eval_examples, WikiReader, download_wiki_pages

MODEL = "openai/gpt-oss-20b"

# Label mapping: FEVER labels -> simplified labels
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}

# Global WikiReader instance for evidence lookup
_wiki_reader: WikiReader | None = None


def get_wiki_reader() -> WikiReader:
    """Get or create the global WikiReader instance."""
    global _wiki_reader
    if _wiki_reader is None:
        print("Initializing WikiReader...")
        wiki_dir = download_wiki_pages()
        _wiki_reader = WikiReader(wiki_dir)
    return _wiki_reader


def build_prompt(claim: str, evidence_texts: list[str]) -> str:
    """Build the same prompt format used during GRPO training."""
    if evidence_texts:
        evidence = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence = "No specific evidence provided."
    
    return f"""You are a fact verification expert. Classify whether a claim is supported, refuted, or has insufficient evidence.

Claim: {claim}

You have additional evidence to consider in case you need it:
{evidence}

Classify into exactly one category:
- PASS: Evidence supports the claim (SUPPORTS)
- FAIL: Evidence contradicts the claim (REFUTES)
- NA: Insufficient evidence or not enough information to make a decision (NOT ENOUGH INFO)

For the confidence, output a number between 0 and 1 with 2 decimal places. 0 being the lowest confidence and 1 being the highest confidence. This is a confidence score for your classification.

For the rationale, output an empty string. There is no need to explain your reasoning.

You must have the LABEL, CONF, and RATIONALE in your response. DO NOT hallucinate the keys for the response, e.g. STANCE instead of LABEL is incorrect. Or CONFIDENCE instead of CONF is incorrect.

Return NA ONLY if the provided evidence is genuinely insufficient to support or refute the claim. If the evidence explicitly supports the claim, choose PASS. If it explicitly contradicts the claim, choose FAIL.

CONF should represent your probability (0 to 1) that your chosen LABEL is correct. Use 2 decimal places.

<Response Format>
Return ONLY the following response format. Output exactly 3 lines and nothing else. Do not add any other text. Example:

LABEL=PASS
CONF=0.65
RATIONALE=

</Response Format>

Now begin your response. Do not write anything before LABEL.
LABEL=
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

async def sample_from_checkpoint(checkpoint_path: str, num_examples: int = None):
    """Sample from a trained checkpoint on FEVER evaluation examples."""
    
    # Get evaluation examples (same as classify_fever.py)
    SEED = 43  # Same seed as classify_fever.py to get the same examples
    print("Getting evaluation examples (excluding training set)...")
    examples = get_eval_examples(
        num_nei=500,
        num_supports=250,
        num_refutes=250,
        seed=SEED
    )
    
    if num_examples is not None:
        examples = examples[:num_examples]
    
    service_client = tinker.ServiceClient()
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)
    else:
        print(f"Using base model: {MODEL}")
        sampling_client = await service_client.create_sampling_client_async(base_model=MODEL)
    
    tokenizer_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = tokenizer_client.get_tokenizer()
    
    # Initialize WikiReader for evidence lookup
    wiki_reader = get_wiki_reader()
    
    async def classify_one(idx: int, ex: dict) -> dict:
        """Classify a single example asynchronously."""
        # Get evidence texts
        evidence_texts = wiki_reader.get_evidence_text(ex.get("evidence", []))
        
        # Convert golden label to simplified format
        golden_label = LABEL_MAP.get(ex["label"], ex["label"])
        
        prompt = build_prompt(ex["claim"], evidence_texts)
        tokens = tokenizer.encode(prompt)
        model_input = types.ModelInput.from_ints(tokens)
        params = types.SamplingParams(max_tokens=10000, temperature=0.1, stop=["\nRATIONALE="])
        
        result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        raw_response = tokenizer.decode(result.sequences[0].tokens)
        predicted_label, predicted_conf, valid = parse_output(raw_response)
        
        status = "✓" if predicted_label == golden_label else "✗"
        print(f"[{idx+1}/{len(examples)}] {status} Golden={golden_label} Pred={predicted_label} Conf={predicted_conf:.2f} Valid={valid} | {ex['claim'][:40]}...")
        
        return {
            "id": ex["id"],
            "claim": ex["claim"],
            "evidence_texts": evidence_texts,
            "golden_label": golden_label,
            "predicted_label": predicted_label,
            "predicted_conf": predicted_conf,
            "format_valid": valid,
            "raw_response": raw_response.strip()
        }
    
    print(f"Classifying {len(examples)} examples concurrently...")
    tasks = [classify_one(i, ex) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["golden_label"] == r["predicted_label"])
    accuracy = correct / len(results) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(results)})")
    
    output = {
        "model": MODEL,
        "source": "sample_grpo",
        "checkpoint": checkpoint_path or "base",
        "num_examples": len(results),
        "results": results
    }
    
    output_path = "grpo_sample_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "tinker://9d4915f4-ed1d-5df7-b60a-56bb4e50a284:train:0/sampler_weights/self-aware-grpo-trial-4"
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(sample_from_checkpoint(checkpoint, num_examples))
