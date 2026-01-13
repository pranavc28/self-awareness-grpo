"""Sample from trained GRPO checkpoint on FEVER/VitaminC/ClimateFEVER examples.

IMPORTANT: This uses the SAME output format as `train_grpo.py` (LABEL/CONF/RATIONALE)
to avoid evaluation/prompt mismatch, which can otherwise make GRPO look worse than it is.

Usage:
    python sample_grpo.py [dataset] [checkpoint] [num_examples]
    
    dataset: fever (default), vitaminc, or climatefever
    checkpoint: Path to GRPO checkpoint (has default)
    num_examples: Optional limit on number of examples
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


def load_dataset(dataset: str = "fever", seed: int = 43):
    """
    Load examples from the specified dataset.
    
    Returns:
        tuple: (examples, needs_wiki) where needs_wiki indicates if WikiReader is needed
    """
    if dataset == "fever":
        return get_eval_examples(num_nei=500, num_supports=250, num_refutes=250, seed=seed), True
    elif dataset == "vitaminc":
        with open("vitaminc_eval_data.json", "r") as f:
            return json.load(f), False
    elif dataset == "climatefever":
        with open("climatefever_eval_data.json", "r") as f:
            return json.load(f), False
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use fever, vitaminc, or climatefever")


def build_prompt(claim: str, evidence_texts: list[str]) -> str:
    """Build the same generalizable NLI-based prompt format used during GRPO training."""
    if evidence_texts:
        evidence = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence = "No relevant evidence available."
    
    return f"""You are a fact verification expert. Classify whether a claim is supported, refuted, or has insufficient evidence.

Claim: {claim}

You have additional evidence to consider in case you need it:
{evidence}

Classify into exactly one category:
- PASS: Evidence supports the claim (SUPPORTS)
- FAIL: Evidence contradicts the claim (REFUTES)
- NA: Insufficient evidence or not enough information to make a decision (NOT ENOUGH INFO)

Return NA ONLY if the provided evidence is genuinely insufficient to support or refute the claim. If the evidence explicitly supports the claim, choose PASS. If it explicitly contradicts the claim, choose FAIL.

<Response Format>
Return ONLY the label. Output exactly one word (PASS, FAIL, or NA) and nothing else.

Example: PASS

</Response Format>

LABEL="""

def parse_output(text):
    """
    Parse model output to extract label and format validity.
    
    Extracts the label from the model's response text. The expected format is
    simply the label (NA, PASS, or FAIL).
    
    Args:
        text: Raw model output string.
    
    Returns:
        tuple: (label, valid) where:
            - label: Extracted label (NA/PASS/FAIL) or None if missing
            - valid: Boolean indicating if output format was correct
    """
    label, valid = None, True
    text = text.strip().upper()
    # Check if the output is exactly one of the valid labels
    if text in ("NA", "PASS", "FAIL"):
        label = text
    else:
        # Try to find a label anywhere in the text
        text_clean = text.replace("<", "").replace(">", "")
        m = re.search(r"\b(NA|PASS|FAIL)\b", text_clean, re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            valid = False  # Penalize non-exact format
        else:
            valid = False
    return label, valid

async def sample_from_checkpoint(checkpoint_path: str, dataset: str = "fever", num_examples: int = None):
    """Sample from a trained checkpoint on evaluation examples."""
    
    SEED = 43  # Same seed as sample_control.py to get the same examples
    print(f"Loading {dataset} dataset...")
    examples, needs_wiki = load_dataset(dataset, SEED)
    
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
    
    # Initialize WikiReader only if needed (for FEVER)
    wiki_reader = get_wiki_reader() if needs_wiki else None
    
    async def classify_one(idx: int, ex: dict) -> dict:
        """Classify a single example asynchronously."""
        # Get evidence based on dataset type
        if needs_wiki and wiki_reader:
            evidence_texts = wiki_reader.get_evidence_text(ex.get("evidence", []))
        else:
            # VitaminC/ClimateFever: evidence is already a string in the example
            ev = ex.get("evidence", "")
            evidence_texts = [ev] if ev else []
        
        # Convert golden label to simplified format
        golden_label = LABEL_MAP.get(ex.get("label", ""), ex.get("label", ""))
        
        prompt = build_prompt(ex["claim"], evidence_texts)
        tokens = tokenizer.encode(prompt)
        model_input = types.ModelInput.from_ints(tokens)
        params = types.SamplingParams(max_tokens=64, temperature=0.1, stop=["\nRATIONALE="])
        
        result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        raw_response = tokenizer.decode(result.sequences[0].tokens)
        predicted_label, valid = parse_output(raw_response)
        
        status = "✓" if predicted_label == golden_label else "✗"
        print(f"[{idx+1}/{len(examples)}] {status} Golden={golden_label} Pred={predicted_label} Valid={valid} | {ex['claim'][:40]}...")
        
        return {
            "id": ex.get("id", idx),
            "claim": ex["claim"],
            "evidence_texts": evidence_texts,
            "golden_label": golden_label,
            "predicted_label": predicted_label,
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
        "source": f"sample_grpo_{dataset}",
        "dataset": dataset,
        "checkpoint": checkpoint_path or "base",
        "num_examples": len(results),
        "results": results
    }
    
    output_path = f"{dataset}_grpo_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fever"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "tinker://8c58d982-9ba2-5469-afe0-5f184f9a7f7c:train:0/sampler_weights/self-aware-grpo-trial-4"
    num_examples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    asyncio.run(sample_from_checkpoint(checkpoint, dataset, num_examples))
