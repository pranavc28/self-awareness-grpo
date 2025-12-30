"""Sample from trained GRPO checkpoint on FEVER examples.

This module provides utilities for sampling from a trained GRPO checkpoint
on FEVER fact verification examples. Uses evaluation examples that are NOT
in the training set, with the same prompt format as classify_fever.py.
"""
import asyncio
import json
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
    """
    Build the classification prompt with claim and evidence context.
    
    Uses the same simplified format as classify_fever.py (label only output).
    """
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

It is ok to return NA if the evidence is insufficient to make a decision. Do not make up evidence that is not provided.

Respond with ONLY one word: PASS, FAIL, or NA

Classification:"""


def parse_classification(response: str) -> str:
    """Extract classification label from model response."""
    response_upper = response.upper().strip()
    
    # Check for simplified labels first
    if "NA" in response_upper:
        return "NA"
    elif "FAIL" in response_upper:
        return "FAIL"
    elif "PASS" in response_upper:
        return "PASS"
    # Fallback to original labels
    elif "NOT ENOUGH INFO" in response_upper:
        return "NA"
    elif "REFUTES" in response_upper:
        return "FAIL"
    elif "SUPPORTS" in response_upper:
        return "PASS"
    return "NA"  # Default


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
        sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    else:
        print(f"Using base model: {MODEL}")
        sampling_client = service_client.create_sampling_client(base_model=MODEL)
    
    tokenizer_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = tokenizer_client.get_tokenizer()
    
    # Initialize WikiReader for evidence lookup
    wiki_reader = get_wiki_reader()
    
    results = []
    correct = 0
    
    for idx, ex in enumerate(examples):
        # Get evidence texts
        evidence_texts = wiki_reader.get_evidence_text(ex.get("evidence", []))
        
        # Convert golden label to simplified format
        golden_label = LABEL_MAP.get(ex["label"], ex["label"])
        
        prompt = build_prompt(ex["claim"], evidence_texts)
        tokens = tokenizer.encode(prompt)
        model_input = types.ModelInput.from_ints(tokens)
        params = types.SamplingParams(max_tokens=10, temperature=0.0, stop=["\n", "\n\n"])
        
        result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        raw_response = tokenizer.decode(result.sequences[0].tokens)
        predicted_label = parse_classification(raw_response)
        
        if predicted_label == golden_label:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        results.append({
            "id": ex["id"],
            "claim": ex["claim"],
            "evidence_texts": evidence_texts,
            "golden_label": golden_label,
            "predicted_label": predicted_label,
            "raw_response": raw_response.strip()
        })
        
        print(f"[{idx+1}/{len(examples)}] {status} Golden={golden_label} Pred={predicted_label} | {ex['claim'][:40]}...")
    
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
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else ""
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(sample_from_checkpoint(checkpoint, num_examples))
