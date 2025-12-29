"""Sample from trained GRPO checkpoint on FEVER examples.

This module provides utilities for sampling from a trained GRPO checkpoint
on FEVER fact verification examples. Uses the same prompt format as training
(with evidence texts from WikiReader) for consistency.
"""
import asyncio, json
import tinker
from tinker import types

from extract_fever_dataset import WikiReader, download_wiki_pages

MODEL = "moonshotai/Kimi-K2-Thinking"

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
    
    Uses the same format as training (train_grpo.py) for consistency.
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
- NA: Insufficient evidence (NOT ENOUGH INFO)

Respond with EXACTLY this format:
LABEL=<NA|PASS|FAIL>
CONF=<float 0-1, 2 decimals>
RATIONALE=<one sentence or empty>
"""


async def sample_from_checkpoint(checkpoint_path: str, num_examples: int = 10):
    """Sample from a trained checkpoint on FEVER examples with evidence."""
    with open("fever_classification_results.json", "r") as f:
        data = json.load(f)
    examples = data["results"][:num_examples]
    
    service_client = tinker.ServiceClient()
    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer_client = await service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
    tokenizer = tokenizer_client.get_tokenizer()
    
    # Initialize WikiReader for fresh evidence lookup
    wiki_reader = get_wiki_reader()
    
    results = []
    for ex in examples:
        # Use stored evidence_texts if available, otherwise they were already fetched
        evidence_texts = ex.get("evidence_texts", [])
        
        prompt = build_prompt(ex["claim"], evidence_texts)
        tokens = tokenizer.encode(prompt)
        model_input = types.ModelInput.from_ints(tokens)
        params = types.SamplingParams(max_tokens=2048, temperature=0.0)
        
        result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        output_ids = list(result.sequences[0].tokens)
        
        index = 0
        for i, tid in enumerate(output_ids):
            decoded_token = tokenizer.decode([tid], skip_special_tokens=False).strip()
            if decoded_token in ["</think>", "<|thought_end|>", "### Response:"]:
                index = i + 1
                break
                
        if index > 0:
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        else:
            full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            # Robust split logic
            match = re.search(r"(?:LABEL|STANCE|Classification)\s*[=:]", full_text, re.IGNORECASE)
            if match:
                split_idx = match.start()
                thinking = full_text[:split_idx].strip()
                content = full_text[split_idx:].strip()
            else:
                thinking = ""
                content = full_text.strip()
                
        results.append({
            "id": ex["id"],
            "claim": ex["claim"],
            "evidence_texts": evidence_texts,
            "golden_label": ex["golden_label"],
            "grpo_response": content,
            "thinking": thinking
        })
        print(f"ID {ex['id']}: Golden={ex['golden_label']}")
        print(f"  Response: {response.strip()[:100]}")
    
    output = {"model": MODEL, "checkpoint": checkpoint_path or "base", "results": results}
    with open("grpo_sample_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} results to grpo_sample_results.json")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    asyncio.run(sample_from_checkpoint(ckpt, 10))

