"""Sample from trained GRPO checkpoint on FEVER examples."""
import asyncio, json
import tinker
from tinker import types

MODEL = "moonshotai/Kimi-K2-Thinking"

def build_prompt(claim):
    return f"""Claim: {claim}

Classify as NA (not enough info), PASS (supports), or FAIL (refutes).
Return EXACTLY:
LABEL=<NA|PASS|FAIL>
CONF=<float 0-1, 2 decimals>
RATIONALE=<one sentence or empty>
"""

async def sample_from_checkpoint(checkpoint_path: str, num_examples: int = 10):
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
    
    results = []
    for ex in examples:
        prompt = build_prompt(ex["claim"])
        tokens = tokenizer.encode(prompt)
        model_input = types.ModelInput.from_ints(tokens)
        params = types.SamplingParams(max_tokens=96, temperature=0.0, stop=["\n\n"])
        
        result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        response = tokenizer.decode(result.sequences[0].tokens)
        
        results.append({
            "id": ex["id"],
            "claim": ex["claim"],
            "evidence_texts": ex.get("evidence_texts", []),
            "golden_label": ex["golden_label"],
            "grpo_response": response.strip()
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

