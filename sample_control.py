"""Classify FEVER examples using openai/gpt-oss-20b via Tinker API with async sampling.

IMPORTANT: This uses the SAME output format as `train_grpo.py` (LABEL/CONF/RATIONALE)
to avoid evaluation/prompt mismatch when comparing to GRPO checkpoints.
"""

import asyncio
import json
import random
from typing import Optional

import tinker
from tinker import types
from transformers import AutoTokenizer

from extract_fever_dataset import get_eval_examples, WikiReader, download_wiki_pages

# Label mapping: FEVER labels -> simplified labels
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}
REVERSE_LABEL_MAP = {"NA": "NOT ENOUGH INFO", "PASS": "SUPPORTS", "FAIL": "REFUTES"}


class FEVERClassifier:
    """FEVER fact verification classifier using Tinker API."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.service_client = tinker.ServiceClient()
        self.sampling_client: Optional[tinker.SamplingClient] = None
        self.tokenizer = None
        
    async def initialize(self):
        """Initialize the sampling client directly from base model."""
        self.sampling_client = await self.service_client.create_sampling_client_async(
            base_model=self.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    
    def build_prompt(self, claim: str, evidence_texts: list[str]) -> str:
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

Return NA ONLY if the provided evidence is genuinely insufficient to support or refute the claim. If the evidence explicitly supports the claim, choose PASS. If it explicitly contradicts the claim, choose FAIL.

Return only the label as N

<Response Format>
Return ONLY the following response format. Do not add any other text:
LABEL=PASS
CONF=0.65
RATIONALE=
</Response Format>
"""
    
    def parse_output(self, text: str) -> tuple[str, float, bool]:
        """Parse LABEL/CONF from the model output."""
        import re

        label, conf, valid = None, 0.5, True
        text = text.replace("<", "").replace(">", "")
        m = re.search(r"LABEL\s*=\s*(NA|PASS|FAIL)", text, re.IGNORECASE)
        if m:
            label = m.group(1).upper()
        else:
            print()
            valid = False
        m = re.search(r"CONF\s*=\s*([\d.]+)", text)
        if m:
            try:
                conf = float(m.group(1))
                conf = max(0.0, min(1.0, conf))
            except Exception:
                valid = False
        else:
            valid = False
        return (label or "NA"), conf, valid
    
    async def classify_single(self, claim: str, evidence_texts: list[str]) -> tuple[str, str]:
        """Classify a single claim asynchronously."""
        prompt = self.build_prompt(claim, evidence_texts)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(prompt_tokens)
        
        sampling_params = types.SamplingParams(
            max_tokens=10000, temperature=0.1, stop=["RATIONALE="]
        )
        
        result = await self.sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        
        raw_response = self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        label, conf, valid = self.parse_output(raw_response)
        return label, raw_response
    
    async def classify_batch(self, examples: list[dict], wiki_reader: WikiReader) -> list[dict]:
        """Classify a batch of examples concurrently."""
        
        async def classify_one(idx: int, example: dict) -> dict:
            evidence_texts = wiki_reader.get_evidence_text(example.get("evidence", []))
            predicted, raw = await self.classify_single(example["claim"], evidence_texts)
            
            # Convert golden label to simplified format
            golden_label = LABEL_MAP.get(example["label"], example["label"])
            
            status = "✓" if predicted == golden_label else "✗"
            print(f"[{idx+1}/{len(examples)}] {status} {example['claim'][:50]}...")
            
            return {
                "id": example["id"],
                "claim": example["claim"],
                "evidence_texts": evidence_texts,
                "golden_label": golden_label,
                "predicted_label": predicted,
                "raw_response": raw.strip()
            }
        
        tasks = [classify_one(i, ex) for i, ex in enumerate(examples)]
        return await asyncio.gather(*tasks)


async def main():
    SEED = 43  # Different from training seed (42) to get different examples
    random.seed(SEED)
    
    print("Loading Wikipedia evidence...")
    wiki_reader = WikiReader(download_wiki_pages())
    
    print("Getting evaluation examples (excluding training set)...")
    samples = get_eval_examples(
        num_nei=500,
        num_supports=250,
        num_refutes=250,
        seed=SEED
    )
    
    print("Initializing openai/gpt-oss-20b...")
    classifier = FEVERClassifier()
    await classifier.initialize()
    
    print(f"Classifying {len(samples)} examples...")
    results = await classifier.classify_batch(samples, wiki_reader)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["golden_label"] == r["predicted_label"])
    accuracy = correct / len(results) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(results)})")
    
    output = {
        "model": "openai/gpt-oss-20b",
        "source": "classify_fever",
        "num_examples": len(results),
        "results": results
    }
    
    output_path = "fever_classification_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
