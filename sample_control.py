"""Classify FEVER/VitaminC/ClimateFEVER examples using openai/gpt-oss-20b via Tinker API.

IMPORTANT: This uses the SAME output format as `train_grpo.py` (LABEL/CONF/RATIONALE)
to avoid evaluation/prompt mismatch when comparing to GRPO checkpoints.

Usage:
    python sample_control.py [dataset]
    
    dataset: fever (default), vitaminc, or climatefever
"""

import asyncio
import json
import random
import sys
from typing import Optional

import tinker
from tinker import types
from transformers import AutoTokenizer

from extract_fever_dataset import get_eval_examples, WikiReader, download_wiki_pages

# Label mapping: FEVER labels -> simplified labels
LABEL_MAP = {"NOT ENOUGH INFO": "NA", "SUPPORTS": "PASS", "REFUTES": "FAIL"}
REVERSE_LABEL_MAP = {"NA": "NOT ENOUGH INFO", "PASS": "SUPPORTS", "FAIL": "REFUTES"}


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
    def parse_output(self, text: str) -> tuple[str, bool]:
        """Parse label from the model output."""
        import re

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
        return (label or "NA"), valid
    
    async def classify_single(self, claim: str, evidence_texts: list[str]) -> tuple[str, str]:
        """Classify a single claim asynchronously."""
        prompt = self.build_prompt(claim, evidence_texts)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(prompt_tokens)
        
        sampling_params = types.SamplingParams(
            max_tokens=64, temperature=0.1, stop=["\nRATIONALE="]
        )
        
        result = await self.sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        
        raw_response = self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        label, valid = self.parse_output(raw_response)
        return label, raw_response
    
    async def classify_batch(self, examples: list[dict], wiki_reader: Optional[WikiReader], dataset: str) -> list[dict]:
        """Classify a batch of examples concurrently."""
        
        async def classify_one(idx: int, example: dict) -> dict:
            # Get evidence based on dataset type
            if dataset == "fever" and wiki_reader:
                evidence_texts = wiki_reader.get_evidence_text(example.get("evidence", []))
            else:
                # VitaminC/ClimateFever: evidence is already a string in the example
                ev = example.get("evidence", "")
                evidence_texts = [ev] if ev else []
            
            predicted, raw = await self.classify_single(example["claim"], evidence_texts)
            
            # Convert golden label to simplified format
            golden_label = LABEL_MAP.get(example.get("label", ""), example.get("label", ""))
            
            status = "✓" if predicted == golden_label else "✗"
            print(f"[{idx+1}/{len(examples)}] {status} {example['claim'][:50]}...")
            
            return {
                "id": example.get("id", idx),
                "claim": example["claim"],
                "evidence_texts": evidence_texts,
                "golden_label": golden_label,
                "predicted_label": predicted,
                "raw_response": raw.strip()
            }
        
        tasks = [classify_one(i, ex) for i, ex in enumerate(examples)]
        return await asyncio.gather(*tasks)


async def main(dataset: str = "fever"):
    SEED = 43  # Different from training seed (42) to get different examples
    random.seed(SEED)
    
    print(f"Loading {dataset} dataset...")
    samples, needs_wiki = load_dataset(dataset, SEED)
    
    wiki_reader = None
    if needs_wiki:
        print("Loading Wikipedia evidence...")
        wiki_reader = WikiReader(download_wiki_pages())
    
    print("Initializing openai/gpt-oss-20b...")
    classifier = FEVERClassifier()
    await classifier.initialize()
    
    print(f"Classifying {len(samples)} examples...")
    results = await classifier.classify_batch(samples, wiki_reader, dataset)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["golden_label"] == r["predicted_label"])
    accuracy = correct / len(results) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(results)})")
    
    output = {
        "model": "openai/gpt-oss-20b",
        "source": f"sample_control_{dataset}",
        "dataset": dataset,
        "num_examples": len(results),
        "results": results
    }
    
    output_path = f"{dataset}_control_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fever"
    asyncio.run(main(dataset))
