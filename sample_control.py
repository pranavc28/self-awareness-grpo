"""Classify FEVER/VitaminC/ClimateFEVER examples using FORMAT-ONLY baseline checkpoint.

This uses the format-only GRPO checkpoint as a baseline comparison.
The model learned output format but NOT accuracy optimization.

Usage:
    python sample_control.py [dataset] [checkpoint]
    
    dataset: fever (default), vitaminc, or climatefever
    checkpoint: Path to format-only checkpoint (has default)
"""

import asyncio
import json
import random
import sys
from typing import Optional

import tinker
from tinker import types

from extract_fever_dataset import get_eval_examples, WikiReader, download_wiki_pages

# Default checkpoint from train_grpo_format_only.py
DEFAULT_CHECKPOINT = "tinker://a61cf45a-6366-58b9-b0c6-6cd0a11e54ca:train:0/sampler_weights/grpo-format-only-baseline"
MODEL = "openai/gpt-oss-20b"

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
    
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT):
        self.checkpoint_path = checkpoint_path
        self.service_client = tinker.ServiceClient()
        self.sampling_client: Optional[tinker.SamplingClient] = None
        self.tokenizer = None
        
    async def initialize(self):
        """Initialize the sampling client from checkpoint."""
        if self.checkpoint_path:
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self.sampling_client = await self.service_client.create_sampling_client_async(
                model_path=self.checkpoint_path
            )
        else:
            print(f"Using base model: {MODEL}")
            self.sampling_client = await self.service_client.create_sampling_client_async(
                base_model=MODEL
            )
        # Get tokenizer from training client
        training_client = await self.service_client.create_lora_training_client_async(base_model=MODEL, rank=32)
        self.tokenizer = training_client.get_tokenizer()
    
    def build_prompt(self, claim: str, evidence_texts: list[str]) -> str:
        """Build the same prompt format used during GRPO training."""
        if evidence_texts:
            evidence = "\n".join(f"- {t}" for t in evidence_texts)
        else:
            evidence = "No specific evidence provided."
        
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

Claim: {claim}

Evidence:
{evidence}

Think: Does the evidence clearly support or contradict this specific claim? If unclear and confident in being unclear, say NA. Respond with only one of the following labels: NA, PASS, or FAIL.

Do not have question marks in your answer and always respond with a single label.

For example it is incorrect to reply with ??? or any other text. Your assessment must be one of (LABEL=NA, LABEL=PASS, or LABEL=FAIL):

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
    
    async def classify_single(self, claim: str, evidence_texts: list[str]) -> tuple[str, str, bool]:
        """Classify a single claim asynchronously."""
        prompt = self.build_prompt(claim, evidence_texts)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(prompt_tokens)
        
        sampling_params = types.SamplingParams(
            max_tokens=21000, temperature=0.1, stop=["\n"]
        )
        
        result = await self.sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        
        raw_response = self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        label, valid = self.parse_output(raw_response)
        return label, raw_response, valid
    
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
            
            predicted, raw, valid = await self.classify_single(example["claim"], evidence_texts)
            
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
                "format_valid": valid,
                "raw_response": raw.strip()
            }
        
        tasks = [classify_one(i, ex) for i, ex in enumerate(examples)]
        return await asyncio.gather(*tasks)


async def main(dataset: str = "fever", checkpoint: str = DEFAULT_CHECKPOINT):
    SEED = 43  # Different from training seed (42) to get different examples
    random.seed(SEED)
    
    print(f"Loading {dataset} dataset...")
    samples, needs_wiki = load_dataset(dataset, SEED)
    
    wiki_reader = None
    if needs_wiki:
        print("Loading Wikipedia evidence...")
        wiki_reader = WikiReader(download_wiki_pages())
    
    print("Initializing format-only baseline...")
    classifier = FEVERClassifier(checkpoint_path=checkpoint)
    await classifier.initialize()
    
    print(f"Classifying {len(samples)} examples...")
    results = await classifier.classify_batch(samples, wiki_reader, dataset)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["golden_label"] == r["predicted_label"])
    accuracy = correct / len(results) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(results)})")
    
    output = {
        "model": MODEL,
        "source": f"sample_control_{dataset}",
        "dataset": dataset,
        "checkpoint": checkpoint,
        "num_examples": len(results),
        "results": results
    }
    
    output_path = f"{dataset}_control_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fever"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHECKPOINT
    asyncio.run(main(dataset, checkpoint))
