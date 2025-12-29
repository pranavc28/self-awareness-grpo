"""Classify FEVER examples using Kimi-K2 via Tinker API with async sampling."""

import asyncio
import json
import random
from typing import Optional

import tinker
from tinker import types
from transformers import AutoTokenizer

from extract_fever_dataset import get_balanced_sample, WikiReader, download_wiki_pages

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class FEVERClassifier:
    """FEVER fact verification classifier using Tinker API."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
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
        """Build classification prompt with claim and evidence."""
        if evidence_texts:
            evidence = "\n".join(f"- {t}" for t in evidence_texts)
        else:
            evidence = "No specific evidence provided."
        
        return f"""You are a fact verification expert. Classify whether a claim is supported, refuted, or has insufficient evidence.

Claim: {claim}

You have additional evidence to consider in case you need it:
{evidence}

Classify into exactly one category:
- SUPPORTS: Evidence supports the claim
- REFUTES: Evidence contradicts the claim  
- NOT ENOUGH INFO: Insufficient evidence

Respond with ONLY: SUPPORTS, REFUTES, or NOT ENOUGH INFO. There is no need to explain your reasoning, just return the classification.

It is ok to return NOT ENOUGH INFO if the evidence is insufficient to make a decision. Do not make up evidence that is not provided.

Classification:"""
    
    def parse_classification(self, response: str) -> str:
        """Extract classification label from model response."""
        response_upper = response.upper().strip()
        
        if "NOT ENOUGH INFO" in response_upper:
            return "NOT ENOUGH INFO"
        elif "REFUTES" in response_upper:
            return "REFUTES"
        elif "SUPPORTS" in response_upper:
            return "SUPPORTS"
        return "NOT ENOUGH INFO"
    
    async def classify_single(self, claim: str, evidence_texts: list[str]) -> tuple[str, str]:
        """Classify a single claim asynchronously."""
        prompt = self.build_prompt(claim, evidence_texts)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(prompt_tokens)
        
        sampling_params = types.SamplingParams(
            max_tokens=2048, temperature=0.0
        )
        
        result = await self.sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        
        output_ids = list(result.sequences[0].tokens)
        index = 0
        for i, tid in enumerate(output_ids):
            decoded_token = self.tokenizer.decode([tid], skip_special_tokens=False).strip()
            if decoded_token in ["</think>", "<|thought_end|>", "### Response:"]:
                index = i + 1
                break
        
        if index > 0:
            thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        else:
            full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            # Robust split logic
            match = re.search(r"(?:Classification|LABEL|STANCE)\s*[=:]", full_text, re.IGNORECASE)
            if match:
                split_idx = match.start()
                thinking = full_text[:split_idx].strip()
                content = full_text[split_idx:].strip()
            else:
                thinking = ""
                content = full_text.strip()
                
        return self.parse_classification(content), content
    
    async def classify_batch(self, examples: list[dict], wiki_reader: WikiReader) -> list[dict]:
        """Classify a batch of examples concurrently."""
        
        async def classify_one(idx: int, example: dict) -> dict:
            evidence_texts = wiki_reader.get_evidence_text(example.get("evidence", []))
            predicted, raw = await self.classify_single(example["claim"], evidence_texts)
            
            status = "✓" if predicted == example["label"] else "✗"
            print(f"[{idx+1}/{len(examples)}] {status} {example['claim'][:50]}...")
            
            return {
                "id": example["id"],
                "claim": example["claim"],
                "evidence_texts": evidence_texts,
                "golden_label": example["label"],
                "predicted_label": predicted,
                "raw_response": raw.strip()
            }
        
        tasks = [classify_one(i, ex) for i, ex in enumerate(examples)]
        return await asyncio.gather(*tasks)


async def main():
    SEED = 42
    random.seed(SEED)
    
    print("Loading Wikipedia evidence...")
    wiki_reader = WikiReader(download_wiki_pages())
    
    print("Sampling FEVER examples...")
    samples = get_balanced_sample(num_nei=10, num_supports=5, num_refutes=5, seed=SEED)
    
    print("Initializing Kimi-K2...")
    classifier = FEVERClassifier()
    await classifier.initialize()
    
    print(f"Classifying {len(samples)} examples...")
    results = await classifier.classify_batch(samples, wiki_reader)
    
    output = {"model": "moonshotai/Kimi-K2-Thinking", "results": results}
    with open("fever_classification_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to fever_classification_results.json")


if __name__ == "__main__":
    asyncio.run(main())
