"""
Download and extract VitaminC dataset for fact verification.

VitaminC (Schuster et al., 2021) contains ~450K claim-evidence pairs from Wikipedia
with contrastive examples. Labels match FEVER: SUPPORTS, REFUTES, NOT ENOUGH INFO.

Paper: https://arxiv.org/abs/2104.08679
GitHub: https://github.com/TalSchuster/VitaminC
"""

import os
import json
import random
import requests
import zipfile
from tqdm import tqdm

# VitaminC data zip URL (from talschuster.github.io)
VITAMINC_ZIP_URL = "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip"

# Label mapping to match FEVER format
LABEL_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    # Some versions use different casing
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "not enough info": "NOT ENOUGH INFO",
    "NEI": "NOT ENOUGH INFO",
}


def download_and_extract_vitaminc(output_dir: str = "vitaminc_data") -> str:
    """Download and extract VitaminC dataset from GitHub."""
    os.makedirs(output_dir, exist_ok=True)
    
    # The zip extracts to a 'vitaminc' subdirectory
    extracted_dir = os.path.join(output_dir, "vitaminc")
    zip_path = os.path.join(output_dir, "vitaminc_data.zip")
    
    # Check if already extracted
    if os.path.exists(extracted_dir) and os.listdir(extracted_dir):
        print(f"VitaminC already extracted at {extracted_dir}")
        return extracted_dir
    
    # Download zip
    if not os.path.exists(zip_path):
        print(f"Downloading VitaminC from {VITAMINC_ZIP_URL}...")
        print("(This may take a few minutes - ~200MB)")
        response = requests.get(VITAMINC_ZIP_URL, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Extract
    print("Extracting VitaminC dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted to {extracted_dir}")
    return extracted_dir


def download_vitaminc(output_dir: str = "vitaminc_data", split: str = "dev"):
    """
    Download VitaminC dataset.
    
    Args:
        output_dir: Directory to save the data
        split: One of 'train', 'dev', 'test'
    
    Returns:
        List of examples with claim, evidence, label
    """
    if split not in ["train", "dev", "test"]:
        raise ValueError("Split must be one of train, dev, test")
    
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"vitaminc_{split}.json")
    
    # Check if already processed
    if os.path.exists(json_path):
        print(f"Loading from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Download and extract if needed
    extracted_dir = download_and_extract_vitaminc(output_dir)
    
    # Find the JSONL file - VitaminC uses train/dev/test.jsonl directly in the extracted folder
    jsonl_path = os.path.join(extracted_dir, f"{split}.jsonl")
    if not os.path.exists(jsonl_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(extracted_dir, "data", f"{split}.jsonl"),
            os.path.join(output_dir, f"{split}.jsonl"),
            os.path.join(output_dir, "vitaminc", f"{split}.jsonl"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                jsonl_path = alt
                break
    
    if not os.path.exists(jsonl_path):
        # List what we have for debugging
        print(f"Looking for {split}.jsonl...")
        print(f"Extracted dir contents: {os.listdir(extracted_dir) if os.path.exists(extracted_dir) else 'N/A'}")
        raise RuntimeError(f"Could not find {split}.jsonl in extracted data")
    
    # Parse JSONL
    print(f"Parsing {jsonl_path}...")
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if line.strip():
                try:
                    item = json.loads(line)
                    # VitaminC format: claim, evidence, label
                    label = item.get("label", "")
                    mapped_label = LABEL_MAP.get(label, LABEL_MAP.get(str(label).upper(), str(label)))
                    
                    examples.append({
                        "id": item.get("unique_id", i),
                        "claim": item.get("claim", ""),
                        "evidence": item.get("evidence", ""),  # Single evidence sentence
                        "label": mapped_label,
                        "page": item.get("page", ""),  # Wikipedia page title
                    })
                except json.JSONDecodeError:
                    continue
    
    print(f"Parsed {len(examples)} examples")
    
    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {json_path}")
    return examples


def get_balanced_sample(
    output_dir: str = "vitaminc_data",
    num_nei: int = 500,
    num_supports: int = 250,
    num_refutes: int = 250,
    seed: int = 42,
    split: str = "dev"
) -> list[dict]:
    """
    Get a balanced sample from VitaminC dataset.
    
    Args:
        output_dir: Directory to save/load the data
        num_nei: Number of NOT ENOUGH INFO examples
        num_supports: Number of SUPPORTS examples
        num_refutes: Number of REFUTES examples
        seed: Random seed for reproducibility
        split: Dataset split to use
    
    Returns:
        List of sampled examples
    """
    random.seed(seed)
    
    # Load or download
    examples = download_vitaminc(output_dir, split)
    
    # Group by label
    by_label = {
        "NOT ENOUGH INFO": [],
        "SUPPORTS": [],
        "REFUTES": []
    }
    
    for ex in examples:
        label = ex.get("label")
        if label in by_label:
            by_label[label].append(ex)
    
    print(f"\nAvailable: NEI={len(by_label['NOT ENOUGH INFO'])}, "
          f"SUPPORTS={len(by_label['SUPPORTS'])}, "
          f"REFUTES={len(by_label['REFUTES'])}")
    
    # Verify we have enough
    for label, count in [("NOT ENOUGH INFO", num_nei), ("SUPPORTS", num_supports), ("REFUTES", num_refutes)]:
        if len(by_label[label]) < count:
            print(f"Warning: Only {len(by_label[label])} {label} examples available, using all")
    
    # Sample from each
    sampled = []
    sampled.extend(random.sample(by_label["NOT ENOUGH INFO"], min(num_nei, len(by_label["NOT ENOUGH INFO"]))))
    sampled.extend(random.sample(by_label["SUPPORTS"], min(num_supports, len(by_label["SUPPORTS"]))))
    sampled.extend(random.sample(by_label["REFUTES"], min(num_refutes, len(by_label["REFUTES"]))))
    
    random.shuffle(sampled)
    
    print(f"Sampled {len(sampled)} examples")
    
    return sampled


def save_for_evaluation(
    output_path: str = "vitaminc_eval_data.json",
    num_nei: int = 500,
    num_supports: int = 250,
    num_refutes: int = 250,
    seed: int = 43
):
    """
    Create and save a balanced evaluation dataset.
    
    Args:
        output_path: Path to save the JSON file
        num_nei: Number of NOT ENOUGH INFO examples
        num_supports: Number of SUPPORTS examples
        num_refutes: Number of REFUTES examples
        seed: Random seed
    """
    sampled = get_balanced_sample(
        num_nei=num_nei,
        num_supports=num_supports,
        num_refutes=num_refutes,
        seed=seed
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(sampled)} examples to {output_path}")
    return sampled


def print_examples(examples: list, num: int = 5):
    """Print sample examples."""
    print(f"\n{'='*60}")
    print(f"SAMPLE EXAMPLES (showing {min(num, len(examples))} of {len(examples)})")
    print('='*60)
    
    for i, ex in enumerate(examples[:num]):
        print(f"\n--- Example {i+1} ---")
        print(f"ID: {ex.get('id')}")
        print(f"Claim: {ex.get('claim')}")
        print(f"Label: {ex.get('label')}")
        print(f"Evidence: {ex.get('evidence')[:200]}..." if len(ex.get('evidence', '')) > 200 else f"Evidence: {ex.get('evidence')}")
        if ex.get('page'):
            print(f"Wikipedia Page: {ex.get('page')}")


def print_stats(examples: list):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print('='*60)
    
    # Label distribution
    labels = {}
    for ex in examples:
        label = ex.get("label", "UNKNOWN")
        labels[label] = labels.get(label, 0) + 1
    
    print("\nLabel Distribution:")
    for label, count in sorted(labels.items()):
        pct = count / len(examples) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Evidence length stats
    evidence_lens = [len(ex.get("evidence", "")) for ex in examples]
    print(f"\nEvidence Length:")
    print(f"  Min: {min(evidence_lens)} chars")
    print(f"  Max: {max(evidence_lens)} chars")
    print(f"  Avg: {sum(evidence_lens)/len(evidence_lens):.0f} chars")


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    num_nei = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    num_supports = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    num_refutes = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    
    # Download and sample
    sampled = get_balanced_sample(
        num_nei=num_nei,
        num_supports=num_supports,
        num_refutes=num_refutes
    )
    
    # Print examples and stats
    print_examples(sampled, num=5)
    print_stats(sampled)
    
    # Save for evaluation
    save_for_evaluation(
        output_path="vitaminc_eval_data.json",
        num_nei=num_nei,
        num_supports=num_supports,
        num_refutes=num_refutes
    )

