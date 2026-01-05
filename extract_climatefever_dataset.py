"""
Download and extract Climate-FEVER dataset for fact verification.

Climate-FEVER (Diggelmann et al., 2020) contains 1,535 real-world climate claims
with Wikipedia evidence. Labels match FEVER: SUPPORTS, REFUTES, NOT_ENOUGH_INFO.

Paper: https://arxiv.org/abs/2012.00614
GitHub: https://github.com/tdiggelm/climate-fever-dataset
"""

import os
import json
import random
import requests
from tqdm import tqdm

# Climate-FEVER data URL (from official GitHub)
CLIMATEFEVER_URL = "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl"

# Label mapping to match FEVER format
LABEL_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
    "DISPUTED": "NOT ENOUGH INFO",  # Map disputed to NEI for consistency
    # Handle different casings
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "not_enough_info": "NOT ENOUGH INFO",
}


def download_file(url: str, output_path: str) -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_climatefever(output_dir: str = "climatefever_data"):
    """
    Download Climate-FEVER dataset.
    
    Args:
        output_dir: Directory to save the data
    
    Returns:
        List of examples with claim, evidence, label
    """
    os.makedirs(output_dir, exist_ok=True)
    
    jsonl_path = os.path.join(output_dir, "climate-fever.jsonl")
    json_path = os.path.join(output_dir, "climate-fever.json")
    
    # Check if already processed
    if os.path.exists(json_path):
        print(f"Loading from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Download if needed
    if not os.path.exists(jsonl_path):
        success = download_file(CLIMATEFEVER_URL, jsonl_path)
        if not success:
            raise RuntimeError("Failed to download Climate-FEVER")
    
    # Parse JSONL
    print(f"Parsing {jsonl_path}...")
    examples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if line.strip():
                try:
                    item = json.loads(line)
                    claim_id = item.get("claim_id", i)
                    claim = item.get("claim", "")
                    claim_label = item.get("claim_label", "")
                    
                    # Map label to FEVER format
                    mapped_label = LABEL_MAP.get(claim_label, LABEL_MAP.get(claim_label.upper(), claim_label))
                    
                    # Collect all evidence sentences
                    evidences = item.get("evidences", [])
                    evidence_texts = []
                    for ev in evidences:
                        ev_text = ev.get("evidence", "")
                        ev_label = ev.get("evidence_label", "")
                        article = ev.get("article", "")
                        if ev_text:
                            evidence_texts.append(f"[{article}] {ev_text}")
                    
                    # Combine evidence into single string (like FEVER format)
                    combined_evidence = "\n".join(evidence_texts) if evidence_texts else ""
                    
                    examples.append({
                        "id": claim_id,
                        "claim": claim,
                        "evidence": combined_evidence,
                        "evidence_list": evidences,  # Keep original structure too
                        "label": mapped_label,
                    })
                except json.JSONDecodeError:
                    continue
    
    print(f"Parsed {len(examples)} claims")
    
    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {json_path}")
    return examples


def get_balanced_sample(
    output_dir: str = "climatefever_data",
    num_nei: int = 500,
    num_supports: int = 250,
    num_refutes: int = 250,
    seed: int = 42
) -> list[dict]:
    """
    Get a balanced sample from Climate-FEVER dataset.
    
    Note: Climate-FEVER is smaller (~1.5K claims), so sample sizes may be limited.
    
    Args:
        output_dir: Directory to save/load the data
        num_nei: Number of NOT ENOUGH INFO examples
        num_supports: Number of SUPPORTS examples
        num_refutes: Number of REFUTES examples
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled examples
    """
    random.seed(seed)
    
    # Load or download
    examples = download_climatefever(output_dir)
    
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
    
    # Sample from each (use min to handle small dataset)
    sampled = []
    
    nei_count = min(num_nei, len(by_label["NOT ENOUGH INFO"]))
    sup_count = min(num_supports, len(by_label["SUPPORTS"]))
    ref_count = min(num_refutes, len(by_label["REFUTES"]))
    
    if nei_count > 0:
        sampled.extend(random.sample(by_label["NOT ENOUGH INFO"], nei_count))
    if sup_count > 0:
        sampled.extend(random.sample(by_label["SUPPORTS"], sup_count))
    if ref_count > 0:
        sampled.extend(random.sample(by_label["REFUTES"], ref_count))
    
    random.shuffle(sampled)
    
    print(f"Sampled {len(sampled)} examples: {nei_count} NEI, {sup_count} SUPPORTS, {ref_count} REFUTES")
    
    return sampled


def get_all_examples(output_dir: str = "climatefever_data") -> list[dict]:
    """
    Get all examples from Climate-FEVER (useful since dataset is small).
    
    Args:
        output_dir: Directory to save/load the data
    
    Returns:
        All examples from the dataset
    """
    examples = download_climatefever(output_dir)
    
    # Filter to only valid labels
    valid_labels = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    filtered = [ex for ex in examples if ex.get("label") in valid_labels]
    
    print(f"Returning all {len(filtered)} valid examples")
    return filtered


def save_for_evaluation(
    output_path: str = "climatefever_eval_data.json",
    use_all: bool = True,
    num_nei: int = 500,
    num_supports: int = 250,
    num_refutes: int = 250,
    seed: int = 43
):
    """
    Create and save evaluation dataset.
    
    Args:
        output_path: Path to save the JSON file
        use_all: If True, use all examples (recommended for small dataset)
        num_nei: Number of NOT ENOUGH INFO examples (if not using all)
        num_supports: Number of SUPPORTS examples (if not using all)
        num_refutes: Number of REFUTES examples (if not using all)
        seed: Random seed
    """
    if use_all:
        sampled = get_all_examples()
    else:
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
        evidence = ex.get('evidence', '')
        if len(evidence) > 300:
            print(f"Evidence: {evidence[:300]}...")
        else:
            print(f"Evidence: {evidence}")


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
    
    # Claim length stats
    claim_lens = [len(ex.get("claim", "")) for ex in examples]
    print(f"\nClaim Length:")
    print(f"  Min: {min(claim_lens)} chars")
    print(f"  Max: {max(claim_lens)} chars")
    print(f"  Avg: {sum(claim_lens)/len(claim_lens):.0f} chars")


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    use_all = "--all" in sys.argv or "-a" in sys.argv
    
    if use_all:
        # Use all examples (recommended for this small dataset)
        examples = get_all_examples()
    else:
        # Get balanced sample with custom counts
        num_nei = 500
        num_supports = 250
        num_refutes = 250
        
        for i, arg in enumerate(sys.argv[1:]):
            if arg.isdigit():
                if i == 0:
                    num_nei = int(arg)
                elif i == 1:
                    num_supports = int(arg)
                elif i == 2:
                    num_refutes = int(arg)
        
        examples = get_balanced_sample(
            num_nei=num_nei,
            num_supports=num_supports,
            num_refutes=num_refutes
        )
    
    # Print examples and stats
    print_examples(examples, num=5)
    print_stats(examples)
    
    # Save for evaluation
    save_for_evaluation(
        output_path="climatefever_eval_data.json",
        use_all=use_all
    )

