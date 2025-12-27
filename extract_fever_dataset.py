"""
Download FEVER v1.0 dataset and Wikipedia pages for fact verification.
"""

import requests
import json
import os
import random
import zipfile
from tqdm import tqdm

FEVER_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "wiki": "https://fever.ai/download/fever/wiki-pages.zip",
}


def download_file(url: str, output_path: str) -> bool:
    """Download a file with progress bar."""
    print(f"Downloading {url}...")
    print("(This is ~6GB and may take 10-30 minutes)")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=65536):  # Larger chunks for speed
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Verify file size
    actual_size = os.path.getsize(output_path)
    if total_size > 0 and actual_size < total_size:
        print(f"WARNING: Download incomplete ({actual_size}/{total_size} bytes)")
        return False
    return True


def is_valid_zip(zip_path: str) -> bool:
    """Check if a zip file is valid."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Test the zip file integrity
            bad_file = zf.testzip()
            return bad_file is None
    except (zipfile.BadZipFile, Exception):
        return False


def download_wiki_pages(output_dir: str = "fever_data"):
    """
    Download and extract FEVER Wikipedia pages (~6GB compressed).
    
    The wiki dump contains jsonl files where each line has:
    - "id": Wikipedia page title
    - "lines": Numbered sentences (format: "0\tFirst sentence.\n1\tSecond sentence.")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    wiki_dir = os.path.join(output_dir, "wiki-pages")
    zip_path = os.path.join(output_dir, "wiki-pages.zip")
    
    # Check if already extracted
    if os.path.exists(wiki_dir) and os.listdir(wiki_dir):
        print(f"Wiki pages already exist at {wiki_dir}")
        return wiki_dir
    
    # Check if zip exists and is valid
    if os.path.exists(zip_path):
        if not is_valid_zip(zip_path):
            print(f"Existing zip file is corrupted, deleting...")
            os.remove(zip_path)
    
    # Download if needed
    if not os.path.exists(zip_path):
        success = download_file(FEVER_URLS["wiki"], zip_path)
        if not success:
            raise RuntimeError("Failed to download wiki pages")
    
    # Validate before extracting
    if not is_valid_zip(zip_path):
        raise RuntimeError("Downloaded zip file is corrupted. Please try again.")
    
    # Extract
    print("Extracting wiki pages (this may take a while)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted to {wiki_dir}")
    return wiki_dir


class WikiReader:
    """
    Reader for FEVER Wikipedia pages.
    Loads pages on-demand and caches them in memory.
    """
    
    def __init__(self, wiki_dir: str = "fever_data/wiki-pages"):
        self.wiki_dir = wiki_dir
        self.pages = {}  # Cache: title -> {"lines": [...]}
        self._index = None  # title -> (file, line_offset)
        
    def _build_index(self):
        """Build an index of which file contains which page."""
        if self._index is not None:
            return
            
        print("Building wiki page index (first time only)...")
        self._index = {}
        
        wiki_subdir = os.path.join(self.wiki_dir, "wiki-pages")
        if os.path.exists(wiki_subdir):
            self.wiki_dir = wiki_subdir
        
        for filename in tqdm(sorted(os.listdir(self.wiki_dir))):
            if not filename.endswith('.jsonl'):
                continue
            filepath = os.path.join(self.wiki_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        page = json.loads(line)
                        title = page.get("id")
                        if title:
                            self._index[title] = (filepath, page)
        
        print(f"Indexed {len(self._index)} pages")
    
    def get_page(self, title: str) -> dict | None:
        """Get a Wikipedia page by title."""
        if title in self.pages:
            return self.pages[title]
        
        self._build_index()
        
        if title not in self._index:
            return None
        
        _, page = self._index[title]
        
        # Parse lines into sentences
        lines_raw = page.get("lines", "")
        sentences = {}
        for line in lines_raw.split("\n"):
            if "\t" in line:
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    sentences[int(parts[0])] = parts[1]
        
        parsed = {
            "title": title,
            "sentences": sentences,
            "text": page.get("text", "")
        }
        self.pages[title] = parsed
        return parsed
    
    def get_sentence(self, title: str, sentence_idx: int) -> str | None:
        """Get a specific sentence from a Wikipedia page."""
        page = self.get_page(title)
        if page is None:
            return None
        return page["sentences"].get(sentence_idx)
    
    def get_evidence_text(self, evidence: list) -> list[str]:
        """
        Get text for evidence items.
        Evidence format: [[annotation_id, evidence_id, page_title, sentence_idx], ...]
        """
        texts = []
        for evidence_group in evidence:
            for item in evidence_group:
                if len(item) >= 4 and item[2] is not None and item[3] is not None:
                    title = item[2]
                    sent_idx = item[3]
                    text = self.get_sentence(title, sent_idx)
                    if text:
                        texts.append(f"[{title}] {text}")
        return texts


def download_fever(output_dir: str = "fever_data", split: str = "dev"):
    """
    Download FEVER v1.0 dataset.
    
    Args:
        output_dir: Directory to save the data
        split: One of 'train', 'dev', 'test'
    """
    if split not in ["train", "dev", "test"]:
        raise ValueError("Split must be one of train, dev, test")
    
    print(f"Downloading FEVER {split} dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    jsonl_path = os.path.join(output_dir, f"fever_{split}.jsonl")
    
    # Download
    print(f"Downloading from {FEVER_URLS[split]}...")
    response = requests.get(FEVER_URLS[split], stream=True, timeout=60)
    response.raise_for_status()
    
    with open(jsonl_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Parse
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                examples.append({
                    "id": item.get("id"),
                    "claim": item.get("claim"),
                    "label": item.get("label"),
                    "verifiable": item.get("verifiable"),
                    "evidence": item.get("evidence", [])
                })
    
    print(f"Parsed {len(examples)} examples")
    
    # Save as JSON
    output_path = os.path.join(output_dir, f"fever_{split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")
    return examples


def get_balanced_sample(
    output_dir: str = "fever_data",
    num_nei: int = 500,
    num_supports: int = 250,
    num_refutes: int = 250,
    seed: int = 42
) -> list[dict]:
    """
    Download dev set and return a balanced sample of examples.
    
    Args:
        output_dir: Directory to save/load the data
        num_nei: Number of NOT ENOUGH INFO examples
        num_supports: Number of SUPPORTS examples
        num_refutes: Number of REFUTES examples
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled examples with id, claim, label
    """
    random.seed(seed)
    
    # Load or download dev set (has labels, unlike test)
    json_path = os.path.join(output_dir, "fever_dev.json")
    
    if os.path.exists(json_path):
        print(f"Loading from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    else:
        examples = download_fever(output_dir, split="dev")
    
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
    
    print(f"Available: NEI={len(by_label['NOT ENOUGH INFO'])}, "
          f"SUPPORTS={len(by_label['SUPPORTS'])}, "
          f"REFUTES={len(by_label['REFUTES'])}")
    
    # Sample from each
    sampled = []
    sampled.extend(random.sample(by_label["NOT ENOUGH INFO"], num_nei))
    sampled.extend(random.sample(by_label["SUPPORTS"], num_supports))
    sampled.extend(random.sample(by_label["REFUTES"], num_refutes))
    
    random.shuffle(sampled)
    
    print(f"Sampled {len(sampled)} examples: {num_nei} NEI, {num_supports} SUPPORTS, {num_refutes} REFUTES")
    
    return sampled


def print_examples(examples: list, num: int = 5, wiki_reader: WikiReader = None):
    """Print sample examples with optional evidence text."""
    print(f"\n{'='*60}")
    print(f"SAMPLE EXAMPLES (showing {min(num, len(examples))} of {len(examples)})")
    print('='*60)
    
    for i, ex in enumerate(examples[:num]):
        print(f"\n--- Example {i+1} ---")
        print(f"ID: {ex.get('id')}")
        print(f"Claim: {ex.get('claim')}")
        print(f"Label: {ex.get('label')}")
        
        # Show evidence if wiki_reader is provided
        if wiki_reader and ex.get('evidence'):
            evidence_texts = wiki_reader.get_evidence_text(ex['evidence'])
            if evidence_texts:
                print("Evidence:")
                for j, text in enumerate(evidence_texts[:3]):  # First 3 evidence items
                    print(f"  {j+1}. {text}")
            else:
                print("Evidence: (none available)")


if __name__ == "__main__":
    # Download wiki pages first
    wiki_dir = download_wiki_pages()
    wiki_reader = WikiReader(wiki_dir)
    
    # Get balanced sample: 500 NEI, 250 SUPPORTS, 250 REFUTES
    sampled = get_balanced_sample(num_nei=500, num_supports=250, num_refutes=250)
    print_examples(sampled, num=5, wiki_reader=wiki_reader)
    
    # Print all sampled IDs
    ids = [ex["id"] for ex in sampled]
    print(f"\n{len(ids)} unique IDs sampled")
