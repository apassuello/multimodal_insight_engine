import os
import json
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import random


class CombinedTranslationDataset(Dataset):
    def __init__(self, data, max_samples=None):
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def to_standard_format(example, src_lang="de", tgt_lang="en"):
    return {
        "source": example.get("translation", {}).get(src_lang)
        or example.get("de")
        or example.get("deu"),
        "target": example.get("translation", {}).get(tgt_lang)
        or example.get("en")
        or example.get("eng"),
    }


def create_combined_dataset_jsonl(
    path="combined_de_en_dataset.jsonl",
    n_wmt=2_000_000,
    n_opus=50_000,
    src_lang="de",
    tgt_lang="en",
):
    print("Downloading WMT14...")
    wmt_dataset = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split="train")
    n_wmt = min(n_wmt, len(wmt_dataset))
    wmt = wmt_dataset.select(range(n_wmt))
    print(f"Selected {n_wmt} samples from WMT14 (total available: {len(wmt_dataset)})")

    print("Downloading OPUS Books...")
    opus_dataset = load_dataset("opus_books", f"{src_lang}-{tgt_lang}", split="train")
    n_opus = min(n_opus, len(opus_dataset))
    opus = opus_dataset.select(range(n_opus))
    print(
        f"Selected {n_opus} samples from OPUS Books (total available: {len(opus_dataset)})"
    )

    print("Normalizing format...")
    wmt = wmt.map(lambda x: to_standard_format(x, src_lang, tgt_lang))
    opus = opus.map(lambda x: to_standard_format(x, src_lang, tgt_lang))

    print("Combining and shuffling...")
    combined = concatenate_datasets([wmt, opus])
    combined = combined.shuffle(seed=42)

    print(f"Saving to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for item in combined:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved combined dataset with {len(combined)} samples.")


def load_dataset_from_file(path="combined_de_en_dataset.jsonl", max_samples=None):
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Creating it...")
        create_combined_dataset_jsonl(path)

    print(f"Loading dataset from {path}...")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            if item["source"] and item["target"]:
                data.append((item["source"], item["target"]))
    print(f"Loaded {len(data)} samples from dataset.")
    return CombinedTranslationDataset(data)
