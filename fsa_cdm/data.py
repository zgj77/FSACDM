import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DATASETS = {
    "math": ("yuntian-deng/im2latex-100k", "formula"),
    "tables": ("yuntian-deng/im2html-100k", "html"),
    "music": ("yuntian-deng/im2ly-35k-syn", "source"),
    "molecules": ("yuntian-deng/im2smiles-20k", "smiles"),
}


def tokenizer_for(config, path=None):
    tokenizer = AutoTokenizer.from_pretrained(path or config["encoder"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class MarkupDataset(Dataset):
    def __init__(self, root, split, config, limit=None):
        self.root = Path(root)
        self.rows = [
            json.loads(line)
            for line in (self.root / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")
        self.config = config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as img:
            image = img.convert("RGB" if self.config["channels"] == 3 else "L")
            target = tuple(reversed(self.config["image_size"]))
            if image.size != target:
                raise ValueError(
                    f"{row['image']}: image {image.size}, expected {target}"
                )
            array = np.array(image, dtype=np.float32, copy=True)
        if array.ndim == 2:
            array = array[..., None]
        return {
            "images": torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1,
            "texts": row["text"],
            "filename": row["filename"],
            "image_id": row.get("source_filename", row["image"]),
        }


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, rows):
        texts = [r["texts"] for r in rows]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if tokens["input_ids"].shape[1] == 0:
            tokens["input_ids"] = torch.zeros(len(rows), 1, dtype=torch.long)
            tokens["attention_mask"] = torch.zeros(len(rows), 1, dtype=torch.long)
        empty = tokens["attention_mask"].sum(1) == 0
        if empty.any():
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is None:
                raise ValueError("Empty markup requires a tokenizer with an EOS token")
            tokens["input_ids"][empty, 0] = eos
            tokens["attention_mask"][empty, 0] = 1
        return {
            "images": torch.stack([r["images"] for r in rows]),
            "texts": texts,
            "filenames": [r["filename"] for r in rows],
            "image_ids": [r["image_id"] for r in rows],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }


def prepare_dataset(name, output, parquet_dir=None, limit=None):
    from datasets import load_dataset

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    repo, field = DATASETS[name]
    if parquet_dir:
        files = {}
        for split in ["train", "val", "test"]:
            files[split] = sorted(
                str(p) for p in Path(parquet_dir).glob(f"{split}-*.parquet")
            )
            if not files[split]:
                raise FileNotFoundError(f"No {split} parquet files in {parquet_dir}")
        dataset = load_dataset("parquet", data_files=files)
    else:
        dataset = load_dataset(repo)
    counts = {}
    for split in ["train", "val", "test"]:
        source_split = (
            "validation" if split == "val" and "val" not in dataset else split
        )
        source = dataset[source_split]

        source = source.shuffle(seed=42)
        n = min(len(source), limit) if limit else len(source)
        if split == "test" and name in ["math", "tables"]:
            n = min(n, 1024)
        images_dir = output / "images" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output / f"{split}.jsonl"
        temporary_path = output / f"{split}.jsonl.tmp"
        used_filenames = set()
        with temporary_path.open("w") as file:
            for i, row in enumerate(source.select(range(n))):
                source_filename = (
                    Path(str(row.get("filename", f"{i:08d}.png"))).stem + ".png"
                )
                filename = source_filename
                while filename in used_filenames:
                    filename = f"{i:08d}_{filename}"
                used_filenames.add(filename)
                image = row["image"]
                if isinstance(image, dict):
                    image = (
                        Image.open(io.BytesIO(image["bytes"]))
                        if image.get("bytes")
                        else Image.open(image["path"])
                    )
                image.save(images_dir / filename)
                record = {
                    "image": f"images/{split}/{filename}",
                    "filename": filename,
                    "source_filename": source_filename,
                    "text": row[field],
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(manifest_path)
        counts[split] = n
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source": repo,
                "split_counts": counts,
                "shuffle_seed": 42,
                "limit_per_split": limit,
                "parquet_dir": str(parquet_dir),
            },
            indent=2,
        )
    )
    print(json.dumps(counts), flush=True)
