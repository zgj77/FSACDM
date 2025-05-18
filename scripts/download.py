import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/datasets"))
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi, snapshot_download

DATA = {
    "math": "im2latex-100k",
    "tables": "im2html-100k",
    "music": "im2ly-35k-syn",
    "molecules": "im2smiles-20k",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and pretrained text encoders"
    )
    parser.add_argument("--dataset", choices=list(DATA), default="math")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    repo = "yuntian-deng/" + DATA[args.dataset]
    model = (
        "DeepChem/ChemBERTa-77M-MLM"
        if args.dataset == "molecules"
        else "EleutherAI/gpt-neo-125m"
    )
    model_dir = (
        ROOT
        / "assets"
        / ("chemberta-77m-mlm" if args.dataset == "molecules" else "gpt-neo-125m")
    )
    dataset_dir = ROOT / "assets" / args.dataset
    api = HfApi()
    model_info = api.model_info(model)
    model_revision = model_info.sha
    model_files = [entry.rfilename for entry in model_info.siblings]
    weight = (
        "model.safetensors"
        if "model.safetensors" in model_files
        else "pytorch_model.bin"
    )
    snapshot_download(
        model,
        revision=model_revision,
        local_dir=model_dir,
        max_workers=4,
        allow_patterns=["*.json", "merges.txt", "vocab.txt", weight],
    )
    if args.model_only:
        print(f"Downloaded {model} to {model_dir}", flush=True)
        return
    data_revision = api.dataset_info(repo).sha
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=data_revision,
        local_dir=dataset_dir,
        allow_patterns=["data/*.parquet", "*.json", "README.md"],
        max_workers=4,
    )
    manifest = {
        "model": model,
        "model_revision": model_revision,
        "dataset": repo,
        "dataset_revision": data_revision,
        "files": {},
    }
    for folder in [model_dir, dataset_dir]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                manifest["files"][str(path.relative_to(ROOT))] = {
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
    (ROOT / "assets" / f"{args.dataset}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"Downloaded {repo}. Prepare with --parquet-dir assets/{args.dataset}/data",
        flush=True,
    )


if __name__ == "__main__":
    main()
