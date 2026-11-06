"""Fetch the reference PDFs and prepare the local embedding model cache."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PDF_BASE = "https://www.analog.com/media/en/technical-documentation/data-sheets/"
PDF_HASHES = {
    "DS18B20": "066c8a3c7e9222c49e84488840d5cfe9dd109cf0da6ab4a55480cebac8f97f76",
    "DS18S20": "8061fc3afcea575131c9f49fdd57534e8f3b3da6795c6824752de4cd1e57886b",
}
MODEL = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def prepare_pdfs():
    folder = ROOT / "learning" / "迁移资料"
    folder.mkdir(parents=True, exist_ok=True)
    for model, digest in PDF_HASHES.items():
        target = folder / f"{model}.pdf"
        if target.is_file():
            data = target.read_bytes()
        else:
            request = Request(f"{PDF_BASE}{model.lower()}.pdf", headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"{model}.pdf differs from the validated revision. Check the source listed in README.md.")
        if not target.exists():
            temporary = target.with_suffix(".pdf.part")
            temporary.write_bytes(data)
            temporary.replace(target)
        print(f"Ready: {target.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-only", action="store_true", help="Skip the embedding model download.")
    args = parser.parse_args()
    prepare_pdfs()
    if not args.pdf_only:
        from huggingface_hub import snapshot_download

        cache = ROOT / "models" / "huggingface"
        snapshot_download(
            repo_id=MODEL,
            revision=REVISION,
            cache_dir=str(cache),
            token=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        # Pinned snapshots do not populate the branch alias used by offline model loading.
        alias = cache / "models--BAAI--bge-small-zh-v1.5" / "refs" / "main"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(REVISION, encoding="utf-8")
        print(f"Ready: {MODEL}")


if __name__ == "__main__":
    main()
