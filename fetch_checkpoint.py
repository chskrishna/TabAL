import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from study import load_config, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_representation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    destination = Path(cfg["paths"]["checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(
        repo_id="Prior-Labs/tabpfn_2_5", filename=destination.name,
        local_dir=destination.parent,
    ))
    actual = sha256(downloaded)
    expected = cfg["paths"]["checkpoint_sha256"]
    if actual != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
    print(f"verified {downloaded}: {actual}")


if __name__ == "__main__":
    main()
