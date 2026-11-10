#!/usr/bin/env python3
"""
1-Click Strata Launcher & Shard-6 Linker for:
Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF
"""
import os
import sys
import subprocess
from pathlib import Path


def ensure_shard6_link(model_dir: Path) -> None:
    engram = model_dir / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
    shard6 = model_dir / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf"
    if engram.exists() and not shard6.exists():
        print(f"[*] Linking {shard6.name} -> {engram.name}...")
        try:
            os.link(engram, shard6)
        except Exception:
            shard6.symlink_to(engram.name)


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    ensure_shard6_link(model_dir)
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else ["--full-quality"]
    cmd = [
        "strata",
        "launch",
        "--family",
        "deepseek41",
        "--model",
        "ist-das-2.00bpw",
        "--model-path",
        str(model_dir),
        *extra_args,
    ]
    print(f"[*] Launching Strata: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
