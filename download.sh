#!/usr/bin/env bash
# Download GGUF weights for this model from Hugging Face.
# This repository contains only the model card, small files and this script;
# weights are NOT included (GitHub single-file limit is 100 MB).
# No token, no credentials: the HF repo is public (gated=false).
set -euo pipefail
HF_REPO="Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF"
BASE="https://huggingface.co/$HF_REPO/resolve/main"

usage() {
  echo "Usage:"
  echo "  ./download.sh                # default: IQ2_XXS"
  echo "  ./download.sh [QUANT]      # download another quant, e.g. ./download.sh Q4_K_S"
  echo "  ./download.sh --list       # list all available quants and sizes"
  echo "  ./download.sh --mmproj [NAME]  # download mmproj file (default: first mmproj listed)"
  echo ""
  echo "Available quants:"
  echo "  IQ2_XXS  (138.58 GB, 5 file(s))"
  echo "  Engram-IQ4_NL  (110.60 GB, 1 file(s))"
}

list_quants() {
  echo "Available quants (from HF file tree; files are NOT in this repo):"
  echo "  IQ2_XXS: 138.58 GB (138581228864 bytes)"
  echo "    DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf  5780704 bytes"
  echo "    DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf  39661282624 bytes"
  echo "    DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf  39851072320 bytes"
  echo "    DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf  39111455584 bytes"
  echo "    DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf  19951637632 bytes"
  echo "  Engram-IQ4_NL: 110.60 GB (110595290816 bytes)"
  echo "    NOTE: Engram file purpose is UNVERIFIED (may be an engram component, not a full model) - not default, not recommended"
  echo "    DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf  110595290816 bytes"
  echo "  mmproj BF16: mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf  972208096 bytes (0.97 GB)"
}

declare -a FILES=()  # entries: filename|size|sha256
TOTAL=0
QUANT="${1:-IQ2_XXS}"

if [ "$QUANT" = "--help" ] || [ "$QUANT" = "-h" ]; then usage; exit 0; fi
if [ "$QUANT" = "--list" ]; then list_quants; exit 0; fi

if [ "$QUANT" = "--mmproj" ]; then
  MMPROJ_NAME="${2:-}"
  case "$MMPROJ_NAME" in
    "BF16"|"") FILES=("mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf|972208096|0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1") ;;
    *) echo "Unknown mmproj: $MMPROJ_NAME (available: BF16)" >&2; exit 1 ;;
  esac
  echo "mmproj download selected."
else
  case "$QUANT" in
    "IQ2_XXS") FILES=("DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf|5780704|f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9" "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf|39661282624|db021c117306070573a8aba333f34f9c47ee757a1072aaacc0cab7bf4c737862" "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf|39851072320|5c6de54009714a6996985f86553348e5d8e0d5ece7f64a89558e2f53d90bfa76" "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf|39111455584|706de34f9c252234cd349081b3b76bf69dc77677ac8d006f7d4538558ba49e87" "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf|19951637632|85e29c00f0d5616de05d20b96e367ed9eb3ba20ed958c860da899dad8c9f7a8e") ;;
    "Engram-IQ4_NL") FILES=("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf|110595290816|3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3") ;;
    *) echo "Unknown quant: $QUANT" >&2; usage >&2; exit 1 ;;
  esac
fi


# Recompute total size robustly
TOTAL=0
for e in "${FILES[@]}"; do sz="${e#*|}"; sz="${sz%%|*}"; TOTAL=$((TOTAL + sz)); done
echo "Total download size: $TOTAL bytes (~$(awk -v b="$TOTAL" 'BEGIN {printf "%.2f", b/1000000000}') GB)"
echo "Make sure you have enough free disk space before continuing."
echo "WARNING: this download is about 138.6 GB. Ensure at least 140 GB free disk space!"

download_one() {
  local entry="$1"; local file="${entry%%|*}"; local rest="${entry#*|}"; local size="${rest%%|*}"; local sha="${rest##*|}"
  echo "Downloading $file ($size bytes) ..."
  curl -L --fail --retry 3 -o "$file.tmp" "$BASE/$file"
  local got; got=$(stat -c%s "$file.tmp")
  if [ "$got" != "$size" ]; then echo "SIZE MISMATCH for $file: got $got, expected $size" >&2; rm -f "$file.tmp"; exit 1; fi
  local gotsha; gotsha=$(sha256sum "$file.tmp" | cut -d" " -f1)
  if [ "$gotsha" != "$sha" ]; then echo "SHA256 MISMATCH for $file: got $gotsha, expected $sha" >&2; rm -f "$file.tmp"; exit 1; fi
  mv "$file.tmp" "$file"
  echo "OK $file"
}

for e in "${FILES[@]}"; do download_one "$e"; done
echo "All files downloaded and verified."
