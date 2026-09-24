#!/usr/bin/env bash
# Download real public log datasets from Loghub (LogPAI).
#
#   bash scripts/download_loghub.sh          # Linux + OpenSSH  (~72 MB)
#   bash scripts/download_loghub.sh --all    # + Apache, Zookeeper, Mac (~175 MB)
#
# Datasets are free for research/academic use. Cite:
#   Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
#   "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics."
#   ISSRE 2023.  https://github.com/logpai/loghub
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/raw/loghub}"
BASE="https://zenodo.org/records/8196385/files"
CORE=("Linux.tar.gz" "SSH.tar.gz")
EXTRA=("Apache.tar.gz" "Zookeeper.tar.gz" "Mac.tar.gz")

mkdir -p "$OUT_DIR"

download() {
  local file="$1"
  local target="$OUT_DIR/$file"
  if [[ -s "$target" ]]; then
    echo "[loghub] cached: $file"
    return 0
  fi
  echo "[loghub] downloading $file ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$target" "$BASE/$file?download=1"
  else
    wget -O "$target" "$BASE/$file?download=1"
  fi
  echo "[loghub] extracting $file ..."
  tar -xzf "$target" -C "$OUT_DIR"
  rm -f "$target"
}

for f in "${CORE[@]}"; do download "$f"; done

if [[ "${1:-}" == "--all" ]]; then
  for f in "${EXTRA[@]}"; do download "$f"; done
fi

echo "[loghub] done. Contents of $OUT_DIR:"
find "$OUT_DIR" -maxdepth 2 -type f | head -40
echo
echo "[loghub] verify a parseable file, e.g.:"
echo "  head -5 $OUT_DIR/Linux/Linux.log"
echo "  head -5 $OUT_DIR/SSH/SSH.log"
echo
echo "[loghub] next: python -m slrag.cli ingest --input data/raw --out data/processed"
