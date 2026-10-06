#!/usr/bin/env bash
# Build the manuscript to PDF.
#
# Uses tectonic, a single self-contained LaTeX engine that needs no root and no TeX Live
# installation. It is downloaded next to this script on first run (~26 MB) and reused
# afterwards. The first compile also fetches the LaTeX package bundle, so it takes a few
# minutes; later ones take seconds.
#
#   ./build.sh                                   # tables come from $DRIVE_BASE
#   ./build.sh /content/drive/MyDrive/EndoGaussian   # or pass the path explicitly
#
# Result files that do not exist yet become "[pending]" placeholders, so this always
# produces a readable PDF regardless of which modules have been run.
set -euo pipefail
cd "$(dirname "$0")"
DRIVE="${1:-${DRIVE_BASE:-}}"

if [ ! -x ./tectonic ]; then
  echo "fetching tectonic..."
  URL=$(curl -sS https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
        | grep -o '"browser_download_url": *"[^"]*x86_64-unknown-linux-musl[^"]*\.tar\.gz"' \
        | head -1 | sed 's/.*"https/https/;s/"$//')
  curl -sSL "$URL" -o tectonic.tar.gz && tar xzf tectonic.tar.gz && rm tectonic.tar.gz
fi

echo "generating tables from results in: ${DRIVE:-<none set>}"
python3 make_tables.py --drive "$DRIVE" --outdir .
./tectonic -X compile main.tex --outdir .
echo "-> $(pwd)/main.pdf"
