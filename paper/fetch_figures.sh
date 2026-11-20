#!/usr/bin/env bash
# Copy the result figures the manuscript includes out of DRIVE_BASE.
# The paper compiles without them (each \includegraphics is guarded by
# \IfFileExists), so this is optional until you want the figures in the PDF.
set -euo pipefail
cd "$(dirname "$0")"
DRIVE="${1:-${DRIVE_BASE:-}}"
[ -n "$DRIVE" ] || { echo "usage: ./fetch_figures.sh /path/to/EndoGaussian"; exit 1; }
mkdir -p figures
copy () { [ -f "$1" ] && cp "$1" "$2" && echo "  $2" || echo "  (missing: $1)"; }
copy "$DRIVE/module6/visuals/side_by_side_workspace_v2.png" figures/qualitative.png
# The denoising figure was absent from this list, so figures/denoising.png kept
# whatever an earlier run left on disk while its caption was regenerated from the
# current one -- a figure and a caption describing different checkpoints.
copy "$DRIVE/module6/visuals/denoising_workspace_v2.png"   figures/denoising.png
copy "$DRIVE/module6/module6_qa.png"                        figures/analysis.png
copy "$DRIVE/module6/module6_timeline.png"                  figures/timeline.png
copy "$DRIVE/module7/module7_comparison.png"                figures/comparison.png
copy "$DRIVE/module8/module8_hpo.png"                       figures/hpo.png
echo "done -- rebuild with ./build.sh \"$DRIVE\""
