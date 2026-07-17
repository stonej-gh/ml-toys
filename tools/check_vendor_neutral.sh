#!/bin/sh
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
# Vendor-neutrality gate: fail if any hardware-vendor string appears in
# tracked files. Runs in CI and from the publish gate.
#
# The deployment bundles in this repo say WHAT a port must compute, never HOW
# or on WHOSE silicon; keeping every vendor name out of the tree is what makes
# that contract universal. The forbidden terms are stored ROT13-encoded so
# this gate script does not itself contain the strings it forbids (the repo
# must be greppable-clean of them). Decoded only in memory at runtime.
set -e
cd "$(dirname "$0")/.."

rot13() { echo "$1" | tr 'A-Za-z' 'N-ZA-Mn-za-m'; }

# ROT13 of: xilinx|versal|vitis|vivado|zynq|kria|ultrascale|adaptive.soc
PATTERNS=$(rot13 'kvyvak|irefny|ivgvf|ivinqb|mlad|xevn|hygenfpnyr|nqncgvir.fbp')
# ROT13 of: dpu|aie|vck  (short tokens, matched with boundaries below)
SHORT=$(rot13 'qch|nvr|ipx')
# ROT13 of: amd
AMD_TOK=$(rot13 'nzq')

BOUND='(^|[^A-Za-z0-9])(%s)([^A-Za-z0-9]|$)'
AMD_PAT=$(printf "$BOUND" "$AMD_TOK")
SHORT_PAT=$(printf "$BOUND" "$SHORT")

# The gate script itself is exempt (stores the terms ROT13-encoded).
#
# Compressed MEDIA and weight binaries are also exempt (by extension,
# deliberately narrow): megabytes of H.264/JPEG/tensor entropy will eventually
# spell a 3-letter token by chance, making the gate a coin flip per re-encode.
# Text-bearing binaries (e.g. PDFs) are NOT exempt: embedded ASCII is a real
# leak channel, so they stay in the raw scan.
EXEMPT='^(tools/check_vendor_neutral\.sh)'
MEDIA='\.(mp4|mov|jpe?g|png|gif|npz|npy|pt)$'
FILES=$(git ls-files | grep -vE "$EXEMPT" | grep -viE "$MEDIA")

FAIL=0
for pat in "$PATTERNS" "$SHORT_PAT" "$AMD_PAT"; do
    HITS=$(echo "$FILES" | tr '\n' '\0' | xargs -0 grep -liE "$pat" 2>/dev/null || true)
    if [ -n "$HITS" ]; then
        echo "VENDOR-NEUTRALITY VIOLATION:"
        echo "$HITS" | sed 's/^/  /'
        FAIL=1
    fi
done

[ "$FAIL" -ne 0 ] && exit 1
echo "vendor-neutral: clean"
