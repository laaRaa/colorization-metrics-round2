#!/usr/bin/env bash
# IPOL run-line wrapper for the colorization-metrics demo.
# Invoked from DDL.json's "run" field. Working directory at entry is IPOL's
# per-execution input folder, where input_0.png (and optionally input_1.png) live.

set -euo pipefail

BIN="${1:-}"
INPUT_COLORED="${2:-}"
INPUT_GT_NAME="${3:-}"
COLOR_SPACE="${4:-lab}"
LPIPS_NET="${5:-alex}"
FID_DIMS="${6:-64}"
COLORFULNESS_TYPE="${7:-3}"

echo "BIN='$BIN'" >&2
echo "INPUT_COLORED='$INPUT_COLORED'" >&2
echo "INPUT_GT_NAME='$INPUT_GT_NAME'" >&2
echo "COLOR_SPACE='$COLOR_SPACE'" >&2
echo "LPIPS_NET='$LPIPS_NET'" >&2
echo "FID_DIMS='$FID_DIMS'" >&2

if [ ! -d "$BIN" ]; then
    echo "ERROR: BIN directory '$BIN' does not exist." >&2
    exit 1
fi

# Force PyTorch and MANIQA to use writable cache directories owned by the ipol user.
export HOME=/home/ipol
export XDG_CACHE_HOME=/home/ipol/.cache
export TORCH_HOME=/home/ipol/.cache/torch
mkdir -p /home/ipol/.cache /home/ipol/.cache/maniqa /home/ipol/.cache/torch/hub/checkpoints

# MANIQA checkpoint: prefer the repo-local version if it exists, otherwise use a
# copy placed at the root of BIN during Docker build.
MANIQA_LOCAL=""
if [ -f "$BIN/maniqa/ckpt_koniq10k.pt" ]; then
    MANIQA_LOCAL="$BIN/maniqa/ckpt_koniq10k.pt"
elif [ -f "$BIN/ckpt_koniq10k.pt" ]; then
    MANIQA_LOCAL="$BIN/ckpt_koniq10k.pt"
fi

if [ -z "$MANIQA_LOCAL" ]; then
    echo "ERROR: MANIQA checkpoint not found at $BIN/maniqa/ckpt_koniq10k.pt or $BIN/ckpt_koniq10k.pt." >&2
    exit 1
fi

cp -f "$MANIQA_LOCAL" /home/ipol/.cache/maniqa/ckpt_koniq10k.pt

echo "Copied MANIQA checkpoint to /home/ipol/.cache/maniqa/ckpt_koniq10k.pt" >&2

# Preload the Torch Hub weights used by LPIPS / FID if they are bundled in the image.
for f in \
    alexnet-owt-7be5be79.pth \
    pt_inception-2015-12-05-6726825d.pth \
    squeezenet1_1-b8a52dc0.pth \
    vgg16-397923af.pth
do
    src="$BIN/torch/hub/checkpoints/$f"
    dst="/home/ipol/.cache/torch/hub/checkpoints/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
    fi
done

# Resolve input paths BEFORE cd, since IPOL invokes us with cwd = input folder.
INPUT_COLORED_ABS=""
if [ -n "$INPUT_COLORED" ] && [ -f "$INPUT_COLORED" ]; then
    INPUT_COLORED_ABS="$(readlink -f "$INPUT_COLORED")"
fi

INPUT_GT_ABS=""
if [ -n "$INPUT_GT_NAME" ] && [ -f "$INPUT_GT_NAME" ]; then
    INPUT_GT_ABS="$(readlink -f "$INPUT_GT_NAME")"
fi

# Ensure the project package and data are importable.
export PYTHONPATH="$BIN/src:${PYTHONPATH:-}"
export BIN_SRC="$BIN/src"
export BIN

# BRISQUE/NIQE load models relative to the project root, so run from project root.
cd "$BIN"

ARGS=()
if [ -n "$INPUT_GT_ABS" ]; then
    ARGS+=(--colored "$INPUT_COLORED_ABS" --ground_truth "$INPUT_GT_ABS")
else
    ARGS+=(--colored "$INPUT_COLORED_ABS")
fi
ARGS+=(--color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE")

CM_ARGS="$(printf '%s\x1f' "${ARGS[@]}")"
export CM_ARGS

python -m colorization_metrics.main_cli "${ARGS[@]}" || {
    echo "Module run failed, falling back to inline runner" >&2
    python - <<'PY'
import os, sys, runpy
bin_src = os.environ.get('BIN_SRC', '')
bin_root = os.environ.get('BIN', '')
sep = '\x1f'
cm_args = os.environ.get('CM_ARGS', '')
args = [x for x in cm_args.split(sep) if x]

candidates = [
    os.path.join(bin_src, 'colorization_metrics', 'main_cli.py'),
    os.path.join(bin_root, 'src', 'colorization_metrics', 'main_cli.py'),
    os.path.join(bin_root, 'main.py'),
]

for p in candidates:
    if p and os.path.isfile(p):
        sys.argv = ['main'] + args
        runpy.run_path(p, run_name='__main__')
        sys.exit(0)

print('ERROR: could not locate main_cli.py', file=sys.stderr)
sys.exit(1)
PY
}
