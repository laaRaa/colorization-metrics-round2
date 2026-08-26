#!/usr/bin/env bash
# IPOL run-line wrapper for the colorization-metrics demo (baked weights in image).
# Invoked from DDL.json's "run" field. Working directory at entry is IPOL's
# per-execution input folder, where input_0.png (and optionally input_1.png) live.

set -euo pipefail

BIN="$1"
INPUT_COLORED="$2"
INPUT_GT_NAME="$3"
COLOR_SPACE="$4"
LPIPS_NET="$5"
FID_DIMS="$6"
COLORFULNESS_TYPE="$7"

echo "BIN='$BIN'" >&2
echo "INPUT_COLORED='$INPUT_COLORED'" >&2
echo "INPUT_GT_NAME='$INPUT_GT_NAME'" >&2
echo "COLOR_SPACE='$COLOR_SPACE'" >&2
echo "LPIPS_NET='$LPIPS_NET'" >&2
echo "FID_DIMS='$FID_DIMS'" >&2

# We assume the Docker image has baked the required pretrained weights into $BIN
# layout used by the container build:
#   $BIN/torch/hub/checkpoints/*.pth
#   $BIN/maniqa/ckpt_koniq10k.pt
# If MANIQA weights are present in $BIN/maniqa, symlink them into the user's
# cache path expected by the maniqa loader.

if [ ! -d "$BIN" ]; then
    echo "ERROR: BIN directory '$BIN' does not exist." >&2
    exit 1
fi

MANIQA_PT="$BIN/maniqa/ckpt_koniq10k.pt"
MANIQA_DIR=""
if [ -f "$MANIQA_PT" ]; then
    echo "Found MANIQA checkpoint at $MANIQA_PT" >&2
    MANIQA_DIR="$BIN/maniqa"
elif [ -f "$BIN/ckpt_koniq10k.pt" ]; then
    echo "Found MANIQA checkpoint at $BIN/ckpt_koniq10k.pt" >&2
    # Avoid writing into $BIN (may be root-owned). Copy into the ipol user's cache.
    mkdir -p /home/ipol/.cache/maniqa
    if cp "$BIN/ckpt_koniq10k.pt" /home/ipol/.cache/maniqa/ckpt_koniq10k.pt 2>/dev/null; then
        echo "Copied checkpoint to /home/ipol/.cache/maniqa/ckpt_koniq10k.pt" >&2
        MANIQA_DIR="/home/ipol/.cache/maniqa"
    else
        echo "ERROR: unable to copy $BIN/ckpt_koniq10k.pt into /home/ipol/.cache/maniqa" >&2
        exit 1
    fi
else
    echo "ERROR: MANIQA weights not found at $MANIQA_PT or $BIN/ckpt_koniq10k.pt." >&2
    echo "       The Docker image should include maniqa/ckpt_koniq10k.pt in $BIN or place ckpt_koniq10k.pt at $BIN." >&2
    exit 1
fi

# Point torchvision / pytorch-fid at the bundled checkpoints
export TORCH_HOME="$BIN/torch"

# MANIQA expects its cache under platformdirs.user_cache_dir('maniqa')
mkdir -p /home/ipol/.cache
if [ "$MANIQA_DIR" = "/home/ipol/.cache/maniqa" ]; then
    # already in the user's cache; nothing else to do
    true
else
    # create a symlink in the user's cache that points to the location where
    # the checkpoint lives (usually under $BIN/maniqa)
    ln -sfn "$MANIQA_DIR" /home/ipol/.cache/maniqa
fi

# Resolve input paths BEFORE we cd, since IPOL invokes us with cwd = input folder.
INPUT_COLORED_ABS="$(readlink -f "$INPUT_COLORED")"
INPUT_GT_ABS=""
if [ -f "$INPUT_GT_NAME" ]; then
    INPUT_GT_ABS="$(readlink -f "$INPUT_GT_NAME")"
fi

# Ensure the package in src/ is importable and data/ is available at cwd.
export PYTHONPATH="$BIN/src:${PYTHONPATH:-}"
export BIN_SRC="$BIN/src"
export BIN
# BRISQUE/NIQE read bundled models via relative paths, so run from project root.
cd "$BIN"

ARGS=()
if [ -n "$INPUT_GT_ABS" ]; then
    ARGS+=(--colored "$INPUT_COLORED_ABS" --ground_truth "$INPUT_GT_ABS")
else
    ARGS+=(--colored "$INPUT_COLORED_ABS")
fi
ARGS+=(--color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE")

# Try running as a module first; if that fails (e.g. PYTHONPATH ignored),
# fall back to an inline runner that inserts $BIN/src into sys.path.

# Export CM_ARGS as a '|||'-separated list for the fallback python snippet.
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

# Candidate explicit paths
candidates = [
    os.path.join(bin_src, 'colorization_metrics', 'main_cli.py'),
    os.path.join(bin_src, 'colorization-metrics', 'main_cli.py'),
    os.path.join(bin_root, 'src', 'colorization_metrics', 'main_cli.py'),
    os.path.join(bin_root, 'src', 'colorization-metrics', 'main_cli.py'),
    os.path.join(bin_root, 'main.py'),
]

for p in candidates:
    try:
        print('DEBUG: checking', p, file=sys.stderr)
    except Exception:
        pass
    if p and os.path.isfile(p):
        print('DEBUG: running', p, file=sys.stderr)
        sys.argv = ['main'] + args
        runpy.run_path(p, run_name='__main__')
        sys.exit(0)

# Walk BIN paths for main_cli.py as a last resort
root_walk = bin_src or bin_root or os.getcwd()
for root, dirs, files in os.walk(root_walk):
    if 'main_cli.py' in files:
        p = os.path.join(root, 'main_cli.py')
        print('DEBUG: found via walk', p, file=sys.stderr)
        sys.argv = ['main'] + args
        runpy.run_path(p, run_name='__main__')
        sys.exit(0)

print('ERROR: could not locate main_cli.py; sys.path follows:', file=sys.stderr)
for i,p in enumerate(sys.path[:20]):
    print(i, p, file=sys.stderr)
raise SystemExit(2)
PY
}
