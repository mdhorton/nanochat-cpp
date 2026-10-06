#!/usr/bin/env bash

# runs on a fresh vast.ai instance; copied there and started by tools/vast_bootstrap.sh. disables vast's python venv,
# installs pixi, clones /workspace/nanochat-cpp, switches to --branch. with > 1 GPU, runs `pixi run nccl-bench` and
# exits if the busbw is below --min-bus-bw (default by PCIe gen: 4 → 15, 5 → 27 GB/s). then runs
# `pixi run tok-train` and `pixi run quick-d12`, and hands over an interactive login shell in the repo.

set -uo pipefail

usage() {
  echo "usage: $0 [--branch NAME] [--min-bus-bw GB/s]" >&2
  exit 2
}

branch=overlap min_busbw=
while (($#)); do
  case $1 in
    --branch) branch=${2:?--branch needs a name}; shift ;;
    --min-bus-bw) min_busbw=${2:?--min-bus-bw needs a value}; shift ;;
    *) usage ;;
  esac
  shift
done

# lines appended (once each) to the end of .bashrc, so they run after vast's own setup on every later login:
# vast's .bashrc activates a python venv, so deactivate it; and cd to $NANOCHAT_CD if set (the final hand over)
add() { grep -qF "# nanochat: $1" ~/.bashrc || echo "$2 # nanochat: $1" >> ~/.bashrc; }
add "no venv" 'type deactivate >/dev/null 2>&1 && deactivate'
add "cd" '[ -n "${NANOCHAT_CD:-}" ] && cd "$NANOCHAT_CD"; unset NANOCHAT_CD'
echo "== Deactivating default vast environment (if any)."
type deactivate >/dev/null 2>&1 && deactivate

# a non-interactive ssh command doesn't source .bashrc, where pixi adds itself to PATH
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash || exit 1

cd /workspace && { [ -d nanochat-cpp ] || git clone https://github.com/mdhorton/nanochat-cpp.git; } || exit 1
cd nanochat-cpp && git switch "$branch" || exit 1

# the bench writes the project busbw to cache/nccl/busbw (not written on failure: no GPUs, a hang, wrong results)
gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')

# pcie.link.gen.max is the lower of the GPU's and host's max gen (the current gen drops while idle). the slowest link
# sets the default --min-bus-bw
gen=
while IFS=', ' read -r i link gpu; do
  [[ $link =~ ^[0-9]+$ && $gpu =~ ^[0-9]+$ ]] || continue
  ((link < gpu)) && echo "== WARNING: GPU $i PCIE LINK IS GEN $link BUT THE GPU SUPPORTS GEN $gpu"
  [[ -z $gen ]] || ((link < gen)) && gen=$link
done < <(nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.gpumax --format=csv,noheader 2>/dev/null)
[[ -n $min_busbw ]] || { [[ -n $gen ]] && ((gen >= 5)) && min_busbw=27 || min_busbw=15; }
echo "== $gpus GPU(s), PCIe gen ${gen:-unknown}, min Bus BW $min_busbw GB/s"
if ((gpus > 1)); then
  rm -f cache/nccl/busbw
  pixi run nccl-bench --result cache/nccl/busbw --force || echo "nccl-bench failed ($?)"
  busbw=$(cat cache/nccl/busbw 2>/dev/null || echo 0)
  if awk -v bw="$busbw" -v min="$min_busbw" 'BEGIN { exit !(bw < min) }'; then
    echo "== NCCL Bus BW $busbw GB/s is below $min_busbw GB/s: this host is too slow, logging out"
    exit 1
  fi
  echo "== NCCL Bus BW $busbw GB/s"
else
  echo "== skipping nccl-bench"
fi

pixi run tok-train || echo "tok-train failed ($?)"
pixi run quick-d12 || echo "quick-d12 failed ($?)"

NANOCHAT_CD=$PWD exec bash -l
