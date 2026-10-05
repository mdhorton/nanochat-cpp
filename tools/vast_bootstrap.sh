#!/usr/bin/env bash

# bootstrap a fresh vast.ai instance from the local machine: disable vast's auto-tmux and python venv, install pixi,
# log back in, clone /workspace/nanochat-cpp, switch to --branch, run `pixi run nccl-bench`. if the project busbw is
# below --min-busbw, log out; else run `pixi run tok-train` and `pixi run quick-d12`, then stay logged in, in the
# repo. takes the ssh line vast shows, eg
#   tools/vast_bootstrap.sh ssh -p 37157 root@198.53.64.194 -L 8080:localhost:8080
# (-L and everything after it is ignored)

set -euo pipefail

usage() {
  echo "usage: $0 [--dry-run] [--branch NAME] [--min-busbw GB/s] [ssh] [-p PORT] [USER@]HOST [-L ...]" >&2
  echo "  --dry-run   print the ssh commands instead of running them" >&2
  echo "  --branch    git branch to run (default overlap)" >&2
  echo "  --min-busbw log out below this NCCL busbw (default 20)" >&2
  exit 2
}

dry_run=0 branch=overlap min_busbw=20 port=22 user=root host=
set -f; set -- $*; set +f # re-split, so the vast line also works as one quoted string
while (($#)); do
  case $1 in
    --dry-run) dry_run=1 ;;
    --branch) branch=${2:?--branch needs a name}; shift ;;
    --branch=*) branch=${1#*=} ;;
    --min-busbw) min_busbw=${2:?--min-busbw needs a value}; shift ;;
    --min-busbw=*) min_busbw=${1#*=} ;;
    -h | --help) usage ;;
    ssh) ;;
    -p) port=${2:?-p needs a port}; shift ;;
    -p*) port=${1#-p} ;;
    -L*) break ;;
    -*) ;;
    *@*) user=${1%@*} host=${1#*@} ;;
    *) host=$1 ;;
  esac
  shift
done
[[ -n $host && $branch =~ ^[A-Za-z0-9._/-]+$ && $min_busbw =~ ^[0-9]+(\.[0-9]+)?$ ]] || usage

opts=(-p "$port" -o ForwardAgent=no -o ServerAliveInterval=60)
dest=$user@$host

# lines appended (once each) to the end of .bashrc, so they run after vast's own setup on every later login:
# vast's .bashrc activates a python venv, so deactivate it; and cd to $NANOCHAT_CD if set (the final hand over)
novenv='type deactivate >/dev/null 2>&1 && deactivate'
setup='touch ~/.no_auto_tmux
add() { grep -qF "# nanochat: $1" ~/.bashrc || echo "$2 # nanochat: $1" >> ~/.bashrc; }
add "no venv" "'"$novenv"'"
add "cd" "[ -n \"\${NANOCHAT_CD:-}\" ] && cd \"\$NANOCHAT_CD\"; unset NANOCHAT_CD"
command -v pixi >/dev/null || [ -x ~/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash'
# a non-interactive ssh command doesn't source .bashrc, where pixi adds itself to PATH. the bench writes the project
# busbw to cache/nccl/busbw (not written on failure: no GPUs, < 2 GPUs, a hang, wrong results)
bench=$novenv'
export PATH="$HOME/.pixi/bin:$PATH"
cd /workspace && { [ -d nanochat-cpp ] || git clone https://github.com/mdhorton/nanochat-cpp.git; } || exit 1
cd nanochat-cpp && git switch '"$branch"' || exit 1
rm -f cache/nccl/busbw
pixi run nccl-bench --result cache/nccl/busbw --force || echo "nccl-bench failed ($?)"
busbw=$(cat cache/nccl/busbw 2>/dev/null || echo 0)
if awk -v bw="$busbw" -v min='"$min_busbw"' "BEGIN { exit !(bw < min) }"; then
  echo "== NCCL busbw $busbw GB/s is below '"$min_busbw"' GB/s: this host is too slow, logging out"
  exit 1
fi
echo "== NCCL busbw $busbw GB/s"
pixi run tok-train || echo "tok-train failed ($?)"
pixi run quick-d12 || echo "quick-d12 failed ($?)"
NANOCHAT_CD=$PWD exec bash -l'

if ((dry_run)); then
  printf '%q ' ssh "${opts[@]}" "$dest" "$setup"; echo
  printf '%q ' ssh -t "${opts[@]}" "$dest" "$bench"; echo
  exit 0
fi

echo "== $dest:$port: disabling auto-tmux, installing pixi"
ssh "${opts[@]}" "$dest" "$setup"
echo "== $dest:$port: cloning /workspace/nanochat-cpp, running nccl-bench (min $min_busbw GB/s)"
exec ssh -t "${opts[@]}" "$dest" "$bench"
