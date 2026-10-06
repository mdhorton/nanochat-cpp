#!/usr/bin/env bash

# bootstrap a fresh vast.ai instance from the local machine: disable vast's auto-tmux, copy tools/vast_remote.sh to the
# home dir, log back in and run it (setup, nccl-bench busbw check, tok-train, quick-d12, then an interactive shell in
# the repo). takes the ssh line vast shows, eg
#   tools/vast_bootstrap.sh ssh -p 37157 root@198.53.64.194 -L 8080:localhost:8080
# (-L and everything after it is ignored)

set -euo pipefail

usage() {
  echo "usage: $0 [--dry-run] [--branch NAME] [--min-bus-bw GB/s] [ssh] [-p PORT] [USER@]HOST [-L ...]" >&2
  echo "  --dry-run    print the ssh commands instead of running them" >&2
  echo "  --branch     git branch to run (default overlap)" >&2
  echo "  --min-bus-bw log out below this NCCL Bus BW (default by PCIe gen: 4 → 15, 5 → 27)" >&2
  exit 2
}

dry_run=0 branch=overlap min_busbw= port=22 user=root host=
set -f; set -- $*; set +f # re-split, so the vast line also works as one quoted string
while (($#)); do
  case $1 in
    --dry-run) dry_run=1 ;;
    --branch) branch=${2:?--branch needs a name}; shift ;;
    --branch=*) branch=${1#*=} ;;
    --min-bus-bw) min_busbw=${2:?--min-bus-bw needs a value}; shift ;;
    --min-bus-bw=*) min_busbw=${1#*=} ;;
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
[[ -n $host && $branch =~ ^[A-Za-z0-9._/-]+$ && $min_busbw =~ ^([0-9]+(\.[0-9]+)?)?$ ]] || usage

opts=(-p "$port" -o ForwardAgent=no -o ServerAliveInterval=60)
dest=$user@$host

remote=$(dirname "$0")/vast_remote.sh
# the first login (no tty) creates ~/.no_auto_tmux, so vast's auto-tmux is off by the second (tty) login
copy='touch ~/.no_auto_tmux && cat > vast_remote.sh'
run=(bash vast_remote.sh --branch "$branch" ${min_busbw:+--min-bus-bw "$min_busbw"})

if ((dry_run)); then
  printf '%q ' ssh "${opts[@]}" "$dest" "$copy"; printf '< %q\n' "$remote"
  printf '%q ' ssh -t "${opts[@]}" "$dest" "${run[@]}"; echo
  exit 0
fi

echo "== $dest:$port: disabling auto-tmux, copying vast_remote.sh"
ssh "${opts[@]}" "$dest" "$copy" < "$remote"
echo "== $dest:$port: running vast_remote.sh"
exec ssh -t "${opts[@]}" "$dest" "${run[@]}"
