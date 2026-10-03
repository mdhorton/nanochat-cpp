#!/usr/bin/env bash

# build NVIDIA nccl-tests against the pixi env's CUDA and the NCCL wheel torch uses. run inside `pixi shell`
# (or `pixi run tools/build_nccl_tests.sh`).

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 [--dir DIR] [--arch SM] [--jobs N] [--clean]" >&2
  echo "  --dir   checkout/build dir (default external/nccl-tests)" >&2
  echo "  --arch  sm arch (default 120)" >&2
  echo "  --jobs  make -j (default nproc)" >&2
  echo "  --clean rebuild from scratch" >&2
  exit 2
}

dir=external/nccl-tests arch=120 jobs=$(nproc) clean=0
while (($#)); do
  case $1 in
    --dir) dir=$2 && shift 2 ;;
    --arch) arch=$2 && shift 2 ;;
    --jobs) jobs=$2 && shift 2 ;;
    --clean) clean=1 && shift ;;
    *) usage ;;
  esac
done

[[ -n ${CONDA_PREFIX:-} && -x $CONDA_PREFIX/bin/nvcc ]] || {
  echo "no nvcc in \$CONDA_PREFIX; run inside 'pixi shell'" >&2
  exit 1
}

# the wheel's NCCL (nvidia is a namespace package, so __path__ not __file__).
nccl=$(python -c 'import nvidia.nccl as n; print(list(n.__path__)[0])')
[[ -f $nccl/lib/libnccl.so.2 ]] || {
  echo "no libnccl.so.2 under $nccl" >&2
  exit 1
}

[[ -d $dir/.git ]] || git clone https://github.com/NVIDIA/nccl-tests "$dir"

# the wheel ships only libnccl.so.2; -lnccl needs libnccl.so.
shim=$dir/nccl-wheel
mkdir -p "$shim/lib"
ln -sfn "$nccl/include" "$shim/include"
ln -sf "$nccl/lib/libnccl.so.2" "$shim/lib/libnccl.so"

((clean)) && make -C "$dir" clean

# rpath the env's cudart and the wheel's NCCL, so no LD_LIBRARY_PATH is needed.
NVLDFLAGS="-Xlinker -rpath,$CONDA_PREFIX/lib:$nccl/lib" make -C "$dir" -j "$jobs" MPI=0 \
  CUDA_HOME="$CONDA_PREFIX" CUDA_LIB="$CONDA_PREFIX/lib" CUDA_INC="$CONDA_PREFIX/targets/x86_64-linux/include" \
  NCCL_HOME="$(readlink -f "$shim")" NVCC_GENCODE="-gencode=arch=compute_$arch,code=sm_$arch"

echo "built: $dir/build ($(ls "$dir"/build/*_perf | wc -l) tests, NCCL from $nccl)"
