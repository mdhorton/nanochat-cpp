# setup

All dependencies (compiler, CUDA, C++ libs, Python oracle incl. torch) come from pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   
pixi install                                   
```

```bash
git clone https://github.com/Dao-AILab/flash-attention external/flash-attention
git -C external/flash-attention submodule update --init csrc/cutlass
```

# common tasks

```bash
pixi run build | test | tok-train | tok-eval
pixi run base-train --depth=12            # pretrain (flags as scripts/base_train.py, see --help); MXFP8 by default
pixi run base-train --depth=12 --run=NAME # checkpoints in base_checkpoints/NAME, metrics (for wandb) in metrics/NAME/metrics-TSTAMP.jsonl
pixi run base-train --depth=24 --nproc=2  # one process per GPU (as torchrun)
pixi run base-train --fp8=false            # bf16 matmuls (Python's default)
pixi run base-train --fp8-recipe=tensorwise # FP8 as Python's --fp8
pixi run quick-d12 | quick-d24            # smoke tests (MXFP8, 2 GPUs): tok/s, early loss, final val bpb
pixi run full-d12 | full-d24              # full runs (MXFP8, 2 GPUs)
pixi run profile-d12 | profile-d24        # nsys (steps 2-4) + ncu (1 GPU, step 2) -> cache/profiles/d12_N.{nsys,ncu}-rep
pixi run profile-d12 ncu --metrics M --kernel-name regex:K -c 5   # one tool only (nsys|ncu); flags after it go to ncu and override its defaults
nsys stats --report nvtx_sum,cuda_gpu_kern_sum cache/profiles/d12_1.nsys-rep   # summary tables in the terminal
pixi run nanochat nanochat.dataset -n 8   # Python nanochat module, shares ./cache
pixi run export-golden                    # golden data for tokenizer parity tests
pixi run export-train-golden              # golden data for training parity tests (cache/golden/train)
pixi run export-train-golden-ddp          # the same tiny training run on 2 GPUs (torchrun)
pixi run export-pretrained                # gpt2 / cl100k_base for tok-eval
python tools/convert_checkpoint.py SRC DST --to safetensors|pt   # Python <-> C++ checkpoints
```

# CLion

use the "Debug (pixi)" / "RelWithDebInfo (pixi)" CMake presets with the Default toolchain.
