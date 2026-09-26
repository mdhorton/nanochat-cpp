# setup

All dependencies (compiler, CUDA, C++ libs, Python oracle incl. torch) come from pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   
pixi install                                   
```

# common tasks

```bash
pixi run build | test | tok-train | tok-eval
pixi run base-train --depth=12            # pretrain (flags as scripts/base_train.py, see --help)
pixi run base-train --depth=12 --run=NAME # checkpoints and metrics.jsonl (for wandb) in base_checkpoints/NAME
pixi run base-train --depth=24 --nproc=2 --fp8  # one process per GPU (as torchrun), FP8 matmuls
pixi run quick-train                      # ~3 min d12 smoke test (bf16: ~123k tok/s, step 10 loss 9.096±0.001, final val bpb 1.778)
pixi run nanochat nanochat.dataset -n 8   # Python nanochat module, shares ./cache
pixi run export-golden                    # golden data for tokenizer parity tests
pixi run export-train-golden              # golden data for training parity tests (cache/golden/train)
pixi run export-train-golden-ddp          # the same tiny training run on 2 GPUs (torchrun)
pixi run export-pretrained                # gpt2 / cl100k_base for tok-eval
python tools/convert_checkpoint.py SRC DST --to safetensors|pt   # Python <-> C++ checkpoints
```

# CLion

use the "Debug (pixi)" / "RelWithDebInfo (pixi)" CMake presets with the Default toolchain.
