# porting nanochat to c++

The goal is to port nanochat (python) to c++ and target sm120 GPUs.

## original nanochat

The original nanochat repo is cloned into `external/nanochat`. Do not make any changes to the original project.

## high level breakdown

nanochat has 4 high-level parts:

1. tokenizer
2. pre-training
3. evaluation
4. inference

## porting phases

### phase 0

- create environment (pixi.toml)
- setup build
- download data

### phase 1

port BPE tokenizer. this uses the same logic as the original tokenizer.

create tests so we can compare output with the original. they should be identical.

### phase 2

port pre-training (`gpt.py`, `optim.py`, `dataloader.py`, `loss_eval.py`, nanochat's `scripts/base_train.py`) to libtorch.

- oracle: python nanochat run uncompiled (`TORCHDYNAMO_DISABLE=1`), SDPA attention, seed 42. the dataloader must
  match exactly; everything else within bf16 tolerances.
- golden data: `scripts/export_train_golden.py` -> `cache/golden/train/torch-<major.minor>/`.
- parity config: d4, head_dim 64, T=256, SSSL, device batch 8, total batch 4096, 20 iterations.
- target: d24 on 2 GPUs + FP8.
