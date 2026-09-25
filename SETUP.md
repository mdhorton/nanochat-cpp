# setup

All dependencies (compiler, CUDA, C++ libs, Python oracle incl. torch) come from pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # install pixi
pixi install                                   # create .pixi/ env
```

# common tasks

```bash
pixi run build | test | tok-train | tok-eval
pixi run base-train --depth=12            # pretrain (flags as scripts/base_train.py, see --help)
pixi run nanochat nanochat.dataset -n 8   # Python nanochat module, shares ./cache
pixi run export-golden                    # golden data for tokenizer parity tests
pixi run export-train-golden              # golden data for training parity tests (cache/golden/train)
pixi run export-pretrained                # gpt2 / cl100k_base for tok-eval
python tools/convert_checkpoint.py SRC DST --to safetensors|pt   # Python <-> C++ checkpoints
```

CLion: use the "Debug (pixi)" / "RelWithDebInfo (pixi)" CMake presets with the Default toolchain.
