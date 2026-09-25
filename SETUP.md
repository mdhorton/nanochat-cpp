# setup

All dependencies (compiler, CUDA, C++ libs, Python oracle incl. torch) come from pixi (`pixi.toml`).

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # install pixi
pixi install                                   # create .pixi/ env
```

# common tasks

```bash
pixi run build | test | tok-train | tok-eval
pixi run nanochat nanochat.dataset -n 8   # Python nanochat module, shares ./cache
pixi run export-golden                    # golden data for parity tests
pixi run export-pretrained                # gpt2 / cl100k_base for tok-eval
```
