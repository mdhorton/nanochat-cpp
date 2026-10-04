# requirements

- sm120 capability GPU (RTX Pro 6000, RTX Pro 4000, RTX 5090, etc...)
- nvidia driver supporting cuda 13+

I've only tested with Linux Ubuntu 24.04.

# setup

Most dependencies (compiler, CUDA, C++ libs, Python oracle incl. torch) come from pixi. Pixi can be installed with:

```bash
curl -fsSL https://pixi.sh/install.sh | bash   
```

Clone project and install the dependencies. Does not require root. Everything is self-contained within the project dir.

```bash
git clone https://github.com/mdhorton/nanochat-cpp.git
cd nanochat-cpp
```

```bash
pixi install                                   
pixi run dataset 170     # optional: tok-train downloads the first 10
pixi run tok-train
```

# wandb (optional)

`base_train --run=<name> --wandb` uploads metrics live; `pixi run wandb-upload --file cache/metrics/<run>` uploads a
finished run.

```bash
pixi run wandb login   # or export WANDB_API_KEY=<key>
```

# CLion

use the "Debug (pixi)" / "RelWithDebInfo (pixi)" CMake presets with the Default toolchain.
