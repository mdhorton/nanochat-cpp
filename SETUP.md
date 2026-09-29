# requirements

- sm120 capability GPU
- nvidia driver supporting cuda 13.2+

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
pixi install                                   
```

# CLion

use the "Debug (pixi)" / "RelWithDebInfo (pixi)" CMake presets with the Default toolchain.
