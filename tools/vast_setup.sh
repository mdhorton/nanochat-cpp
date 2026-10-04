#!/usr/bin/env bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh
    export PATH="$HOME/.pixi/bin:$PATH"
fi

mkdir external
(cd external && git clone https://github.com/karpathy/nanochat.git)

pixi install
pixi run nanochat nanochat.dataset -n 10
pixi run tok-train
