#!/usr/bin/env bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh
    export PATH="$HOME/.pixi/bin:$PATH"
fi

pixi install
pixi run tok-train
