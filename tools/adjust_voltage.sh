#!/usr/bin/env bash

# this script likely needs to be run as root.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

nvidia-smi -lgc 0,2800
python3 ./gpu_offset.py --mem 1000
python3 ./gpu_offset.py --gpu 0 --gpc 200
python3 ./gpu_offset.py --gpu 1 --gpc 300
