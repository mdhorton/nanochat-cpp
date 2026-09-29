#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

sudo nvidia-smi -lgc 0,2800
sudo python3 ./gpu_offset.py --mem 1000
sudo python3 ./gpu_offset.py --gpu 0 --gpc 200
sudo python3 ./gpu_offset.py --gpu 1 --gpc 300
