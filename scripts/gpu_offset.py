#!/usr/bin/env python3

# read or set NVIDIA GPC clock offsets via NVML (libnvidia-ml, no extra packages). setting needs root.

import argparse
import ctypes as c

nvml = c.CDLL("libnvidia-ml.so.1")


def check(ret, what):
    if ret != 0:
        msg = nvml.nvmlErrorString
        msg.restype = c.c_char_p
        raise SystemExit(f"{what}: {msg(ret).decode()}")


def handle(i):
    h = c.c_void_p()
    check(nvml.nvmlDeviceGetHandleByIndex_v2(i, c.byref(h)), f"gpu{i}")
    return h


def show(i):
    h = handle(i)
    lo, hi, gpc, mem = c.c_int(), c.c_int(), c.c_int(), c.c_int()
    check(nvml.nvmlDeviceGetGpcClkMinMaxVfOffset(h, c.byref(lo), c.byref(hi)), "range")
    check(nvml.nvmlDeviceGetGpcClkVfOffset(h, c.byref(gpc)), "gpc offset")
    check(nvml.nvmlDeviceGetMemClkVfOffset(h, c.byref(mem)), "mem offset")
    print(f"gpu{i}: gpc offset {gpc.value:+d} MHz (range {lo.value}..{hi.value}), mem offset {mem.value:+d} MHz")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, action="append", help="gpu index (repeatable; default all)")
    p.add_argument("--gpc", type=int, help="set GPC clock offset in MHz (0 resets)")
    p.add_argument("--mem", type=int, help="set memory clock offset in MHz (0 resets)")
    a = p.parse_args()
    check(nvml.nvmlInit_v2(), "init")
    count = c.c_uint()
    check(nvml.nvmlDeviceGetCount_v2(c.byref(count)), "count")
    for i in a.gpu or range(count.value):
        if a.gpc is not None:
            check(nvml.nvmlDeviceSetGpcClkVfOffset(handle(i), a.gpc), f"gpu{i} set gpc")
        if a.mem is not None:
            check(nvml.nvmlDeviceSetMemClkVfOffset(handle(i), a.mem), f"gpu{i} set mem")
        show(i)


if __name__ == "__main__":
    main()
