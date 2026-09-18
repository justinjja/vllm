# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured CMP memory, PCIe and dense tensor-core baselines."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def read_stream(X, SUM, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    tl.store(SUM + row, tl.sum(values, 0))


def graph_ms(fn):
    return triton.testing.do_bench_cudagraph(fn, rep=100)


def main(args):
    torch.set_num_threads(1)
    torch.accelerator.set_device_index(args.device)
    # UUIDs preserve the physical NUMA mapping under CUDA_VISIBLE_DEVICES.
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    buses = {
        uuid.strip().removeprefix("GPU-"): bus.strip().lower()[-12:]
        for uuid, bus in (row.split(",") for row in gpu_rows)
    }
    bus = buses[str(torch.cuda.get_device_properties(args.device).uuid)]
    node = int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text())
    if node >= 0:
        cpus = []
        for part in (
            Path(f"/sys/devices/system/node/node{node}/cpulist")
            .read_text()
            .strip()
            .split(",")
        ):
            endpoints = [int(x) for x in part.split("-")]
            cpus.extend(range(endpoints[0], endpoints[-1] + 1))
        os.sched_setaffinity(0, set(cpus) & os.sched_getaffinity(0))
    count = args.mib * 1024 * 1024 // 4
    x = torch.full((count,), 1.25, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    sums = torch.empty(triton.cdiv(count, 4096), device="cuda")

    def read(x=x, sums=sums):
        read_stream[(triton.cdiv(count, 4096),)](x, sums, count, 4096)

    read()
    torch.testing.assert_close(sums, torch.full_like(sums, 4096 * 1.25), rtol=0, atol=0)
    read_ms = graph_ms(read)
    copy_ms = graph_ms(lambda y=y, x=x: y.copy_(x))
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    host = torch.full((count,), 1.25, pin_memory=True)
    h2d_ms = graph_ms(lambda y=y, host=host: y.copy_(host, non_blocking=True))
    d2h_ms = graph_ms(lambda host=host, y=y: host.copy_(y, non_blocking=True))
    assert bool((host == 1.25).all())
    results = {
        "read_stream_gb_s": x.nbytes / (read_ms * 1e6),
        "copy_read_plus_write_gb_s": 2 * x.nbytes / (copy_ms * 1e6),
        "h2d_gb_s": x.nbytes / (h2d_ms * 1e6),
        "d2h_gb_s": x.nbytes / (d2h_ms * 1e6),
    }
    del x, y, host, sums, read
    mm = []
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        torch.mm(a, b, out=out)
        assert bool(torch.isfinite(out).all())
        elapsed = graph_ms(lambda a=a, b=b, out=out: torch.mm(a, b, out=out))
        mm.append({"n": n, "ms": elapsed, "tflops": 2 * n**3 / (elapsed * 1e9)})
        del a, b, out
    report = {
        "device_index": args.device,
        "device": str(torch.cuda.get_device_properties(args.device)),
        "numa_node": node,
        "bus_id": bus,
        "buffer_mib": args.mib,
        "torch": torch.__version__,
        "bandwidth": results,
        "bf16_gemm": mm,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mib", type=int, default=1024)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
