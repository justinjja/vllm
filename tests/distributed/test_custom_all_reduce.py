# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import os
import random
from datetime import timedelta

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cmp_pair_tree import (
    CmpPairTreeAllReduce,
    pair_tree_config,
    tree_rank_order,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms.interface import DeviceCapability

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@pytest.mark.parametrize("devices", [[0, 1, 2, 3], [7, 4, 6, 5]])
def test_pair_tree_preserves_physical_pairs_after_rank_remapping(monkeypatch, devices):
    monkeypatch.setattr(
        car.current_platform, "device_control_id_to_physical_device_id", int
    )
    rows, groups = pair_tree_config(
        {
            "cmp_pair_tree_max_rows": 6,
            "cmp_pair_tree_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        }
    )
    order = tree_rank_order(devices, groups)
    assert rows == 6
    assert [devices[rank] for rank in order] == sorted(devices)
    assert tree_rank_order([0, 1, 6, 7], groups) == []


@pytest.mark.parametrize(
    "rows,groups",
    [
        (True, [[0, 1, 2, 3]]),
        (7, [[0, 1, 2, 3]]),
        (6, [[0, 1, 2]]),
        (6, [[0, 1, 2, 3], [2, 3, 4, 5]]),
    ],
)
def test_pair_tree_rejects_ambiguous_configuration(monkeypatch, rows, groups):
    monkeypatch.setattr(
        car.current_platform, "device_control_id_to_physical_device_id", int
    )
    with pytest.raises(ValueError):
        pair_tree_config(
            {"cmp_pair_tree_max_rows": rows, "cmp_pair_tree_groups": groups}
        )


def _pair_tree_worker(rank, physical_group, rendezvous):
    """Check actual IPC, asymmetric layouts, graph replay and workspace reuse."""
    torch.set_num_threads(1)
    # Deliberately scramble process ranks relative to physical PIX pairs.
    devices = [physical_group[i] for i in (3, 0, 2, 1)]
    visible_physical = [
        car.current_platform.visible_device_id_to_physical_device_id(i)
        for i in range(torch.accelerator.device_count())
    ]
    device = torch.device("cuda", visible_physical.index(devices[rank]))
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    group = dist.group.WORLD
    # A missing library on one worker must disable the whole group.
    from unittest.mock import Mock, patch

    import vllm.distributed.device_communicators.cmp_pair_tree as module
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    loader = module._load_library

    def missing_on_one_rank():
        if rank == 1:
            raise OSError("injected missing optional library")
        return loader()

    with patch.object(module, "_load_library", missing_on_one_rank):
        disabled = CmpPairTreeAllReduce(group, device, [physical_group], 6)
        assert disabled.disabled
        disabled.destroy()

    def fail_second_import():
        library = loader()
        proxy = Mock(wraps=library)
        imports = 0

        def open_handle(*args):
            nonlocal imports
            imports += 1
            if devices[rank] == physical_group[0] and imports == 2:
                return 1  # cudaErrorInvalidValue after one successful import.
            return library.cmp_pair_tree_open(*args)

        proxy.cmp_pair_tree_open.side_effect = open_handle
        return proxy

    with patch.object(module, "_load_library", fail_second_import):
        disabled = CmpPairTreeAllReduce(group, device, [physical_group], 6)
        assert disabled.disabled and disabled._owned == 0
        disabled.destroy()

    for generation in range(2):
        comm = CmpPairTreeAllReduce(group, device, [physical_group], 6)
        assert not comm.disabled
        # Seed the state just below 2**32 to catch truncated sequence numbers.
        # This writes only locally owned memory before any kernels are queued.
        cuda = CudaRTLibrary()
        initial = (ctypes.c_uint64 * 386)(*([2**32 - 2] * 386))
        torch.accelerator.synchronize()
        cuda.cudaMemcpy(
            ctypes.c_void_p(comm._owned + comm.capacity * 14),
            ctypes.cast(initial, ctypes.c_void_p),
            ctypes.sizeof(initial),
        )
        dist.barrier()
        for rows in (1, 3, 6):
            torch.manual_seed(1987 + rank + generation * 11)
            values = torch.randn(rows, 6144, device=device, dtype=torch.bfloat16)
            if rank == 0:
                storage = torch.empty(
                    rows, 6144 * 2, device=device, dtype=torch.bfloat16
                )
                value = storage[:, ::2]
                value.copy_(values)
            elif rank == 1:
                storage = torch.empty(
                    values.numel() + 1, device=device, dtype=torch.bfloat16
                )
                value = storage[1:].view_as(values)
                value.copy_(values)
            else:
                value = values

            def reference(value=value):
                peers = [torch.empty_like(value, device="cpu") for _ in range(4)]
                dist.all_gather(peers, value.cpu().contiguous())
                ordered = [peers[devices.index(i)].float() for i in physical_group]
                return ((ordered[0] + ordered[1]) + (ordered[2] + ordered[3])).to(
                    device=device, dtype=torch.bfloat16
                )

            expected = reference()
            for iteration in range(4):
                if rank == iteration:
                    torch.cuda._sleep(30000)
                result = comm.all_reduce(value)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with comm.capture(), torch.cuda.graph(graph):
                result = comm.all_reduce(value)
            for iteration in range(4):
                value.mul_(-1).add_(rank * 0.0625)
                expected = reference()
                graph.replay()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            graph.reset()
        assert not comm.should_use(
            torch.empty(7, 6144, device=device, dtype=torch.bfloat16)
        )
        comm.destroy()
        comm.destroy()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "physical_group",
    json.loads(os.getenv("VLLM_TEST_CMP_PAIR_TREE_GROUPS", "[]")),
)
def test_pair_tree_ipc_graph_replay_and_lifetime(physical_group, tmp_path):
    """Opt in with ordered physical pairs; run exclusively on idle GPUs."""
    torch.multiprocessing.spawn(
        _pair_tree_worker,
        nprocs=4,
        args=(physical_group, f"file://{tmp_path / 'rendezvous'}"),
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int8, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_custom_allreduce_filters_dtype(
    dtype: torch.dtype,
    expected: bool,
) -> None:
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator.world_size = 2
    communicator.max_size = 1024
    communicator._ptr = 0

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype)) is expected


@pytest.mark.parametrize("pcie_mesh", [False, True])
@pytest.mark.parametrize("numel", [5120, 4 * 5120, 65536])
def test_pcie_allreduce_keeps_large_messages_on_fallback(pcie_mesh, numel):
    """Only an opted-in mesh may reduce small PCIe tensors with the IPC kernel."""
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator.world_size = 4
    communicator.fully_connected = False
    communicator.pcie_mesh = pcie_mesh
    communicator.max_size = 256 * 1024
    communicator._ptr = 0
    tensor = torch.empty(numel, dtype=torch.float32)
    assert communicator.should_custom_ar(tensor) == (
        pcie_mesh and tensor.nbytes < communicator.max_size
    )


@pytest.mark.parametrize(
    "devices,capability,expected",
    [
        ([0, 1, 2, 3], DeviceCapability(8, 0), 262144),
        ([4, 5, 6, 7], DeviceCapability(8, 0), 0),
        (list(range(8)), DeviceCapability(8, 0), 0),
        ([0, 1, 2, 3], DeviceCapability(9, 0), 0),
    ],
)
def test_pcie_allreduce_rejects_unqualified_socket_and_world(
    monkeypatch, devices, capability, expected
):
    """A tested four-GPU IPC group must not enable the other socket or TP8."""
    monkeypatch.setattr(
        car.current_platform, "device_control_id_to_physical_device_id", int
    )
    additional = {
        "cmp_pcie_allreduce_max_bytes": 262144,
        "cmp_pcie_allreduce_devices": [0, 1, 2, 3],
    }
    assert car._pcie_mesh_limit(additional, devices, capability, False) == expected


@pytest.mark.parametrize(
    ("major", "local_multicast", "expected"),
    [
        (8, True, False),
        (9, True, False),
        (10, False, False),
        (10, True, True),
    ],
)
def test_cross_node_mnnvl_gate_checks_generation_and_multicast(
    monkeypatch,
    major,
    local_multicast,
    expected,
):
    def has_device_capability(capability, device_id):
        assert capability == 100
        assert device_id == 3
        return major >= 10

    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        has_device_capability,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: local_multicast,
    )
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)

    assert car._group_can_attempt_mnnvl(object(), torch.device("cuda:3")) is expected


def test_cross_node_mnnvl_gate_requires_support_on_every_rank(monkeypatch):
    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: True,
    )

    def report_unsupported_peer(support, **_kwargs):
        support.zero_()

    monkeypatch.setattr(car.dist, "all_reduce", report_unsupported_peer)

    assert not car._group_can_attempt_mnnvl(object(), torch.device("cuda:0"))


def test_local_multicast_support_rejects_non_cuda(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: False)

    assert not car._has_local_multicast_support(torch.device("cuda:0"))


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
