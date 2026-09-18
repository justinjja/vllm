# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-stream SM80 pair-tree reductions for small GLM batches."""

import ctypes
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def pair_tree_config(additional: dict) -> tuple[int, list[list[int]]]:
    rows = additional.get("cmp_pair_tree_max_rows", 0)
    if rows == 0 and type(rows) is int:
        return 0, []
    if type(rows) is not int or not 1 <= rows <= 6:
        raise ValueError("cmp_pair_tree_max_rows must be an integer from 0 to 6")
    groups = additional.get("cmp_pair_tree_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("cmp_pair_tree_groups must list ordered four-GPU groups")
    result: list[list[int]] = []
    seen: set[int] = set()
    for group in groups:
        if not isinstance(group, list) or len(group) != 4:
            raise ValueError("Each cmp_pair_tree_groups entry must contain four GPUs")
        ids = [
            current_platform.device_control_id_to_physical_device_id(str(device))
            for device in group
        ]
        if len(set(ids)) != 4 or seen.intersection(ids):
            raise ValueError("cmp_pair_tree_groups must contain distinct GPUs")
        seen.update(ids)
        result.append(ids)
    return rows, result


def tree_rank_order(physical_ids: list[int], groups: list[list[int]]) -> list[int]:
    """Map configured physical pair order to this process group's rank order."""
    if len(physical_ids) == 4 and len(set(physical_ids)) == 4:
        for group in groups:
            if set(group) == set(physical_ids):
                return [physical_ids.index(device) for device in group]
    return []


def _load_library():
    paths = sorted(Path(__file__).resolve().parents[2].glob("_cmp_pair_tree*.so"))
    if len(paths) != 1:
        raise RuntimeError("Build the optional _cmp_pair_tree CMake target first")
    lib = ctypes.CDLL(str(paths[0]))
    if lib.cmp_pair_tree_abi_version() != 1:
        raise RuntimeError("Unsupported CMP pair-tree library ABI")
    lib.cmp_pair_tree_error.argtypes = [ctypes.c_int]
    lib.cmp_pair_tree_error.restype = ctypes.c_char_p
    lib.cmp_pair_tree_allocate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.cmp_pair_tree_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cmp_pair_tree_close.argtypes = [ctypes.c_void_p]
    lib.cmp_pair_tree_free.argtypes = [ctypes.c_void_p]
    lib.cmp_pair_tree_reduce.argtypes = (
        [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    return lib


class CmpPairTreeAllReduce:
    """Own IPC workspaces until all captured graphs have finished using them.

    Collective initialization and destruction use the CPU process group.
    Callers must serialize graph replays and eager calls on one execution
    stream. ``capture`` orders temporary capture streams against that stream.
    """

    def __init__(self, group, device, groups, max_rows):
        self.group = group
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Pair-tree requires an explicit CUDA device index")
        if type(max_rows) is not int or not 1 <= max_rows <= 6:
            raise ValueError("Pair-tree max_rows must be an integer from 1 to 6")
        self.max_rows = max_rows
        self.capacity = max_rows * 6144
        self.disabled = True
        self._owned = 0
        self._opened: list[int] = []
        self._closed = False
        self._stream = None
        self._lib = None
        self._pointers: dict[int, int] = {}
        self.rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if world_size != 4:
            self._closed = True
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            self._closed = True
            return
        physical_id = current_platform.visible_device_id_to_physical_device_id(
            self.device.index
        )
        physical_ids = [-1] * world_size
        dist.all_gather_object(physical_ids, physical_id, group=group)
        order = tree_rank_order(physical_ids, groups)
        error = None
        try:
            capability = current_platform.get_device_capability(self.device.index)
            if not order or capability is None or capability.to_int() != 80:
                raise RuntimeError("The TP group is not a configured SM80 pair tree")
            if torch.accelerator.current_device_index() != self.device.index:
                raise RuntimeError(
                    "Pair-tree device must be the worker's current device"
                )
            self._lib = _load_library()
        except (AttributeError, OSError, RuntimeError) as exc:
            error = str(exc)
        if not self._agree(error):
            self._closed = True
            return
        self.rank = order.index(self.rank)
        assert self._lib is not None
        handle = ctypes.create_string_buffer(self._lib.cmp_pair_tree_handle_size())
        pointer = ctypes.c_void_p()
        error = None
        try:
            self._check(
                self._lib.cmp_pair_tree_allocate(
                    ctypes.byref(pointer), handle, self.capacity
                )
            )
            assert pointer.value is not None
            self._owned = pointer.value
        except RuntimeError as exc:
            error = str(exc)
        if not self._agree(error):
            self.destroy()
            return
        handles = [None] * world_size
        dist.all_gather_object(handles, handle.raw, group=group)
        self._pointers[self.rank] = self._owned
        peers = [self.rank ^ 1]
        if self.rank % 2 == 0:
            peers.append(self.rank ^ 2)
        error = None
        try:
            for peer in peers:
                peer_handle = ctypes.create_string_buffer(handles[order[peer]])
                pointer = ctypes.c_void_p()
                self._check(
                    self._lib.cmp_pair_tree_open(ctypes.byref(pointer), peer_handle)
                )
                assert pointer.value is not None
                self._opened.append(pointer.value)
                self._pointers[peer] = pointer.value
        except RuntimeError as exc:
            error = str(exc)
        if not self._agree(error):
            self.destroy()
            return
        self.disabled = False
        # Exercise actual transfers, not merely reported peer capability.
        value = (torch.arange(6144, device=self.device) % 31).to(torch.bfloat16)
        value = (value + self.rank).view(1, 6144)
        output = self.all_reduce(value)
        expected = (torch.arange(6144, device=self.device) % 31) * 4 + 6
        error = (
            None
            if torch.equal(output.view(-1), expected)
            else "Peer transfer probe failed"
        )
        if not self._agree(error):
            self.destroy()
            return
        logger.info(
            "Enabled CMP pair-tree all-reduce on physical GPUs %s for <=%d rows",
            physical_ids,
            max_rows,
        )

    def _check(self, code):
        if code:
            assert self._lib is not None
            raise RuntimeError(self._lib.cmp_pair_tree_error(code).decode())

    def _agree(self, error):
        errors = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(errors, error, group=self.group)
        if any(errors):
            logger.warning("CMP pair-tree disabled for the whole TP group: %s", errors)
            return False
        return True

    def should_use(self, value):
        return (
            not self.disabled
            and value.device == self.device
            and value.dtype == torch.bfloat16
            and value.ndim == 2
            and value.shape[1] == 6144
            and 0 < value.shape[0] <= self.max_rows
        )

    def all_reduce(self, value):
        if not self.should_use(value):
            return None
        assert self._lib is not None
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not torch.cuda.is_current_stream_capturing():
            if self._stream is not None and stream != self._stream:
                raise RuntimeError(
                    "CMP pair-tree requires serialized use on one stream"
                )
            self._stream = stream
        # All ranks select the same backend even if their input layouts differ.
        if not value.is_contiguous():
            value = value.contiguous()
        elif value.data_ptr() % 16:
            value = value.clone()
        output = torch.empty_like(value)
        self._check(
            self._lib.cmp_pair_tree_reduce(
                value.data_ptr(),
                output.data_ptr(),
                self._owned,
                self._pointers[self.rank ^ 1],
                self._pointers.get(self.rank ^ 2, 0),
                value.numel(),
                self.capacity,
                self.rank,
                stream,
            )
        )
        return output

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        torch.accelerator.synchronize(self.device)
        self._stream = None
        try:
            yield
        finally:
            torch.accelerator.synchronize(self.device)
            self._stream = None

    def destroy(self):
        if self._closed:
            return
        assert self._lib is not None
        self.disabled = True
        torch.accelerator.synchronize(self.device)
        error = None
        for pointer in self._opened:
            code = self._lib.cmp_pair_tree_close(pointer)
            if code:
                error = self._lib.cmp_pair_tree_error(code).decode()
        self._opened.clear()
        # Every importer closes before any owner frees its allocation.
        if self._agree(error) and self._owned:
            self._check(self._lib.cmp_pair_tree_free(self._owned))
            self._owned = 0
        self._closed = True
