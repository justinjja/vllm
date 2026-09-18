# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import (
    FP8_DTYPE,
    ref_dynamic_per_tensor_fp8_quant,
    ref_dynamic_per_token_quant,
)
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_compat import (
    _encode_e4m3_software,
    decode_fp8,
    encode_fp8,
)
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SCALE_UBS = [True, False]
SEEDS = [0]


@triton.jit
def _encode_e4m3_test_kernel(X, Y, Z, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _encode_e4m3_software(x), i < N)
    tl.store(Z + i, encode_fp8(x), i < N)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA FP8 conversion")
def test_e4m3_software_rounding_and_saturation():
    """Check ties and adjacent FP32 values, all BF16 values, and raw FP32 bits."""
    finite = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (finite[:-1] + finite[1:]) / 2
    near = torch.cat(
        [
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
        ]
    )
    generator = torch.Generator().manual_seed(170)
    random_bits = torch.randint(
        -(2**31), 2**31, (65536,), generator=generator, dtype=torch.int32
    )
    all_bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    x_cpu = torch.cat(
        [
            finite,
            -finite,
            near,
            -near,
            all_bf16.view(torch.bfloat16).float(),
            random_bits.view(torch.float32),
        ]
    )
    expected = x_cpu.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    x = x_cpu.cuda()
    software = torch.empty_like(x, dtype=torch.uint8)
    dispatched = torch.empty_like(software)
    _encode_e4m3_test_kernel[(triton.cdiv(x.numel(), 256),)](
        x, software, dispatched, x.numel()
    )
    finite_mask = ~x_cpu.isnan()
    for actual in (software.cpu(), dispatched.cpu()):
        torch.testing.assert_close(
            actual[finite_mask], expected[finite_mask], rtol=0, atol=0
        )
        assert ((actual[~finite_mask] & 127) == 127).all()


@triton.jit
def _decode_e4m3_test_kernel(X, Y):
    i = tl.arange(0, 256)
    tl.store(Y + i, decode_fp8(tl.load(X + i)))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA FP8 conversion")
def test_e4m3_byte_decode_preserves_values_and_zero_sign():
    bits = torch.arange(256, device="cuda", dtype=torch.uint8)
    expected = bits.cpu().view(torch.float8_e4m3fn).float()
    actual = torch.empty(256, device="cuda", dtype=torch.float32)
    _decode_e4m3_test_kernel[(1,)](bits, actual)
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        torch.signbit(actual[expected == 0]), torch.signbit(expected[expected == 0])
    )


def opcheck_fp8_quant(
    output,
    input,
    scale=None,
    scale_ub=None,
    use_per_token_if_dynamic=False,
    group_shape=None,
):
    if scale is not None:
        opcheck(
            torch.ops._C.static_scaled_fp8_quant,
            (output, input, scale, group_shape),
        )
    elif use_per_token_if_dynamic:
        scale = torch.empty(
            (input.shape[0], 1), device=input.device, dtype=torch.float32
        )
        opcheck(
            torch.ops._C.dynamic_per_token_scaled_fp8_quant,
            (output, input, scale, scale_ub),
        )
    else:
        scale = torch.empty(
            (input.numel() // input.shape[-1], 1),
            device=input.device,
            dtype=torch.float32,
        )
        opcheck(torch.ops._C.dynamic_scaled_fp8_quant, (output, input, scale))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("do_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, do_scale_ub: bool, seed: int
) -> None:
    set_random_seed(seed)

    x = (
        torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") + 1e-6
    )  # avoid nans

    scale_ub = (
        torch.mean(x).to(dtype=torch.float32, device="cuda") if do_scale_ub else None
    )
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, FP8_DTYPE, scale_ub)
    ops_out, ops_scales = ops.scaled_fp8_quant(
        x, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )

    torch.testing.assert_close(ref_scales, ops_scales)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x, None, scale_ub, use_per_token_if_dynamic=True)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int
) -> None:
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    torch.testing.assert_close(ref_scale, ops_scale)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x)


# Regression test for a case with large activations where an int32 index cannot
# represent the number of elements.
@torch.inference_mode()
@pytest.mark.parametrize("seed", SEEDS)
def test_fp8_quant_large(seed: int) -> None:
    set_random_seed(seed)

    num_tokens = 1024000  # Mistral-Nemo's max_position_embeddings
    hidden_size = 1152  # Smallest hidden_size to reproduce the error
    dtype = torch.bfloat16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, _ = ops.scaled_fp8_quant(x, scale)

    # Minimize memory footprint in this test by freeing x and upconverting
    # the outputs in place. (torch.allclose does not support fp8)
    del x
    ref_out = ref_out.to(dtype=dtype)
    ops_out = ops_out.to(dtype=dtype)

    torch.testing.assert_close(ref_out, ops_out)


# Test static FP8 quantization with 2D group scales
GROUP_SHAPES_2D = [
    (-1, -1),  # Per-tensor
    (-1, 1),  # Per-channel
    (1, -1),  # Per-token
    (-1, 128),  # Per-head quantization
    (1, 128),  # DeepSeek-style per-token-per-group (group_m=1, group_n=128)
    (128, 128),  # DeepSeek-style block quantization
    (1, 64),  # Smaller group size
    (1, 16),  # Small group (scalar path in kernel)
    (4, 256),  # Non-trivial both dimensions
]
# Use sizes divisible by all group shapes
NUM_TOKENS_GROUP = [128, 512]
HIDDEN_SIZES_GROUP = [256, 1024, 2048]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("group_shape", GROUP_SHAPES_2D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_static_fp8_quant_group_2d(
    num_tokens: int,
    hidden_size: int,
    group_shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int,
) -> None:
    """Test static FP8 quantization with 2D group scales using scaled_quantize."""
    # Normalize group_shape (-1 means full extent)
    norm_group_m = num_tokens if group_shape[0] == -1 else group_shape[0]
    norm_group_n = hidden_size if group_shape[1] == -1 else group_shape[1]

    # Skip if sizes are not divisible by group shape
    if num_tokens % norm_group_m != 0 or hidden_size % norm_group_n != 0:
        pytest.skip(
            f"Skipping: ({num_tokens}, {hidden_size}) not divisible by "
            f"group_shape ({group_shape[0]}, {group_shape[1]})"
        )

    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = scaled_quantize(
        x, group_shape, current_platform.fp8_dtype(), compute_dtype=torch.float32
    )
    ops_out, ops_scale = ops.scaled_fp8_quant(x, scale=scale, group_shape=group_shape)

    torch.testing.assert_close(scale, ops_scale)
    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=1.2e-1, atol=1e-5)

    opcheck_fp8_quant(ops_out, x, scale=scale)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("group_shape", [(1, -1), (-1, 1)])  # per-token, per-channel
@torch.inference_mode()
def test_static_fp8_quant_1d_scale(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    group_shape: tuple[int, int],
) -> None:
    """Test static FP8 quantization with 1D scale (per-token or per-channel)."""
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale_2d = scaled_quantize(
        x, group_shape, FP8_DTYPE, compute_dtype=torch.float32
    )

    # Flatten scale to 1D for testing 1D scale path
    scale_1d = scale_2d.flatten()
    ops_out, ops_scale = ops.scaled_fp8_quant(
        x, scale=scale_1d, group_shape=group_shape
    )

    torch.testing.assert_close(scale_1d, ops_scale)
    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=0.12, atol=0.0)

    opcheck_fp8_quant(ops_out, x, scale=scale_1d, group_shape=group_shape)
