# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E4M3 byte conversions on devices without native FP8 instructions."""

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

SOFTWARE_FP8 = tl.constexpr(
    current_platform.is_cuda() and not current_platform.has_device_capability(89)
)


@triton.jit
def _encode_e4m3_software(x):
    """Round FP32 to nearest-even E4M3FN, saturating overflow to finite values."""
    x = x.to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    magnitude_bits = bits & 0x7FFFFFFF
    clipped = tl.minimum(magnitude_bits, 0x43E00000)  # 448.0
    # Drop 20 mantissa bits; carry propagates across exponent boundaries.
    normal = ((clipped + 0x7FFFF + ((clipped >> 20) & 1)) >> 20) - 960
    # E4M3 subnormals have a fixed 2^-9 step. Adding 2^14 rounds to that
    # step in FP32, including the tie at zero and the smallest normal.
    small = clipped.to(tl.float32, bitcast=True) + 16384.0
    subnormal = small.to(tl.uint32, bitcast=True) - 0x46800000
    payload = tl.where(clipped < 0x3C800000, subnormal, normal)
    payload = tl.where(magnitude_bits > 0x7F800000, 127, payload)
    return (sign | payload).to(tl.uint8)


@triton.jit
def encode_fp8(x, use_fnuz: tl.constexpr = False):
    """Return encoded bytes; retain native conversions on supported devices."""
    if use_fnuz:
        return x.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    elif SOFTWARE_FP8:
        return _encode_e4m3_software(x)
    else:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def decode_fp8(bits, use_fnuz: tl.constexpr = False):
    """Decode stored FP8 bytes to FP32, preserving signed zero and NaNs."""
    if use_fnuz:
        return bits.to(tl.uint8).to(tl.float8e4b8, bitcast=True).to(tl.float32)
    elif SOFTWARE_FP8:
        # E4M3FN values, including subnormals, are exact in FP16. Correct
        # the exponent bias in FP16 to avoid wide intermediate registers.
        bits = bits.to(tl.uint16)
        payload = bits & 127
        half_bits = (payload << 7) | ((bits & 128) << 8)
        value = half_bits.to(tl.float16, bitcast=True)
        value = value * tl.full((), 256.0, tl.float16)
        return tl.where(payload == 127, float("nan"), value.to(tl.float32))
    else:
        return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
