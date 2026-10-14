"""Base64 rendering for embedding vectors."""

from __future__ import annotations

import base64
import struct

FLOAT32_LITTLE_ENDIAN = "<{count}f"
"""The `struct` format base64 vectors are packed with: little-endian float32.

Named rather than inlined because the byte order is this route's one silent
corruption risk - a big-endian pack, or a float64 one, returns a base64 string
that decodes to plausible numbers rather than to an error. Pinned against the
IEEE-754 bytes of `1.0` by `test_to_base64_is_little_endian_float32`."""


def to_base64(vector: tuple[float, ...]) -> str:
    """Render one vector as the OpenAI SDK decodes it: little-endian float32, base64.

    The SDK sends `encoding_format: "base64"` by default whenever numpy is
    installed, so this is the format most real clients will actually receive. Any other byte order or width decodes cleanly into
    wrong numbers, which is why the encoding is pinned against the bytes of
    `1.0` rather than only round-tripped (`test_to_base64_is_little_endian_float32`).
    """
    packed = struct.pack(FLOAT32_LITTLE_ENDIAN.format(count=len(vector)), *vector)
    return base64.b64encode(packed).decode("ascii")
