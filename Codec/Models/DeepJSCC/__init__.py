"""DeepJSCC integration components."""

from Codec.Models.DeepJSCC.adapter import (
    DeepJSCCAdapter,
    register_deepjscc,
)
from Codec.Models.DeepJSCC.channel_wrapper import (
    DeepJSCCChannelWrapper,
)

__all__ = [
    "DeepJSCCAdapter",
    "DeepJSCCChannelWrapper",
    "register_deepjscc",
]