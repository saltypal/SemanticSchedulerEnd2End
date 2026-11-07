"""Corrected modern-runtime SwinJSCC adapter."""

from Codec.Models.SwinJSCC.adapter import SwinJSCCAdapter, register_swinjscc
from Codec.Models.SwinJSCC.swin_config import SwinJSCCConfig

__all__ = ["SwinJSCCAdapter", "SwinJSCCConfig", "register_swinjscc"]
