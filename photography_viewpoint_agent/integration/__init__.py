"""Optional integration with the independently installed video_guide package."""
from .guidance_adapter import build_target_package, to_guide_input

__all__ = ["build_target_package", "to_guide_input"]
