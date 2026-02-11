"""Safety-Lens: MRI-style introspection for Hugging Face models."""

from safety_lens.core import SafetyLens, LensHooks
from safety_lens.vectors import STIMULUS_SETS

__version__ = "0.1.0"
__all__ = ["SafetyLens", "LensHooks", "STIMULUS_SETS"]
