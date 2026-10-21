"""Model registry. Importing this package registers the built-in models and
any plugins found in custom_models/."""
from __future__ import annotations

from .base import SRModel, auto_device, register, registry
from . import baselines, satlas, sen2sr_models  # noqa: F401  (registration side effects)
from .custom import discover

PLUGIN_ERRORS = discover()

__all__ = ["SRModel", "register", "registry", "auto_device", "PLUGIN_ERRORS", "discover"]
