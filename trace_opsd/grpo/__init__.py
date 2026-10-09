"""Trace-OPSD adapters for GRPO and solution-hindsight TRACE training."""

from .schema import (
    DATASET_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    DatasetSummary,
    DatasetValidationError,
    validate_dataset_directory,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "DatasetSummary",
    "DatasetValidationError",
    "validate_dataset_directory",
]
