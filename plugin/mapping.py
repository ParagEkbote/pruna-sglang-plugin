from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pruna_config import PrunaConfig


@dataclass(frozen=True)
class SGLangMapping:
    """SGLang configuration derived from a Pruna configuration."""

    model_dtype: str | None = None
    quantization: str | None = None
    extra_args: dict[str, Any] | None = None


def map_to_sglang(pruna_config: PrunaConfig) -> SGLangMapping:
    """Map a compatible Pruna configuration to SGLang settings."""

    model_dtype = None
    quantization = None
    extra_args: dict[str, Any] = {}

    if pruna_config.quantizer == "half":
        model_dtype = "half"

    elif pruna_config.quantizer == "gptq":
        quantization = "gptq"

    elif pruna_config.quantizer == "awq":
        quantization = "awq"

    return SGLangMapping(
        model_dtype=model_dtype,
        quantization=quantization,
        extra_args=extra_args or None,
    )