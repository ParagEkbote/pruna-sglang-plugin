from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .pruna_config import PrunaConfig


class CompatibilityStatus(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SGLangEnvironment:
    version: str | None = None
    torch_version: str | None = None
    torchao_version: str | None = None


@dataclass(frozen=True)
class CompatibilityResult:
    status: CompatibilityStatus
    reason: str

    @property
    def compatible(self) -> bool:
        return self.status in {
            CompatibilityStatus.SUPPORTED,
            CompatibilityStatus.EXPERIMENTAL,
        }


def check_compatibility(
    pruna_config: PrunaConfig,
    sglang: SGLangEnvironment,
) -> CompatibilityResult:
    """Check whether a Pruna artifact can be used with SGLang."""
    algorithms = pruna_config.algorithms

    if not algorithms:
        return CompatibilityResult(
            status=CompatibilityStatus.SUPPORTED,
            reason="No Pruna optimization requiring compatibility handling.",
        )

    if pruna_config.quantizer == "half":
        return CompatibilityResult(
            status=CompatibilityStatus.SUPPORTED,
            reason="Pruna half-precision model is compatible with SGLang.",
        )

    if pruna_config.quantizer == "gptq":
        return CompatibilityResult(
            status=CompatibilityStatus.EXPERIMENTAL,
            reason=(
                "Pruna GPTQ configuration maps to SGLang GPTQ, "
                "but artifact loading requires runtime validation."
            ),
        )

    if pruna_config.quantizer == "awq":
        return CompatibilityResult(
            status=CompatibilityStatus.EXPERIMENTAL,
            reason=(
                "Pruna AWQ configuration maps to SGLang AWQ, "
                "but artifact loading requires runtime validation."
            ),
        )

    return CompatibilityResult(
        status=CompatibilityStatus.UNSUPPORTED,
        reason=(
            "The Pruna optimization configuration is not currently "
            "supported by the SGLang plugin."
        ),
    )