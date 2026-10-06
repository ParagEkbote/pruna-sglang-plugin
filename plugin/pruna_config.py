from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json


SMASH_CONFIG_FILE_NAME = "smash_config.json"


@dataclass(frozen=True)
class PrunaConfig:
    compiler: str | None
    quantizer: str | None
    batcher: str | None
    cacher: str | None
    factorizer: str | None
    kernel: str | None
    pruner: str | None

    raw: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "PrunaConfig":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Pruna configuration not found: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object in {path}, "
                f"got {type(data).__name__}"
            )

        return cls(
            compiler=data.get("compiler"),
            quantizer=data.get("quantizer"),
            batcher=data.get("batcher"),
            cacher=data.get("cacher"),
            factorizer=data.get("factorizer"),
            kernel=data.get("kernel"),
            pruner=data.get("pruner"),
            raw=data,
        )

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "PrunaConfig":
        return cls.from_file(
            Path(model_path) / SMASH_CONFIG_FILE_NAME
        )

    @property
    def algorithms(self) -> dict[str, str]:
        """Return the configured Pruna optimization algorithms."""
        return {
            name: value
            for name in (
                "compiler",
                "quantizer",
                "batcher",
                "cacher",
                "factorizer",
                "kernel",
                "pruner",
            )
            if (value := getattr(self, name)) is not None
        }