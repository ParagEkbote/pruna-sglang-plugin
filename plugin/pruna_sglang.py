from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from .compatibility import SGLangEnvironment, check_compatibility
from .mapping import map_to_sglang
from .pruna_config import PrunaConfig


def configure_pruna_model(result, server_args, *args, **kwargs):
    """Apply Pruna configuration to SGLang ServerArgs."""

    pruna_config = PrunaConfig.from_model_path(
        server_args.model_path
    )

    compatibility = check_compatibility(
        pruna_config,
        SGLangEnvironment(),
    )

    if not compatibility.compatible:
        raise RuntimeError(
            "Pruna configuration is not compatible with SGLang: "
            f"{compatibility.reason}"
        )

    mapping = map_to_sglang(pruna_config)

    if mapping.model_dtype is not None:
        server_args.dtype = mapping.model_dtype

    if mapping.quantization is not None:
        server_args.quantization = mapping.quantization

    if mapping.extra_args:
        for name, value in mapping.extra_args.items():
            setattr(server_args, name, value)

    return result


def register():
    """Register the Pruna-SGLang integration hook."""

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.post_init",
        configure_pruna_model,
        HookType.AFTER,
    )