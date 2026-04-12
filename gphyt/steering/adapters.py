"""Backend adapters for GPhyT activation steering."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import torch


try:
    import pyvene as pv
except ImportError:  # pragma: no cover - optional dependency
    pv = None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    while True:
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        break
    return current


def _build_config_shim(model: torch.nn.Module) -> SimpleNamespace:
    hidden_dim = int(model.attention_blocks[0].norm1.normalized_shape[0])

    class _Config(SimpleNamespace):
        def to_dict(self):
            return dict(self.__dict__)

    return _Config(
        hidden_dim=hidden_dim,
        hidden_size=hidden_dim,
        d_model=hidden_dim,
        num_hidden_layers=len(model.attention_blocks),
        n_layer=len(model.attention_blocks),
    )


def _ensure_pyvene_ready(model: torch.nn.Module) -> torch.nn.Module:
    model = unwrap_model(model)
    if not hasattr(model, "config"):
        model.config = _build_config_shim(model)
    if pv is not None:
        model_type = type(model)
        if model_type not in pv.type_to_dimension_mapping:
            pv.type_to_dimension_mapping[model_type] = {}
        for layer_id in list_layer_ids(model):
            pv.type_to_dimension_mapping[model_type][_component_name(layer_id)] = (
                "hidden_dim",
            )
    return model


def list_layer_ids(model: torch.nn.Module) -> list[str]:
    model = unwrap_model(model)
    if not hasattr(model, "attention_blocks"):
        raise ValueError("Model does not expose attention_blocks")
    return [f"block_out:{idx}" for idx in range(len(model.attention_blocks))]


def _parse_layer_id(layer_id: str) -> int:
    prefix, _, suffix = layer_id.partition(":")
    if prefix != "block_out" or not suffix.isdigit():
        raise ValueError(f"Unsupported layer id {layer_id}")
    return int(suffix)


def get_block_module(model: torch.nn.Module, layer_id: str) -> torch.nn.Module:
    model = unwrap_model(model)
    index = _parse_layer_id(layer_id)
    return model.attention_blocks[index]


def _component_name(layer_id: str) -> str:
    index = _parse_layer_id(layer_id)
    return f"attention_blocks.{index}.output"


def _broadcast_direction(direction: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    direction = direction.to(device=activation.device, dtype=activation.dtype)
    if direction.ndim == 1:
        return direction.view(*([1] * (activation.ndim - 1)), direction.shape[0])
    if direction.ndim == activation.ndim - 1:
        return direction.unsqueeze(0)
    if direction.ndim == activation.ndim:
        return direction
    raise ValueError(
        f"Cannot broadcast direction shape {tuple(direction.shape)} "
        f"to activation shape {tuple(activation.shape)}"
    )


class SteeringBackend(Protocol):
    name: str

    def collect_activation(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor: ...

    def run_with_direction(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
        direction: torch.Tensor,
        scale: float,
    ) -> torch.Tensor: ...

    def run_with_noop(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass
class HookBackend:
    name: str = "hook"

    def collect_activation(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}
        module = get_block_module(model, layer_id)

        def _capture(_module, _args, output):
            captured["activation"] = output.detach().clone()

        handle = module.register_forward_hook(_capture)
        try:
            model(x)
        finally:
            handle.remove()

        return captured["activation"]

    def run_with_direction(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
        direction: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        module = get_block_module(model, layer_id)

        def _intervene(_module, _args, output):
            delta = _broadcast_direction(direction, output) * float(scale)
            return output + delta

        handle = module.register_forward_hook(_intervene)
        try:
            return model(x)
        finally:
            handle.remove()

    def run_with_noop(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        module = get_block_module(model, layer_id)

        def _identity(_module, _args, output):
            return output

        handle = module.register_forward_hook(_identity)
        try:
            return model(x)
        finally:
            handle.remove()


if pv is not None:  # pragma: no branch
    class _TensorCollectIntervention(pv.CollectIntervention):
        def __init__(self, **kwargs):
            kwargs.setdefault("keep_last_dim", True)
            super().__init__(**kwargs)


    class _TensorAdditionIntervention(pv.Intervention):
        def __init__(self, **kwargs):
            kwargs.setdefault("keep_last_dim", True)
            super().__init__(**kwargs)

        def forward(self, base, source=None, subspaces=None, **kwargs):
            source = source if self.source_representation is None else self.source_representation
            return base + source.to(device=base.device, dtype=base.dtype)


    class _IdentityIntervention(pv.Intervention):
        def __init__(self, **kwargs):
            kwargs.setdefault("keep_last_dim", True)
            super().__init__(**kwargs)

        def forward(self, base, source=None, subspaces=None, **kwargs):
            return base


@dataclass
class PyveneBackend:
    name: str = "pyvene"

    @staticmethod
    def is_available() -> bool:
        return pv is not None

    def _build_model(
        self,
        model: torch.nn.Module,
        layer_id: str,
        intervention_type,
        source_representation: torch.Tensor | None = None,
    ):
        if pv is None:  # pragma: no cover - guarded by tests
            raise RuntimeError("pyvene is not installed")
        model = _ensure_pyvene_ready(model)
        representation = {
            "component": _component_name(layer_id),
            "intervention_type": intervention_type,
        }
        if source_representation is not None:
            representation["source_representation"] = source_representation.detach()
        config = pv.IntervenableConfig(representation)
        return pv.IntervenableModel(config, model=model)

    @staticmethod
    def _extract_intervened_output(output):
        if hasattr(output, "intervened_outputs"):
            return output.intervened_outputs
        if isinstance(output, tuple):
            return output[-1]
        return output

    @staticmethod
    def _extract_collected_activation(output) -> torch.Tensor:
        activations = output.collected_activations
        if isinstance(activations, list):
            if len(activations) != 1:
                raise ValueError(f"Expected one collected activation, got {len(activations)}")
            return activations[0]
        return activations

    def collect_activation(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        intervenable = self._build_model(model, layer_id, _TensorCollectIntervention)
        output = intervenable(base={"x": x}, return_dict=True)
        activation = self._extract_collected_activation(output)
        return activation.detach().clone()

    def run_with_direction(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
        direction: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        activation = self.collect_activation(model, layer_id, x)
        source_representation = _broadcast_direction(direction, activation) * float(scale)
        intervenable = self._build_model(
            model,
            layer_id,
            _TensorAdditionIntervention,
            source_representation=source_representation,
        )
        output = intervenable(base={"x": x}, return_dict=True)
        return self._extract_intervened_output(output)

    def run_with_noop(
        self,
        model: torch.nn.Module,
        layer_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        intervenable = self._build_model(model, layer_id, _IdentityIntervention)
        output = intervenable(base={"x": x}, return_dict=True)
        return self._extract_intervened_output(output)


def get_backend(name: str = "auto") -> SteeringBackend:
    normalized = name.lower()
    if normalized == "hook":
        return HookBackend()
    if normalized == "pyvene":
        if not PyveneBackend.is_available():
            raise RuntimeError("Requested pyvene backend, but pyvene is not installed")
        return PyveneBackend()
    if normalized == "auto":
        if PyveneBackend.is_available():
            return PyveneBackend()
        return HookBackend()
    raise ValueError(f"Unknown backend {name}")
