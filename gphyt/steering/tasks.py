"""Task registry for GPhyT steering experiments."""

from __future__ import annotations

from dataclasses import dataclass

from gphyt.steering.features import FEATURE_NAMES, required_fields_for_feature


FIELD_NAMES_BY_INDEX = {
    0: "pressure",
    1: "density",
    2: "temperature",
    3: "velocity_x",
    4: "velocity_y",
}

DATASET_FIELDS = {
    "cylinder_sym_flow_water": (0, 3, 4),
    "cylinder_pipe_flow_water": (0, 3, 4),
    "object_periodic_flow_water": (0, 3, 4),
    "object_sym_flow_water": (0, 3, 4),
    "object_sym_flow_air": (0, 3, 4),
    "heated_object_pipe_flow_air": (0, 1, 2, 3, 4),
    "cooled_object_pipe_flow_air": (0, 1, 2, 3, 4),
    "rayleigh_benard_obstacle": (0, 1, 2, 3, 4),
    "twophase_flow": (0, 1, 3, 4),
    "rayleigh_benard": (0, 1, 3, 4),
    "shear_flow": (0, 3, 4),
    "euler_multi_quadrants_periodicbc": (0, 1, 3, 4),
    "acoustic_scattering_inclusions": (0, 3, 4),
    "turbulent_radiative_layer_2d": (0, 1, 3, 4),
    "supersonic_flow": (0, 1, 3, 4),
    "open_obj_water": (0, 3, 4),
    "euler_multi_quadrants_openbc": (0, 1, 3, 4),
}


@dataclass(frozen=True)
class SteeringTask:
    task_id: str
    task_type: str
    required_fields: tuple[str, ...]
    feature_name: str | None = None
    positive_datasets: tuple[str, ...] = ()
    negative_datasets: tuple[str, ...] = ()
    eligible_datasets: tuple[str, ...] = ()
    transfer_datasets: tuple[str, ...] = ()


def canonicalize_dataset_name(dataset_name: str) -> str:
    canonical = dataset_name.strip().lower()
    canonical = canonical.replace("-", "_")
    return canonical


def get_dataset_fields(dataset_name: str) -> tuple[str, ...]:
    canonical = canonicalize_dataset_name(dataset_name)
    if canonical not in DATASET_FIELDS:
        raise KeyError(f"Unknown dataset {dataset_name}")
    return tuple(FIELD_NAMES_BY_INDEX[idx] for idx in DATASET_FIELDS[canonical])


def dataset_supports_fields(dataset_name: str, required_fields: tuple[str, ...]) -> bool:
    available = set(get_dataset_fields(dataset_name))
    return set(required_fields).issubset(available)


def _build_feature_task(feature_name: str) -> SteeringTask:
    required_fields = required_fields_for_feature(feature_name)
    eligible_datasets = tuple(
        dataset_name
        for dataset_name in sorted(DATASET_FIELDS)
        if dataset_supports_fields(dataset_name, required_fields)
    )
    return SteeringTask(
        task_id=feature_name,
        task_type="feature",
        feature_name=feature_name,
        required_fields=required_fields,
        eligible_datasets=eligible_datasets,
        transfer_datasets=("turbulent_radiative_layer_2d",)
        if "turbulent_radiative_layer_2d" in eligible_datasets
        else (),
    )


TASKS = {
    "regime_shear_vs_euler": SteeringTask(
        task_id="regime_shear_vs_euler",
        task_type="regime",
        required_fields=("pressure", "density", "velocity_x", "velocity_y"),
        positive_datasets=("shear_flow",),
        negative_datasets=("euler_multi_quadrants_periodicbc",),
        transfer_datasets=("turbulent_radiative_layer_2d",),
    ),
    "regime_shear_vs_rayleigh": SteeringTask(
        task_id="regime_shear_vs_rayleigh",
        task_type="regime",
        required_fields=("pressure", "density", "velocity_x", "velocity_y"),
        positive_datasets=("shear_flow",),
        negative_datasets=("rayleigh_benard",),
        transfer_datasets=("turbulent_radiative_layer_2d",),
    ),
}

for _feature_name in FEATURE_NAMES:
    TASKS[_feature_name] = _build_feature_task(_feature_name)


def get_task(task_id: str) -> SteeringTask:
    canonical = canonicalize_dataset_name(task_id)
    if canonical in TASKS:
        return TASKS[canonical]
    raise KeyError(f"Unknown steering task {task_id}")


def get_tasks(task_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, SteeringTask]:
    if task_ids is None:
        return dict(TASKS)
    return {task_id: get_task(task_id) for task_id in task_ids}
