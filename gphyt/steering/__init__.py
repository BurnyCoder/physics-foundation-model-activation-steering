"""Activation steering utilities for GPhyT models."""

from gphyt.steering.adapters import (
    HookBackend,
    PyveneBackend,
    get_backend,
    get_block_module,
    list_layer_ids,
)
from gphyt.steering.eval import (
    ActivationCollection,
    aggregate_sweep_reports,
    collect_activation_dataset,
    fit_direction,
    fit_task_vectors,
    load_activation_collection,
    load_direction,
    run_scale_sweep,
    save_activation_collection,
    save_direction,
    write_sweep_report,
)
from gphyt.steering.features import (
    FEATURE_NAMES,
    compute_feature_batch,
    compute_features,
    fit_quartile_split,
    fit_zscore_stats,
    quartile_contrast_labels,
    required_fields_for_feature,
    zscore,
)
from gphyt.steering.tasks import (
    SteeringTask,
    canonicalize_dataset_name,
    dataset_supports_fields,
    get_task,
    get_tasks,
)
from gphyt.steering.visualize import extract_display_field, render_comparison_gif

__all__ = [
    "ActivationCollection",
    "FEATURE_NAMES",
    "HookBackend",
    "PyveneBackend",
    "SteeringTask",
    "aggregate_sweep_reports",
    "canonicalize_dataset_name",
    "collect_activation_dataset",
    "compute_feature_batch",
    "compute_features",
    "dataset_supports_fields",
    "fit_direction",
    "fit_quartile_split",
    "fit_task_vectors",
    "fit_zscore_stats",
    "get_backend",
    "get_block_module",
    "get_task",
    "get_tasks",
    "list_layer_ids",
    "load_activation_collection",
    "load_direction",
    "quartile_contrast_labels",
    "render_comparison_gif",
    "required_fields_for_feature",
    "run_scale_sweep",
    "save_activation_collection",
    "save_direction",
    "extract_display_field",
    "write_sweep_report",
    "zscore",
]
