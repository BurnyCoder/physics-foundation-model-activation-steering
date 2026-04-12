from gphyt.steering.tasks import (
    canonicalize_dataset_name,
    dataset_supports_fields,
    get_task,
)


def test_canonicalize_dataset_name():
    assert canonicalize_dataset_name("Euler_Multi_Quadrants_PeriodicBC") == (
        "euler_multi_quadrants_periodicbc"
    )
    assert canonicalize_dataset_name("turbulent-radiative-layer-2D") == (
        "turbulent_radiative_layer_2d"
    )


def test_dataset_supports_required_fields():
    assert dataset_supports_fields(
        "rayleigh_benard",
        ("pressure", "density", "velocity_x", "velocity_y"),
    )
    assert not dataset_supports_fields("shear_flow", ("density",))


def test_get_task_for_feature_and_regime():
    feature_task = get_task("mean_pressure")
    regime_task = get_task("regime_shear_vs_euler")

    assert feature_task.task_type == "feature"
    assert "pressure" in feature_task.required_fields
    assert regime_task.task_type == "regime"
    assert regime_task.positive_datasets == ("shear_flow",)
