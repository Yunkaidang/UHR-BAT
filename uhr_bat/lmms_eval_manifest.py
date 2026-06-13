"""lmms-eval model registry entry point for UHR-BAT."""


def manifest():
    from lmms_eval.models.registry_v2 import ModelManifest

    return ModelManifest(
        model_id="uhr_bat",
        simple_class_path="uhr_bat.lmms_eval_model.UHRBAT",
        aliases=("uhr-bat", "uhrbat"),
    )
