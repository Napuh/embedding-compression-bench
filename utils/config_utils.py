from pathlib import Path
from typing import Any, Dict, Literal

import yaml

from core.configs import ExperimentConfig, PCAConfig, QuantizationType


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_configs(
    experiments_config: list, experiment_type: Literal["qdrant", "mteb"] = "mteb"
) -> list[ExperimentConfig]:
    # Filter for Qdrant-supported quantization types if using qdrant
    supported_types = {
        QuantizationType.FLOAT32,
        QuantizationType.FLOAT16,
        QuantizationType.INT8,
        QuantizationType.BINARY,
    }

    experiment_configs = []
    for exp in experiments_config:
        quant_type = QuantizationType[exp["quantization_type"]]

        # Only add experiments with supported quantization types for qdrant
        if experiment_type == "mteb" or quant_type in supported_types:
            experiment_configs.append(
                ExperimentConfig(
                    exp["name"],
                    quant_type,
                    PCAConfig(exp["pca_config"]),
                    exp.get(
                        "calibration_dataset"
                    ),  # Get calibration dataset if present
                )
            )
    return experiment_configs
