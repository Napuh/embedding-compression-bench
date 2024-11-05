from dataclasses import dataclass

from .pca_config import PCAConfig
from .quantization_type import QuantizationType


@dataclass
class ExperimentConfig:
    name: str
    quantization_type: QuantizationType | None
    pca_config: PCAConfig | None
