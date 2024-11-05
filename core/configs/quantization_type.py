from dataclasses import dataclass
from enum import Enum


class QuantizationType(Enum):
    FLOAT32 = "float32"  # float32
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    FLOAT8_E4M3 = "float8_e4m3"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1 = "float4_e2m1"
    BINARY = "binary"


@dataclass
class QuantizationConfig:
    quant_type: QuantizationType
