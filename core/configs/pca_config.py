from dataclasses import dataclass


@dataclass
class PCAConfig:
    n_components: int | float | None
    random_state: int = 42
