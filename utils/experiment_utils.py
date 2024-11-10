import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_existing_results(
    output_dir: str, experiment_name: str, rerun_existing: bool = False
) -> Optional[Dict[str, Any]]:
    """Load existing results if present and rerun not requested."""
    result_path = Path(output_dir) / f"results_{experiment_name}.json"
    if result_path.exists() and not rerun_existing:
        with open(result_path) as f:
            return json.load(f)
    return None


def save_experiment_results(
    output_dir: str, experiment_name: str, results: Dict[str, Any]
) -> None:
    """Save experiment results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/results_{experiment_name}.json", "w") as f:
        json.dump(results, f, indent=2)
