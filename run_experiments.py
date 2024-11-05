import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import mteb
import yaml

from core.configs import ExperimentConfig, PCAConfig, QuantizationType
from core.engine import EmbeddingEngine


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_configs(experiments_config: list) -> list[ExperimentConfig]:
    experiment_configs = []
    for exp in experiments_config:
        experiment_configs.append(
            ExperimentConfig(
                exp["name"],
                QuantizationType[exp["quantization_type"]],
                PCAConfig(exp["pca_config"]),
            )
        )
    return experiment_configs


def run_experiments(
    model_name: str,
    tasks: list[str],
    experiment_configs: list[ExperimentConfig],
    batch_size: int = 32,
    output_dir: str = "",
    rerun_existing: bool = False,
) -> Dict[str, Any]:

    if not output_dir:
        output_dir = f"results/{model_name}"

    results = {}

    model = EmbeddingEngine(model_name=model_name)

    for experiment in experiment_configs:

        # Load existing results if present
        result_path = Path(output_dir) / f"results_{experiment.name}.json"
        if result_path.exists() and not rerun_existing:
            with open(result_path) as f:
                results[experiment.name] = json.load(f)
                ndcg_scores = results[experiment.name]["scores"]
                start_time = time.time() - results[experiment.name]["time"]
        else:
            ndcg_scores = {}
            start_time = time.time()

        print(f"\nRunning experiments with {experiment.name}")

        model.set_quant_type(experiment.quantization_type)

        for benchmark in tasks:
            # Skip if benchmark already evaluated
            if (
                any(key.startswith(f"{benchmark}-") for key in ndcg_scores)
                and not rerun_existing
            ):
                print(f"\tSkipping benchmark {benchmark} - already evaluated")
                continue

            print(f"\tRunning benchmark: {benchmark}")

            model.set_benchmark(benchmark)
            model.set_pca_config(experiment.pca_config)

            mteb_tasks = mteb.get_tasks(tasks=[benchmark], languages=["eng"])

            evaluation = mteb.MTEB(tasks=mteb_tasks)

            mteb_eval_result = evaluation.run(
                model,
                output_folder=None,
                encode_kwargs={"batch_size": batch_size},
                verbosity=0,
            )

            # Extract only ndcg_at_10 scores
            for task in mteb_eval_result:
                for test_set in task.scores:
                    ndcg_scores["-".join([task.task_name, test_set])] = task.scores[
                        test_set
                    ][0]["ndcg_at_10"]

            duration = time.time() - start_time

            # Update results after each benchmark
            results[experiment.name] = {
                "scores": ndcg_scores.copy(),
                "time": duration,
            }

            # Save results after each benchmark
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{output_dir}/results_{experiment.name}.json", "w") as f:
                json.dump(results[experiment.name], f, indent=2)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_configs = create_experiment_configs(config["experiments"])

    results = run_experiments(
        model_name=config["model_name"],
        tasks=config["tasks"],
        experiment_configs=experiment_configs,
    )
