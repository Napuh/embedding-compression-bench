import argparse
import time
from typing import Any, Dict

import mteb

from core.configs import ExperimentConfig
from core.engine import EmbeddingEngine
from utils.config_utils import create_experiment_configs, load_config
from utils.experiment_utils import load_existing_results, save_experiment_results


def run_experiments(
    model_name: str,
    tasks: list[str],
    experiment_configs: list[ExperimentConfig],
    batch_size: int = 32,
    output_dir: str = "",
    rerun_existing: bool = False,
    cache_location: str = ":memory:",
) -> Dict[str, Any]:

    if not output_dir:
        output_dir = f"results/{model_name}"

    results = {}
    model = EmbeddingEngine(model_name=model_name, cache_location=cache_location)

    for experiment in experiment_configs:
        # Load existing results if present
        existing_results = load_existing_results(
            output_dir, experiment.name, rerun_existing
        )
        if existing_results:
            results[experiment.name] = existing_results
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

            # Update results after each benchmark
            results[experiment.name] = {
                "scores": ndcg_scores.copy(),
                "time": time.time() - start_time,
            }
            save_experiment_results(
                output_dir, experiment.name, results[experiment.name]
            )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cache-location", type=str, default=":memory:", help="Cache location"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_configs = create_experiment_configs(config["experiments"])

    results = run_experiments(
        model_name=config["model_name"],
        tasks=config["tasks"],
        experiment_configs=experiment_configs,
        cache_location=args.cache_location,
    )
