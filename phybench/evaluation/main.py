from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Final

from tabulate import tabulate

from phybench.config_loader import get_settings
from phybench.logging_config import get_logger, setup_logging

from .expression_distance import EED

logger = get_logger(__name__)

__all__: Final[list[str]] = [
    "evaluate",
    "main",
]

progress = 0


processing_lis: list[Any] = []
processed_list: list[Any] = []


def initialize_logging(log_file_path: str) -> None:
    """Initialize logging with configurable file path."""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("")


def normalize_json_filename(filename: str) -> str:
    """
    Normalizes a filename to ensure it has a .json extension.

    Args:
        filename: The input filename (with or without .json extension)

    Returns:
        The filename with .json extension ensured
    """
    if not filename.endswith(".json"):
        return f"{filename}.json"
    return filename


def process_single_problem(data: dict[str, Any]) -> list[Any]:
    logger = get_logger(__name__)
    model_name = data["model"]
    ai_ans = data["model_answer"]
    right_ans = data["right_answer"]
    problem_id = data["id"]

    scoring_pars = data["scoring_pars"]

    t0 = time.time()

    score, rel_distance, treesize, distance_num = EED(
        right_ans, ai_ans, scoring_parameters=scoring_pars, debug_mode=False
    )
    t1 = time.time()

    logger.info(
        f"Finished evaluating {model_name}. Problem id: {data['id']}, Time: {t1 - t0:.2f}s\n"
    )

    return [model_name, score, problem_id, rel_distance, treesize, distance_num]


def evaluate(
    gt_file: str,
    gen_file: str,
    output_file: str,
    scoring_parameters: list[int],
    log_file: str = "evaluation.log",
) -> str:
    if not scoring_parameters:
        raise ValueError("Scoring parameters must be provided and cannot be empty")

    # Initialize logging with configurable path
    initialize_logging(log_file)

    with open(gen_file, encoding="utf-8") as f:
        final_answer = json.load(f)

    with open(gt_file, encoding="utf-8") as f:
        approved_problems = json.load(f)

    approved_problems_dict = {}
    for data in approved_problems:
        data["model_name"] = []
        data["model_score"] = []
        data["model_answer"] = []
        data["model_distance"] = []
        data["model_score_var"] = 0
        data["answer_size"] = 0
        approved_problems_dict[data["id"]] = data

    model_list = []
    final_answer_dict = {}
    for data in final_answer:
        name = data["model"]
        if name not in model_list:
            model_list.append(name)

        final_answer_dict[(data["id"], data["model"])] = data

    work_list = []

    for answers in approved_problems[0:]:
        if answers["id"] == 108:
            continue
        for model in model_list:
            id_number = answers["id"]
            query_answer = (id_number, model)
            if query_answer in final_answer_dict:
                model_answer = final_answer_dict[(id_number, model)]["model_answer"]
                right_answer = approved_problems_dict[id_number]["answer"]

                work_list.append(
                    {
                        "id": id_number,
                        "model": model,
                        "model_answer": model_answer,
                        "right_answer": right_answer,
                        "scoring_pars": scoring_parameters,
                        "log_file_path": log_file,
                    }
                )

    logger.info(f"Successfully built worklist, total length: {len(work_list)}")

    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"Available CPU cores for evaluation: {cpu_cores}")
    t0 = time.time()
    results = []

    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = list(pool.map(process_single_problem, work_list))

    t1 = time.time()
    logger.info(f"Evaluation finished, total time: {t1 - t0:.2f}s")

    model_scores: dict[str, float] = {}
    model_nums: dict[str, int] = {}
    for name in model_list:
        model_scores[name] = 0.0
        model_nums[name] = 0

    dist_data = []
    for result in results:
        model = result[0]
        score_i = result[1]
        problem_id = result[2]
        rel_dist = result[3]
        tree_size = result[4]
        distance_number = result[5]

        model_scores[model] += score_i
        model_nums[model] += 1

        approved_problems_dict[problem_id]["answer_size"] = max(
            tree_size, approved_problems_dict[problem_id]["answer_size"]
        )
        approved_problems_dict[problem_id]["model_distance"].append(distance_number)
        approved_problems_dict[problem_id]["model_name"].append(model)
        approved_problems_dict[problem_id]["model_score"].append(score_i)
        approved_problems_dict[problem_id]["model_answer"].append(
            final_answer_dict[(problem_id, model)]["model_answer"]
        )

        dist_data.append(rel_dist)

    for data in approved_problems:
        score_list = data["model_score"]
        if score_list:
            avg_score = sum(score_list) / len(score_list)
            score_2 = 0
            for score in score_list:
                score_2 += (score - avg_score) ** 2
            data["model_score_var"] = score_2 / len(score_list)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(approved_problems, f, ensure_ascii=False, indent=4)

    output_table = []
    for model in model_scores:
        model_scores[model] = model_scores[model] / model_nums[model]
        output_table.append([model, model_scores[model]])

    s_opt = tabulate(
        output_table, headers=["Model", "Score"], tablefmt="fancy_grid", floatfmt=".2f"
    )
    logger.info("Evaluation results:")
    logger.info(f"\n{s_opt}")

    logger.success("Evaluation complete!")

    return s_opt


def main() -> None:
    """Command line interface entry point."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config-file",
        default="config.toml",
        help="Path to the configuration file (e.g., config.toml)",
    )
    args, remaining_argv = pre_parser.parse_known_args()

    try:
        settings = get_settings(args.config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        if args.config_file == "config.toml":
            print(
                "Please create a 'config.toml' file or use --config-file to specify a path."
            )
        return

    parser = argparse.ArgumentParser(
        description="Evaluate model answers against ground truth using EED scoring",
        parents=[pre_parser],
    )

    parser.add_argument(
        "--gt-dir",
        default=settings.evaluation.paths.gt_dir,
        help="Directory containing ground truth files",
    )
    parser.add_argument(
        "--gt-file",
        default=settings.evaluation.paths.gt_file,
        help="Ground truth filename",
    )
    parser.add_argument(
        "--model-answers-dir",
        default=settings.evaluation.paths.model_answers_dir,
        help="Directory containing model answer files",
    )
    parser.add_argument(
        "--model-answers-file",
        default=settings.evaluation.paths.model_answers_file,
        help="Model answers filename",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.evaluation.paths.output_dir,
        help="Output directory",
    )
    parser.add_argument(
        "--output-file",
        default=settings.evaluation.paths.output_file,
        help="Output filename template",
    )
    parser.add_argument(
        "--log-dir",
        default=settings.logging.log_dir,
        help="Log directory",
    )
    parser.add_argument(
        "--log-file",
        default=settings.logging.log_file,
        help="Log filename template",
    )
    parser.add_argument(
        "--initial-score",
        type=int,
        default=settings.evaluation.scoring.initial_score,
        help="Base score assigned before distance penalty (higher = more lenient, range: 0-100)",
    )
    parser.add_argument(
        "--scoring-slope",
        type=int,
        default=settings.evaluation.scoring.scoring_slope,
        help="Scaling factor for expression distance penalty (higher = steeper penalty curve)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=settings.evaluation.execution.num_processes,
        help="Number of processes to use (0 = auto-detect)",
    )

    final_args = parser.parse_args(remaining_argv)

    log_file_path = Path(final_args.log_dir) / final_args.log_file

    setup_logging(
        log_file=log_file_path,
        log_level=settings.logging.file_level,
        console_level=settings.logging.console_level,
    )

    logger = get_logger(__name__)

    gt_file_path = Path(final_args.gt_dir) / final_args.gt_file
    model_answers_file_path = (
        Path(final_args.model_answers_dir) / final_args.model_answers_file
    )
    output_file_path = Path(final_args.output_dir) / final_args.output_file
    log_file_path = Path(final_args.log_dir) / final_args.log_file

    scoring_params = [final_args.initial_score, final_args.scoring_slope]

    logger.info("🎯 Starting evaluation process:")
    logger.info(f"  - Ground truth file: {gt_file_path}")
    logger.info(f"  - Model answers file: {model_answers_file_path}")
    logger.info(f"  - Output file: {output_file_path}")
    logger.info(f"  - Log file: {log_file_path}")
    logger.info(f"  - Scoring parameters: {scoring_params}")
    logger.info(
        f"  - Processes: {final_args.num_processes if final_args.num_processes > 0 else 'auto-detect'}"
    )

    evaluate(
        str(gt_file_path),
        str(model_answers_file_path),
        str(output_file_path),
        scoring_params,
        str(log_file_path),
    )


if __name__ == "__main__":
    main()
