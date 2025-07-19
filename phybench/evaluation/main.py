from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Final, TypedDict

from loguru import logger
from tabulate import tabulate

from phybench.config_loader import get_settings
from phybench.logging_config import setup_logging
from phybench.path_resolver import PathResolver
from phybench.settings import EvaluationEEDSettings

from .expression_distance import EED

__all__: Final[list[str]] = [
    "evaluate",
    "main",
]


class WorkItem(TypedDict):
    id: int
    model: str
    model_answer: str
    right_answer: str


progress = 0


processing_lis: list[Any] = []
processed_list: list[Any] = []


def worker_init(log_file: str, file_level: str, console_level: str) -> None:
    """Initialize logging for worker processes."""
    setup_logging(log_file, file_level, console_level)


def process_single_problem(
    data: WorkItem, eed_settings: EvaluationEEDSettings
) -> list[Any]:
    global progress, processing_lis, processed_list
    model_name = data["model"]
    ai_ans = data["model_answer"]
    right_ans = data["right_answer"]
    problem_id = data["id"]

    t0 = time.time()

    score, relative_distance, treesize, distance_num = EED(
        right_ans, ai_ans, eed_settings=eed_settings, debug_mode=False
    )
    t1 = time.time()

    logger.info(
        f"Evaluated {model_name}. Problem id: {data['id']: >3}, Time: {t1 - t0:.2f}s"
    )

    return [model_name, score, problem_id, relative_distance, treesize, distance_num]


def evaluate(
    gt_file: str,
    model_answers_file: str,
    output_file: str,
    eed_settings: EvaluationEEDSettings,
    log_file: str = "logs/evaluation.log",
    file_log_level: str = "DEBUG",
    console_log_level: str = "INFO",
) -> str:
    logger.info("Starting evaluation...")

    with open(model_answers_file, encoding="utf-8") as f:
        model_answers = json.load(f)

    with open(gt_file, encoding="utf-8") as f:
        gt = json.load(f)

    # Here on we will start updating gt and finally output it
    gt_dict = {}
    for data in gt:
        if "model" in data:
            del data["model"]
        data["models"] = []
        data["model_score"] = []
        data["model_answer"] = []
        data["model_distance"] = []
        data["model_score_var"] = 0
        data["answer_size"] = 0
        gt_dict[data["id"]] = data

    model_list = []
    model_answers_dict = {}
    for data in model_answers:
        name = data["model"]
        if name not in model_list:
            model_list.append(name)

        model_answers_dict[(data["id"], data["model"])] = data

    work_list: list[WorkItem] = []

    for answers in gt[:]:
        if answers["id"] == 108:  # wrong problem, skip it
            continue
        for model in model_list:
            id_number = answers["id"]
            query_answer = (id_number, model)
            if query_answer in model_answers_dict:
                model_answer = model_answers_dict[(id_number, model)]["model_answer"]
                right_answer = gt_dict[id_number]["answer"]

                work_list.append(
                    WorkItem(
                        id=id_number,
                        model=model,
                        model_answer=model_answer,
                        right_answer=right_answer,
                    )
                )

    logger.info(f"Successfully built worklist, total length: {len(work_list)}")

    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"Available CPU cores for evaluation: {cpu_cores}")
    t0 = time.time()
    results = []

    with multiprocessing.Pool(
        processes=cpu_cores,
        initializer=worker_init,
        initargs=(log_file, file_log_level, console_log_level),
    ) as pool:
        results = pool.starmap(
            process_single_problem, [(item, eed_settings) for item in work_list]
        )

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

        gt_dict[problem_id]["answer_size"] = max(
            tree_size, gt_dict[problem_id]["answer_size"]
        )
        gt_dict[problem_id]["model_distance"].append(distance_number)
        gt_dict[problem_id]["models"].append(model)
        gt_dict[problem_id]["model_score"].append(score_i)
        gt_dict[problem_id]["model_answer"].append(
            model_answers_dict[(problem_id, model)]["model_answer"]
        )

        dist_data.append(rel_dist)

    for data in gt:
        score_list = data["model_score"]
        if not score_list or len(score_list) <= 1:
            continue  # hide the variance if only one model
        avg_score = sum(score_list) / len(score_list)
        score_2 = 0
        for score in score_list:
            score_2 += (score - avg_score) ** 2
        data["model_score_var"] = score_2 / len(score_list)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=4)

    output_table = []
    for model in model_scores:
        model_scores[model] = model_scores[model] / model_nums[model]
        output_table.append([model, model_scores[model]])

    s_opt = tabulate(
        output_table, headers=["Model", "Score"], tablefmt="fancy_grid", floatfmt=".2f"
    )
    logger.info(f"Evaluation results:\n{s_opt}")

    logger.success("Evaluation complete!")

    return s_opt


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config-file",
        default="config.toml",
        help="Path to the configuration file (e.g., config.toml)",
    )
    args, remaining_argv = pre_parser.parse_known_args()

    try:
        settings = get_settings(args.config_file)
        logger.info(f"Loaded settings from {args.config_file}")
    except FileNotFoundError as e:
        logger.error(f"Missing config file: {e}")
        if args.config_file == "config.toml":
            logger.error(
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
        "--num-processes",
        type=int,
        default=settings.evaluation.execution.num_processes,
        help="Number of processes to use (0 = auto-detect)",
    )
    parser.add_argument(
        "--model",
        default=settings.api_caller.model.model,
        help="Model name to evaluate (just for filename template resolution)",
    )
    parser.add_argument(
        "--api-caller-input-file",
        default=settings.api_caller.paths.input_file,
        help="Input file in api caller (just for filename template resolution)",
    )
    parser.add_argument(
        "--api-caller-output-file",
        default=settings.api_caller.paths.output_file,
        help="Output file for API caller (just for filename template resolution)",
    )

    final_args = parser.parse_args(remaining_argv)

    resolver = PathResolver(
        final_args.model,
        settings.api_caller.paths.input_dir,
        final_args.api_caller_input_file,
        settings.api_caller.paths.output_dir,
        final_args.api_caller_output_file,
        final_args.gt_dir,
        final_args.gt_file,
        final_args.model_answers_dir,
        final_args.model_answers_file,
        final_args.output_dir,
        final_args.output_file,
        final_args.log_dir,
        final_args.log_file,
    )

    setup_logging(
        log_file=resolver.get_log_file(),
        log_level=settings.logging.file_level,
        console_level=settings.logging.console_level,
    )

    gt_file_path = resolver.get_evaluation_gt_file()
    model_answers_file_path = resolver.get_evaluation_model_answers_file()
    output_file_path = resolver.get_evaluation_output_file()
    log_file_path = resolver.get_log_file()

    evaluate(
        str(gt_file_path),
        str(model_answers_file_path),
        str(output_file_path),
        settings.evaluation.eed,
        str(log_file_path),
        settings.logging.file_level,
        settings.logging.console_level,
    )


if __name__ == "__main__":
    main()
