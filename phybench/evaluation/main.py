from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Final

from tabulate import tabulate

from phybench.logging_config import get_logger, setup_logging

from .evaluation_config import (
    EvaluationConfig,
    get_log_file_path,
    load_evaluation_config,
)
from .expression_distance import EED

logger = get_logger(__name__)

__all__: Final[list[str]] = [
    "main",
    "main_cli",
    "load_evaluation_config",
    "EvaluationConfig",
    "get_file_path_with_normalization",
    "normalize_json_filename",
    "expand_template_placeholders",
    "expand_log_template_placeholders",
    "get_evaluation_output_file",
    "resolve_file_path",
    "resolve_log_file_path",
    "resolve_output_file_path",
]

progress = 0


def write_log(s: str, file: str = "./logs.txt") -> None:
    with open(file, "a") as f:
        f.write(s + "\n")


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


def get_file_path_with_normalization(dir: str, filename: str) -> str:
    """
    Constructs the full file path from directory and filename, normalizing the JSON extension.

    Args:
        dir: The directory containing files
        filename: The filename (with or without .json extension)

    Returns:
        The full path to the file with .json extension ensured
    """
    normalized_filename = normalize_json_filename(filename)
    return str(Path(dir) / normalized_filename)


def get_evaluation_output_file(
    output_dir_path: Path,
    gt_filename: str,
    model_answers_filename: str,
    output_template: str,
    api_caller_model: str = "",
    api_caller_input_file: str = "",
    api_caller_output_file: str = "",
) -> Path:
    """
    Generates the full path for the evaluation output JSON file using a template.

    Args:
        output_dir_path: The directory where the output file will be saved.
        gt_filename: The ground truth filename (with or without .json extension).
        model_answers_filename: The model answers filename (with or without .json extension).
        output_template: The output filename template with various placeholders.
        api_caller_model: The model name from API caller for cross-module placeholders.
        api_caller_input_file: The input filename from API caller for cross-module placeholders.
        api_caller_output_file: The output file template from API caller for cross-module placeholders.

    Returns:
        The absolute path to the output JSON file.
    """
    # Extract filename without extension (handle both with/without .json)
    gt_base = Path(gt_filename).stem
    model_answers_base = Path(model_answers_filename).stem

    # Handle cross-module placeholders
    api_caller_model_sanitized = api_caller_model.replace("/", "_").replace(":", "_")
    api_caller_input_base = (
        Path(api_caller_input_file).stem if api_caller_input_file else ""
    )

    # Compute the API caller output filename
    api_caller_output_computed = ""
    if api_caller_output_file and api_caller_model and api_caller_input_file:
        api_caller_output_computed = api_caller_output_file.replace(
            "{input_file}", api_caller_input_base
        ).replace("{model}", api_caller_model_sanitized)
        # Remove .json extension for base name
        api_caller_output_computed = Path(api_caller_output_computed).stem

    # Replace placeholders in the template
    output_filename = (
        output_template.replace("{gt_file}", gt_base)
        .replace("{model_answers_file}", model_answers_base)
        .replace("{api_caller_model}", api_caller_model_sanitized)
        .replace("{api_caller_input_file}", api_caller_input_base)
        .replace("{api_caller_output_file}", api_caller_output_computed)
    )

    # Ensure .json extension
    if not output_filename.endswith(".json"):
        output_filename += ".json"

    return output_dir_path / output_filename


def expand_template_placeholders(
    template: str,
    api_caller_model: str,
    api_caller_input_file: str,
    api_caller_output_file: str = "",
) -> str:
    """
    Expands cross-module template placeholders in a string.

    Args:
        template: The template string with placeholders.
        api_caller_model: The model name from API caller.
        api_caller_input_file: The input filename from API caller.
        api_caller_output_file: The output file template from API caller.

    Returns:
        The template with placeholders replaced.
    """
    # Sanitize model name
    api_caller_model_sanitized = api_caller_model.replace("/", "_").replace(":", "_")
    api_caller_input_base = (
        Path(api_caller_input_file).stem if api_caller_input_file else ""
    )

    # Compute the API caller output filename
    api_caller_output_computed = ""
    if api_caller_output_file and api_caller_model and api_caller_input_file:
        api_caller_output_computed = api_caller_output_file.replace(
            "{input_file}", api_caller_input_base
        ).replace("{model}", api_caller_model_sanitized)
        # Remove .json extension for base name
        api_caller_output_computed = Path(api_caller_output_computed).stem

    # Replace cross-module placeholders
    expanded = (
        template.replace("{api_caller_model}", api_caller_model_sanitized)
        .replace("{api_caller_input_file}", api_caller_input_base)
        .replace("{api_caller_output_file}", api_caller_output_computed)
    )

    # Ensure .json extension if needed
    if template != expanded and not expanded.endswith(".json"):
        expanded += ".json"

    return expanded


def expand_log_template_placeholders(
    template: str,
    api_caller_model: str,
    api_caller_input_file: str,
    api_caller_output_file: str = "",
) -> str:
    """
    Expands cross-module template placeholders in a log filename string.
    Unlike expand_template_placeholders, this doesn't add .json extension.

    Args:
        template: The template string with placeholders.
        api_caller_model: The model name from API caller.
        api_caller_input_file: The input filename from API caller.
        api_caller_output_file: The output file template from API caller.

    Returns:
        The template with placeholders replaced (no automatic .json extension).
    """
    # Sanitize model name
    api_caller_model_sanitized = api_caller_model.replace("/", "_").replace(":", "_")
    api_caller_input_base = (
        Path(api_caller_input_file).stem if api_caller_input_file else ""
    )

    # Compute the API caller output filename
    api_caller_output_computed = ""
    if api_caller_output_file and api_caller_model and api_caller_input_file:
        api_caller_output_computed = api_caller_output_file.replace(
            "{input_file}", api_caller_input_base
        ).replace("{model}", api_caller_model_sanitized)
        # Remove .json extension for base name
        api_caller_output_computed = Path(api_caller_output_computed).stem

    # Replace cross-module placeholders
    expanded = (
        template.replace("{api_caller_model}", api_caller_model_sanitized)
        .replace("{api_caller_input_file}", api_caller_input_base)
        .replace("{api_caller_output_file}", api_caller_output_computed)
    )

    return expanded


def process_single_problem(data: dict[str, Any]) -> list[Any]:
    model_name = data["model"]
    ai_ans = data["model_answer"]
    right_ans = data["right_answer"]
    problem_id = data["id"]
    log_file_path = data.get("log_file_path", "logging.txt")

    scoring_pars = data["scoring_pars"]

    t0 = time.time()

    score, rel_distance, treesize, distance_num = EED(
        right_ans, ai_ans, debug_mode=False, scoring_parameters=scoring_pars
    )
    t1 = time.time()

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"Finished evaluating {model_name}. Problem id: {data['id']}, Time: {t1 - t0:.2f}s\n"
        )

    return [model_name, score, problem_id, rel_distance, treesize, distance_num]


def main(
    gt_file_dir: str,
    gen_file_dir: str,
    output_file: str,
    parameters: list[int] | None,
    log_file: str = "logging.txt",
) -> str:
    if not parameters:
        raise ValueError("Scoring parameters must be provided and cannot be empty")

    # Initialize logging with configurable path
    initialize_logging(log_file)

    final_answer_f = gen_file_dir
    approved_problems_f = gt_file_dir

    with open(final_answer_f, encoding="utf-8") as f:
        final_answer = json.load(f)

    with open(approved_problems_f, encoding="utf-8") as f:
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
            # print(id_number,model)
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
                        "scoring_pars": parameters,
                        "log_file_path": log_file,
                    }
                )

    logger.info(f"Successfully built worklist, total length: {len(work_list)}")

    # scoring
    # 根据final_answer中的模型回答和approved_problems中的答案进行评分,修改final_answer文件

    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"Available CPU cores for evaluation: {cpu_cores}")
    t0 = time.time()
    # cpu_cores=1
    results = []

    with multiprocessing.Pool(processes=cpu_cores) as pool:
        # 提交任务并获取结果（按顺序）
        results = list(pool.map(process_single_problem, work_list))

    t1 = time.time()
    logger.info(f"Evaluation finished, total time: {t1 - t0:.2f}s")

    # plot
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
        # print(model,result[1],problem_id)

    for data in approved_problems:
        score_list = data["model_score"]
        if score_list:
            avg_score = sum(score_list) / len(score_list)
            score_2 = 0
            for score in score_list:
                score_2 += (score - avg_score) ** 2
            data["model_score_var"] = score_2 / len(score_list)
        # print(data)

    # 存储data为json
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


def parse_args(config: EvaluationConfig) -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model answers against ground truth using EED scoring"
    )

    # Build default file paths from directories + filenames, with cross-module placeholder support
    default_gt_path = None
    if config.gt_dir and config.gt_file:
        gt_file_expanded = expand_template_placeholders(
            config.gt_file,
            config.api_caller_model or "",
            config.api_caller_input_file or "",
            config.api_caller_output_file or "",
        )
        default_gt_path = get_file_path_with_normalization(
            config.gt_dir, gt_file_expanded
        )

    default_model_answers_path = None
    if config.model_answers_dir and config.model_answers_file:
        model_answers_file_expanded = expand_template_placeholders(
            config.model_answers_file,
            config.api_caller_model or "",
            config.api_caller_input_file or "",
            config.api_caller_output_file or "",
        )
        default_model_answers_path = get_file_path_with_normalization(
            config.model_answers_dir, model_answers_file_expanded
        )

    # Ground truth arguments (can specify either full path or dir+file)
    gt_group = parser.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--gt-file",
        default=default_gt_path,
        help="Full path to ground truth JSON file (contains reference problems with correct answers)",
    )
    gt_group.add_argument(
        "--gt-dir",
        default=config.gt_dir,
        help="Directory containing ground truth files (use with --gt-filename)",
    )
    parser.add_argument(
        "--gt-filename",
        default=config.gt_file,
        help="Ground truth filename (use with --gt-dir, supports placeholders)",
    )

    # Model answers arguments (can specify either full path or dir+file)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-answers-file",
        default=default_model_answers_path,
        help="Full path to model answers JSON file (contains generated solutions from models)",
    )
    model_group.add_argument(
        "--model-answers-dir",
        default=config.model_answers_dir,
        help="Directory containing model answer files (use with --model-answers-filename)",
    )
    parser.add_argument(
        "--model-answers-filename",
        default=config.model_answers_file,
        help="Model answers filename (use with --model-answers-dir, supports placeholders)",
    )

    # Output arguments (can specify either full path or dir+file)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-file",
        help="Full path to output JSON file (where evaluation results will be saved)",
    )
    output_group.add_argument(
        "--output-dir",
        default=config.output_dir,
        help="Output directory (use with --output-filename)",
    )
    parser.add_argument(
        "--output-filename",
        default=config.output_file,
        help="Output filename template (use with --output-dir, supports placeholders)",
    )

    # Log arguments (can specify either full path or dir+file)
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-file",
        default=get_log_file_path(config),
        help="Full path to log file",
    )
    log_group.add_argument(
        "--log-dir",
        default=config.log_dir,
        help="Log directory (use with --log-filename)",
    )
    parser.add_argument(
        "--log-filename",
        default=config.log_file,
        help="Log filename template (use with --log-dir, supports placeholders)",
    )

    # Scoring parameters
    parser.add_argument(
        "--initial-score",
        type=int,
        default=config.initial_score,
        help="Base score assigned before distance penalty (higher = more lenient, range: 0-100)",
    )
    parser.add_argument(
        "--scoring-slope",
        type=int,
        default=config.scoring_slope,
        help="Scaling factor for expression distance penalty (higher = steeper penalty curve)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=config.num_processes,
        help="Number of processes to use (0 = auto-detect)",
    )

    return parser.parse_args()


def resolve_file_path(
    full_path: str | None,
    directory: str | None,
    filename: str | None,
    config: EvaluationConfig,
    fallback_filename: str = "default.json",
) -> str:
    """
    Resolve file path from either full path or directory + filename.

    Args:
        full_path: Full file path if provided
        directory: Directory path if provided
        filename: Filename if provided (supports placeholders)
        config: Configuration object for placeholder expansion
        fallback_filename: Fallback filename if none provided

    Returns:
        Resolved full file path
    """
    if full_path:
        return str(Path(full_path))

    if directory and filename:
        expanded_filename = expand_template_placeholders(
            filename,
            config.api_caller_model or "",
            config.api_caller_input_file or "",
            config.api_caller_output_file or "",
        )
        return get_file_path_with_normalization(directory, expanded_filename)

    if directory:
        return get_file_path_with_normalization(directory, fallback_filename)

    return fallback_filename


def resolve_log_file_path(
    full_path: str | None,
    directory: str | None,
    filename: str | None,
    config: EvaluationConfig,
) -> str:
    """
    Resolve log file path from either full path or directory + filename.

    Args:
        full_path: Full log file path if provided
        directory: Log directory if provided
        filename: Log filename if provided (supports placeholders)
        config: Configuration object for placeholder expansion

    Returns:
        Resolved full log file path
    """
    if full_path:
        return str(Path(full_path))

    if directory and filename:
        expanded_filename = expand_log_template_placeholders(
            filename,
            config.api_caller_model or "",
            config.api_caller_input_file or "",
            config.api_caller_output_file or "",
        )
        # Ensure .log extension for log files
        if not expanded_filename.endswith(".log"):
            expanded_filename += ".log"
        return str(Path(directory) / expanded_filename)

    return get_log_file_path(config)  # fallback to config default


def resolve_output_file_path(
    full_path: str | None,
    directory: str | None,
    filename_template: str | None,
    gt_filename: str,
    model_answers_filename: str,
    config: EvaluationConfig,
) -> Path:
    """
    Resolve output file path from either full path or directory + template.

    Args:
        full_path: Full output file path if provided
        directory: Output directory if provided
        filename_template: Output filename template if provided
        gt_filename: Ground truth filename for template expansion
        model_answers_filename: Model answers filename for template expansion
        config: Configuration object for placeholder expansion

    Returns:
        Resolved full output file path as Path object
    """
    if full_path:
        return Path(full_path)

    if directory:
        output_dir = Path(directory)
        template = filename_template or config.output_file or "evaluation_results"

        return get_evaluation_output_file(
            output_dir,
            gt_filename,
            model_answers_filename,
            template,
            api_caller_model=config.api_caller_model or "",
            api_caller_input_file=config.api_caller_input_file or "",
            api_caller_output_file=config.api_caller_output_file or "",
        )

    # Fallback to current directory
    return Path("evaluation_results.json")


def main_cli() -> None:
    """Command line interface entry point."""
    config = load_evaluation_config()
    args = parse_args(config)

    # Resolve log file path and setup logging early
    log_file_path = resolve_log_file_path(
        args.log_file,
        args.log_dir,
        args.log_filename,
        config,
    )

    setup_logging(
        log_file=log_file_path,
        log_level="DEBUG" if hasattr(args, "debug") and args.debug else "INFO",
        console_level="INFO",
    )

    logger = get_logger(__name__)

    # Resolve file paths consistently
    gt_file_path = resolve_file_path(
        args.gt_file,
        args.gt_dir,
        args.gt_filename,
        config,
        "ground_truth.json",
    )

    model_answers_file_path = resolve_file_path(
        args.model_answers_file,
        args.model_answers_dir,
        args.model_answers_filename,
        config,
        "model_answers.json",
    )

    output_file_path = resolve_output_file_path(
        args.output_file,
        args.output_dir,
        args.output_filename,
        Path(gt_file_path).name,
        Path(model_answers_file_path).name,
        config,
    )

    log_file_path = resolve_log_file_path(
        args.log_file,
        args.log_dir,
        args.log_filename,
        config,
    )

    # Validate required inputs
    if not Path(gt_file_path).exists():
        logger.error(f"Ground truth file not found: {gt_file_path}")
        logger.error("Use --gt-file or --gt-dir with --gt-filename")
        return

    if not Path(model_answers_file_path).exists():
        logger.error(f"Model answers file not found: {model_answers_file_path}")
        logger.error(
            "Use --model-answers-file or --model-answers-dir with --model-answers-filename"
        )
        return

    # Create output directory if needed
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create log directory if needed
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    scoring_params = [args.initial_score, args.scoring_slope]

    logger.info("🎯 Starting evaluation process:")
    logger.info(f"  - Ground truth file: {gt_file_path}")
    logger.info(f"  - Model answers file: {model_answers_file_path}")
    logger.info(f"  - Output file: {output_file_path}")
    logger.info(f"  - Log file: {log_file_path}")
    logger.info(f"  - Scoring parameters: {scoring_params}")
    logger.info(
        f"  - Processes: {args.num_processes if args.num_processes > 0 else 'auto-detect'}"
    )

    result_table = main(
        gt_file_path,
        model_answers_file_path,
        str(output_file_path),
        scoring_params,
        log_file_path,
    )
    logger.info("Evaluation results:")
    logger.info(f"\n{result_table}")


if __name__ == "__main__":
    main_cli()
