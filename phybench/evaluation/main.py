from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Final

from tabulate import tabulate

from .evaluation_config import (
    EvaluationConfig,
    get_log_file_path,
    load_evaluation_config,
)
from .expression_distance import EED

__all__: Final[list[str]] = [
    "main",
    "main_cli",
    "load_evaluation_config",
    "EvaluationConfig",
    "get_file_path_with_normalization",
    "normalize_json_filename",
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
    print(f"Log directory created/verified: {log_path.parent}")

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


def get_file_path_with_normalization(folder: str, filename: str) -> str:
    """
    Constructs the full file path from folder and filename, normalizing the JSON extension.

    Args:
        folder: The directory containing files
        filename: The filename (with or without .json extension)

    Returns:
        The full path to the file with .json extension ensured
    """
    normalized_filename = normalize_json_filename(filename)
    return str(Path(folder) / normalized_filename)


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

    print(f"Successfully Built Worklist, total_length:{len(work_list)}")

    # scoring
    # 根据final_answer中的模型回答和approved_problems中的答案进行评分,修改final_answer文件

    cpu_cores = multiprocessing.cpu_count()
    print(f"We have {cpu_cores} cores for grading...")
    t0 = time.time()
    # cpu_cores=1
    results = []

    with multiprocessing.Pool(processes=cpu_cores) as pool:
        # 提交任务并获取结果（按顺序）
        results = list(pool.map(process_single_problem, work_list))

    t1 = time.time()
    print(f"Evaluation Finished, total time: {t1 - t0:.2f}s")

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
    print(f"Output directory created/verified: {output_path.parent}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(approved_problems, f, ensure_ascii=False, indent=4)

    output_table = []
    for model in model_scores:
        model_scores[model] = model_scores[model] / model_nums[model]
        output_table.append([model, model_scores[model]])

    s_opt = tabulate(
        output_table, headers=["Model", "Score"], tablefmt="fancy_grid", floatfmt=".2f"
    )
    print(s_opt)
    # print(model_scores)

    print("Complete!")

    return s_opt


def parse_args(config: EvaluationConfig) -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model answers against ground truth using EED scoring"
    )

    # Build default file paths from folders + filenames, with cross-module placeholder support
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

    parser.add_argument(
        "--gt-file",
        default=default_gt_path,
        help="Path to ground truth JSON file (contains reference problems with correct answers)",
    )
    parser.add_argument(
        "--model-answers-file",
        default=default_model_answers_path,
        help="Path to model answers JSON file (contains generated solutions from models)",
    )
    parser.add_argument(
        "--output-dir",
        default=config.output_dir,
        help="Output directory (where the final grading results will be saved)",
    )
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
    parser.add_argument(
        "--log-file", default=get_log_file_path(config), help="Log file path"
    )

    return parser.parse_args()


def main_cli() -> None:
    """Command line interface entry point."""
    config = load_evaluation_config()
    args = parse_args(config)

    if not args.gt_file:
        print(
            "Error: No ground truth file specified. Use --gt-file or set gt_file in config."
        )
        return

    if not args.model_answers_file:
        print(
            "Error: No model answers file specified. Use --model-answers-file or set model_answers_file in config."
        )
        return

    if not args.output_dir:
        print(
            "Error: No output directory specified. Use --output-dir or set output_dir in config."
        )
        return

    # Generate output file path using template
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    gt_filename = Path(args.gt_file).name
    model_answers_filename = Path(args.model_answers_file).name
    output_template = config.output_file or "{gt_file}_{model_answers_file}_evaluation"

    output_file = get_evaluation_output_file(
        output_dir_path,
        gt_filename,
        model_answers_filename,
        output_template,
        api_caller_model=config.api_caller_model or "",
        api_caller_input_file=config.api_caller_input_file or "",
        api_caller_output_file=config.api_caller_output_file or "",
    )

    scoring_params = [args.initial_score, args.scoring_slope]

    print("🎯 Starting evaluation process:")
    print(f"  - Ground truth file: {args.gt_file}")
    print(f"  - Model answers file: {args.model_answers_file}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Output file: {output_file}")
    print(f"  - Scoring parameters: {scoring_params}")
    print(
        f"  - Processes: {args.num_processes if args.num_processes > 0 else 'auto-detect'}"
    )
    print(f"  - Log file: {args.log_file}")

    # Normalize file paths to ensure .json extension
    # If user provided just filename, use configured folder; otherwise use provided path
    gt_path = Path(args.gt_file)
    if gt_path.parent == Path(".") and config.gt_dir:
        normalized_gt_file = get_file_path_with_normalization(
            config.gt_dir, gt_path.name
        )
    else:
        normalized_gt_file = str(gt_path.parent / normalize_json_filename(gt_path.name))

    model_path = Path(args.model_answers_file)
    if model_path.parent == Path(".") and config.model_answers_dir:
        normalized_model_file = get_file_path_with_normalization(
            config.model_answers_dir, model_path.name
        )
    else:
        normalized_model_file = str(
            model_path.parent / normalize_json_filename(model_path.name)
        )

    result_table = main(
        normalized_gt_file,
        normalized_model_file,
        str(output_file),
        scoring_params,
        args.log_file,
    )
    print(result_table)


if __name__ == "__main__":
    main_cli()
