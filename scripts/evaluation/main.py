from __future__ import annotations
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from .expression_distance import EED
import multiprocessing
from tabulate import tabulate
from .evaluation_config import load_evaluation_config, EvaluationConfig

progress = 0


def write_log(s: str, file: str = "./logs.txt") -> None:
    with open(file, "a") as f:
        f.write(s + "\n")


processing_lis: List[Any] = []
processed_list: List[Any] = []

def initialize_logging(log_file_path: str) -> None:
    """Initialize logging with configurable file path."""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Log directory created/verified: {log_path.parent}")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("")


def process_single_problem(data: Dict[str, Any]) -> List[Any]:
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
        f.write(f"Finished evaluating {model_name}. Problem id: {data['id']}, Time: {t1-t0:.2f}s\n")

    return [model_name, score, problem_id, rel_distance, treesize, distance_num]


def main(gt_file_dir: str, gen_file_dir: str, output_file: str, parameters: Optional[List[int]], log_file: str = "logging.txt") -> str:
    if not parameters:
        raise ValueError("Scoring parameters must be provided and cannot be empty")

    # Initialize logging with configurable path
    initialize_logging(log_file)

    final_answer_f = gen_file_dir
    approved_problems_f = gt_file_dir

    with open(final_answer_f, "r", encoding="utf-8") as f:
        final_answer = json.load(f)

    with open(approved_problems_f, "r", encoding="utf-8") as f:
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
        if not name in model_list:
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
                model_answer = final_answer_dict[(id_number, model)]["boxed_answer"]
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
    print(f"Evaluation Finished, total time: {t1-t0:.2f}s")

    # plot
    model_scores, model_nums = {}, {}
    for name in model_list:
        model_scores[name] = 0
        model_nums[name] = 0
    num = len(approved_problems)

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
            final_answer_dict[(problem_id, model)]["boxed_answer"]
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
    parser.add_argument(
        "--gt-file",
        default=config.gt_file,
        help="Path to ground truth JSON file (contains reference problems with correct answers)"
    )
    parser.add_argument(
        "--model-answers-file", 
        default=config.model_answers_file,
        help="Path to model answers JSON file (contains generated solutions from models)"
    )
    parser.add_argument(
        "--output-dir",
        default=config.output_file, 
        help="Output file path (where the final grading results will be saved)"
    )
    parser.add_argument(
        "--initial-score",
        type=int,
        default=config.initial_score,
        help="Base score assigned before distance penalty (higher = more lenient, range: 0-100)"
    )
    parser.add_argument(
        "--scoring-slope",
        type=int,
        default=config.scoring_slope,
        help="Scaling factor for expression distance penalty (higher = steeper penalty curve)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=config.num_processes,
        help="Number of processes to use (0 = auto-detect)"
    )
    parser.add_argument(
        "--log-file",
        default=config.log_file,
        help="Log file path"
    )
    
    return parser.parse_args()


def main_cli() -> None:
    """Command line interface entry point."""
    config = load_evaluation_config()
    args = parse_args(config)
    
    if not args.gt_file:
        print("Error: No ground truth file specified. Use --gt-file or set gt_file in config.")
        return
        
    if not args.model_answers_file:
        print("Error: No model answers file specified. Use --model-answers-file or set model_answers_file in config.")
        return
        
    if not args.output_file:
        print("Error: No output directory specified. Use --output-dir or set output_file in config.")
        return

    scoring_params = [args.initial_score, args.scoring_slope]
    
    print(f"🎯 Starting evaluation process:")
    print(f"  - Ground truth file: {args.gt_file}")
    print(f"  - Model answers file: {args.model_answers_file}")
    print(f"  - Output directory: {args.output_file}")
    print(f"  - Scoring parameters: {scoring_params}")
    print(f"  - Processes: {args.num_processes if args.num_processes > 0 else 'auto-detect'}")
    print(f"  - Log file: {args.log_file}")
    
    result_table = main(args.gt_file, args.model_answers_file, args.output_file, scoring_params, args.log_file)
    print(result_table)
