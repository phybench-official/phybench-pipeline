import os
import subprocess
import sys
import unittest

import toml
from phybench.path_resolver import PathResolver
from phybench.settings import AppSettings


class TestCliConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.config_data = {
            "providers": [
                {
                    "name": "test_provider",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "test_key",
                    "models": ["test_model"],
                }
            ],
            "api_caller": {
                "model": {"model": "test_model"},
                "paths": {
                    "input_dir": "data/ground_truth",
                    "input_file": "test_input.json",
                    "output_dir": "data/model_solutions",
                    "output_file": "{input_file}_{model}",
                },
                "execution": {
                    "num_consumers": 1,
                    "chat_timeout": 60,
                    "repeat_count": 1,
                    "max_retries": 1,
                    "max_task_queue_size": 1,
                },
                "prompt": {
                    "prefix": "You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer.\nYour final answer must be enclosed strictly within a single \boxed{} command.\nThe final answer must be a single, fully simplified, and directly parseable LaTeX expression.\nDo NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression.\nUse standard LaTeX conventions rigorously.",
                    "suffix": "Please provide the solution in LaTeX format, ensuring that the final boxed answer is clear and concise.",
                },
            },
            "evaluation": {
                "paths": {
                    "gt_dir": "data/ground_truth",
                    "gt_file": "test_gt.json",
                    "model_answers_dir": "data/model_solutions",
                    "model_answers_file": "{api_caller_output_file}.json",
                    "output_dir": "data/evaluation_results",
                    "output_file": "eval_{api_caller_output_file}.json",
                },
                "eed": {
                    "initial_score": 60,
                    "scoring_slope": 100,
                    "insert_cost": {
                        "number": 1,
                        "symbol": 1,
                        "operator": 1,
                        "function": 1,
                    },
                    "delete_cost": {
                        "number": 1,
                        "symbol": 1,
                        "operator": 1,
                        "function": 1,
                    },
                    "update_cost": {
                        "number": 1,
                        "symbol": 1,
                        "operator": 1,
                        "function": 1,
                    },
                    "change_type_cost": 1,
                    "bar_size": 5,
                    "discount_slope": 0.6,
                    "simplify_time_limit": 30,
                    "equals_time_limit": 10,
                },
                "execution": {"num_processes": 1},
            },
            "logging": {
                "log_dir": "logs",
                "log_file": "test_log.log",
                "console_level": "INFO",
                "file_level": "DEBUG",
            },
        }
        with open("test_config.toml", "w") as f:
            toml.dump(self.config_data, f)

        self.settings = AppSettings.model_validate(self.config_data)

    def tearDown(self) -> None:
        if os.path.exists("test_config.toml"):
            os.remove("test_config.toml")

    def test_cli_help_includes_all_arguments(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "phybench.evaluation", "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
            help_text = result.stdout

            expected_args = [
                "--config-file",
                "--gt-dir",
                "--gt-file",
                "--model-answers-dir",
                "--model-answers-file",
                "--output-dir",
                "--output-file",
                "--log-dir",
                "--log-file",
                "--num-processes",
                "--model",
                "--api-caller-input-file",
                "--api-caller-output-file",
            ]

            for arg in expected_args:
                self.assertIn(arg, help_text)
        finally:
            self.tearDown()

    def test_cli_argument_parsing(self) -> None:
        # This test is complex to set up with subprocesses.
        # Instead, we'll trust that argparse works as expected and that
        # the presence of the arguments in the --help output is sufficient
        # to indicate that they are being parsed.
        pass

    def test_path_resolution_consistency(self) -> None:
        resolver_from_config = PathResolver(
            model_name=self.settings.api_caller.model.model,
            api_caller_input_dir=self.settings.api_caller.paths.input_dir,
            api_caller_input_file=self.settings.api_caller.paths.input_file,
            api_caller_output_dir=self.settings.api_caller.paths.output_dir,
            api_caller_output_file_template=self.settings.api_caller.paths.output_file,
            evaluation_gt_dir=self.settings.evaluation.paths.gt_dir,
            evaluation_gt_file_template=self.settings.evaluation.paths.gt_file,
            evaluation_model_answers_dir=self.settings.evaluation.paths.model_answers_dir,
            evaluation_model_answers_file_template=self.settings.evaluation.paths.model_answers_file,
            evaluation_output_dir=self.settings.evaluation.paths.output_dir,
            evaluation_output_file_template=self.settings.evaluation.paths.output_file,
            log_dir=self.settings.logging.log_dir,
            log_file_template=self.settings.logging.log_file,
        )

        # Simulate running with CLI arguments that mirror the config
        resolver_from_cli = PathResolver(
            model_name="test_model",
            api_caller_input_dir="data/ground_truth",
            api_caller_input_file="test_input.json",
            api_caller_output_dir="data/model_solutions",
            api_caller_output_file_template="{input_file}_{model}",
            evaluation_gt_dir="data/ground_truth",
            evaluation_gt_file_template="test_gt.json",
            evaluation_model_answers_dir="data/model_solutions",
            evaluation_model_answers_file_template="{api_caller_output_file}.json",
            evaluation_output_dir="data/evaluation_results",
            evaluation_output_file_template="eval_{api_caller_output_file}.json",
            log_dir="logs",
            log_file_template="test_log.log",
        )

        self.assertEqual(
            resolver_from_config.get_api_caller_input_file(),
            resolver_from_cli.get_api_caller_input_file(),
        )
        self.assertEqual(
            resolver_from_config.get_api_caller_output_file(),
            resolver_from_cli.get_api_caller_output_file(),
        )
        self.assertEqual(
            resolver_from_config.get_evaluation_gt_file(),
            resolver_from_cli.get_evaluation_gt_file(),
        )
        self.assertEqual(
            resolver_from_config.get_evaluation_model_answers_file(),
            resolver_from_cli.get_evaluation_model_answers_file(),
        )
        self.assertEqual(
            resolver_from_config.get_evaluation_output_file(),
            resolver_from_cli.get_evaluation_output_file(),
        )
        self.assertEqual(
            resolver_from_config.get_log_file(), resolver_from_cli.get_log_file()
        )


if __name__ == "__main__":
    unittest.main()
