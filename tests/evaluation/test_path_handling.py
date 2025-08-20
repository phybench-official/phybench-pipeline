import os
import unittest
from pathlib import Path

import toml
from phybench.path_resolver import PathResolver
from phybench.settings import AppSettings


class TestPathResolver(unittest.TestCase):
    def setUp(self) -> None:
        # Create a dummy config.toml for testing
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
                    "prefix": "",
                    "suffix": "",
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

    def get_resolver(self, model_name: str = "test_model") -> PathResolver:
        return PathResolver(
            model_name=model_name,
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

    def test_path_resolution(self) -> None:
        resolver = self.get_resolver()
        # API Caller Paths
        self.assertEqual(
            resolver.get_api_caller_input_file(),
            Path("data/ground_truth/test_input.json"),
        )
        self.assertEqual(
            resolver.get_api_caller_output_file(),
            Path("data/model_solutions/test_input_test_model.json"),
        )

        # Evaluation Paths
        self.assertEqual(
            resolver.get_evaluation_gt_file(), Path("data/ground_truth/test_gt.json")
        )
        self.assertEqual(
            resolver.get_evaluation_model_answers_file(),
            Path("data/model_solutions/test_input_test_model.json"),
        )
        self.assertEqual(
            resolver.get_evaluation_output_file(),
            Path("data/evaluation_results/eval_test_input_test_model.json"),
        )

        # Log Path
        self.assertEqual(resolver.get_log_file(), Path("logs/test_log.log"))

    def test_model_name_sanitization(self) -> None:
        resolver = self.get_resolver(model_name="test/model:v1")
        self.assertEqual(
            resolver.get_api_caller_output_file(),
            Path("data/model_solutions/test_input_test_model_v1.json"),
        )

    def test_complex_placeholders(self) -> None:
        self.settings.evaluation.paths.output_file = (
            "eval_{gt_file}_{model}_{api_caller_output_file}.json"
        )
        resolver = self.get_resolver()
        self.assertEqual(
            resolver.get_evaluation_output_file(),
            Path(
                "data/evaluation_results/eval_test_gt_test_model_test_input_test_model.json"
            ),
        )

    def test_empty_placeholders(self) -> None:
        resolver = self.get_resolver(model_name="")
        self.assertEqual(
            resolver.get_api_caller_output_file(),
            Path("data/model_solutions/test_input_.json"),
        )

    def test_unknown_placeholders(self) -> None:
        self.settings.evaluation.paths.output_file = "eval_{unknown_placeholder}.json"
        resolver = self.get_resolver()
        self.assertEqual(
            resolver.get_evaluation_output_file(),
            Path("data/evaluation_results/eval_{unknown_placeholder}.json"),
        )


if __name__ == "__main__":
    unittest.main()
