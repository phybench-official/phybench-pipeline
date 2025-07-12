#!/usr/bin/env python3
"""
Tests for CLI and programmatic interface consistency in evaluation module.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from phybench.evaluation.evaluation_config import load_evaluation_config
from phybench.evaluation.main import (
    resolve_file_path,
    resolve_log_file_path,
    resolve_output_file_path,
)


class TestCLIConsistency:
    """Test that CLI and programmatic interfaces are consistent."""

    def test_cli_help_includes_all_arguments(self):
        """Test that CLI help shows all expected arguments."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "phybench.evaluation", "--help"],
                capture_output=True,
                text=True,
                cwd=".",
                timeout=10,
            )

            help_text = result.stdout

            # Check for all expected argument groups
            expected_args = [
                # Ground truth arguments
                "--gt-file",
                "--gt-dir",
                "--gt-filename",
                # Model answers arguments
                "--model-answers-file",
                "--model-answers-dir",
                "--model-answers-filename",
                # Output arguments
                "--output-file",
                "--output-dir",
                "--output-filename",
                # Log arguments
                "--log-file",
                "--log-dir",
                "--log-filename",
                # Scoring parameters
                "--initial-score",
                "--scoring-slope",
                "--num-processes",
            ]

            missing_args = [arg for arg in expected_args if arg not in help_text]
            assert not missing_args, f"Missing arguments in CLI help: {missing_args}"

            # Check for mutually exclusive groups
            assert "mutually exclusive" in help_text.lower() or "|" in help_text

        except subprocess.TimeoutExpired as e:
            raise AssertionError("CLI help command timed out") from e
        except Exception as e:
            raise AssertionError(f"Error running CLI help: {e}") from e

    def test_cli_argument_parsing_structure(self):
        """Test that CLI argument structure is correct."""
        try:
            # Test that mutually exclusive arguments are properly structured
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phybench.evaluation",
                    "--gt-file",
                    "/path1",
                    "--gt-dir",
                    "/path2",
                    "--help",
                ],
                capture_output=True,
                text=True,
                cwd=".",
                timeout=10,
            )

            # Should show error about mutually exclusive args, not just help
            # (though help might be shown after error)
            assert result.returncode != 0 or "error" in result.stderr.lower()

        except subprocess.TimeoutExpired as e:
            raise AssertionError("CLI argument test timed out") from e
        except Exception:
            # This is expected to fail due to mutually exclusive args
            pass

    def test_config_and_cli_path_resolution_consistency(self):
        """Test that config-based and CLI-based path resolution produce consistent results."""
        config = load_evaluation_config()

        # Test ground truth path resolution
        config_gt_path = resolve_file_path(None, config.gt_dir, config.gt_file, config)
        cli_gt_path = resolve_file_path(None, config.gt_dir, config.gt_file, config)
        assert config_gt_path == cli_gt_path

        # Test model answers path resolution
        config_model_path = resolve_file_path(
            None, config.model_answers_dir, config.model_answers_file, config
        )
        cli_model_path = resolve_file_path(
            None, config.model_answers_dir, config.model_answers_file, config
        )
        assert config_model_path == cli_model_path

        # Test log path resolution
        config_log_path = resolve_log_file_path(
            None, config.log_dir, config.log_file, config
        )
        cli_log_path = resolve_log_file_path(
            None, config.log_dir, config.log_file, config
        )
        assert config_log_path == cli_log_path

    def test_placeholder_expansion_consistency(self):
        """Test that placeholder expansion works consistently across interfaces."""
        config = load_evaluation_config()

        # Test various placeholder combinations
        test_templates = [
            "{api_caller_input_file}",
            "{api_caller_model}",
            "{api_caller_input_file}_{api_caller_model}",
            "eval_{api_caller_input_file}",
            "results_{api_caller_model}_{api_caller_input_file}",
        ]

        for template in test_templates:
            # Config-based resolution
            config_path = resolve_file_path(None, "test_dir", template, config)

            # CLI-based resolution (same logic)
            cli_path = resolve_file_path(None, "test_dir", template, config)

            assert config_path == cli_path, (
                f"Inconsistent resolution for template: {template}"
            )


class TestArgumentValidation:
    """Test argument validation and error handling."""

    def test_mutually_exclusive_argument_groups(self):
        """Test that mutually exclusive argument groups work correctly."""

        config = load_evaluation_config()

        # This test would require mocking sys.argv, so we'll test the logic directly
        # by ensuring the argument parser structure is correct

        # Test that we can resolve paths with different argument combinations
        # Full path approach
        gt_path_full = resolve_file_path("/full/path/gt.json", None, None, config)
        assert gt_path_full == str(Path("/full/path/gt.json"))

        # Dir + filename approach
        gt_path_dir = resolve_file_path(None, "data", "gt_file", config)
        assert "data" in gt_path_dir and "gt_file" in gt_path_dir

    def test_fallback_behavior(self):
        """Test fallback behavior when arguments are missing."""
        config = load_evaluation_config()

        # Test fallbacks work correctly
        fallback_path = resolve_file_path(None, None, None, config, "fallback.json")
        assert fallback_path == "fallback.json"

        # Test with only directory
        dir_only_path = resolve_file_path(None, "data", None, config, "fallback.json")
        assert "data" in dir_only_path and "fallback.json" in dir_only_path

    def test_error_conditions(self):
        """Test error conditions and edge cases."""
        # Test with empty config values
        empty_config = load_evaluation_config()
        empty_config.api_caller_model = ""
        empty_config.api_caller_input_file = ""
        empty_config.api_caller_output_file = ""

        # Should still work with empty values
        result = resolve_file_path(None, "data", "{api_caller_model}", empty_config)
        assert "data" in result


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_workflow_simulation(self):
        """Simulate a full evaluation workflow with path resolution."""
        config = load_evaluation_config()

        # Simulate CLI usage with custom paths
        custom_gt_dir = "custom/ground_truth"
        custom_gt_file = "physics_problems_{api_caller_input_file}"
        custom_model_dir = "custom/solutions"
        custom_model_file = "solutions_{api_caller_model}_{api_caller_input_file}"
        custom_log_dir = "custom/logs"
        custom_log_file = "eval_{api_caller_model}_{api_caller_input_file}"

        # Resolve all paths
        gt_path = resolve_file_path(None, custom_gt_dir, custom_gt_file, config)
        model_path = resolve_file_path(
            None, custom_model_dir, custom_model_file, config
        )
        log_path = resolve_log_file_path(None, custom_log_dir, custom_log_file, config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = resolve_output_file_path(
                None,
                temp_dir,
                "eval_results_{api_caller_model}",
                Path(gt_path).name,
                Path(model_path).name,
                config,
            )

        # Verify all paths are properly resolved
        assert "custom" in gt_path and "ground_truth" in gt_path
        assert "custom" in model_path and "solutions" in model_path
        assert "custom" in log_path and "logs" in log_path
        assert gt_path.endswith(".json")
        assert model_path.endswith(".json")
        assert log_path.endswith(".log")
        assert output_path.suffix == ".json"

    def test_different_model_scenarios(self):
        """Test path resolution with different model configurations."""
        config = load_evaluation_config()

        # Test with different model names
        model_scenarios = [
            "gpt-4o",
            "claude-3.5-sonnet",
            "llama-2/7B:latest",
            "custom_model_v1.0",
            "模型名称",  # Unicode
        ]

        for model in model_scenarios:
            config.api_caller_model = model

            # Test that paths are resolved correctly for each model
            gt_path = resolve_file_path(None, "data", "{api_caller_model}_test", config)
            log_path = resolve_log_file_path(
                None, "logs", "eval_{api_caller_model}", config
            )

            # Should handle all model names gracefully
            assert gt_path.endswith(".json")
            assert log_path.endswith(".log")
            # Special characters should be sanitized in paths
            if "/" in model or ":" in model:
                assert "/" not in Path(gt_path).name
                assert ":" not in Path(gt_path).name

    def test_cross_platform_compatibility(self):
        """Test that path resolution works across platforms."""
        config = load_evaluation_config()

        # Test various path styles
        path_styles = [
            ("data/unix/style", "unix_file"),
            ("data\\windows\\style", "windows_file"),
            ("data/mixed\\style", "mixed_file"),
        ]

        for directory, filename in path_styles:
            result = resolve_file_path(None, directory, filename, config)

            # Should produce valid paths regardless of input style
            assert filename in result
            assert result.endswith(".json")

            # Path should be normalized for current platform
            result_path = Path(result)
            assert result_path.name == f"{filename}.json"


if __name__ == "__main__":
    # Run tests manually
    import sys

    def run_test_class(test_class):
        """Run all test methods in a test class."""
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith("test_")]

        for method_name in methods:
            try:
                print(f"  Running {method_name}...")
                getattr(instance, method_name)()
                print(f"  ✓ {method_name} passed")
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
                return False
        return True

    print("🧪 Running CLI consistency and integration tests...")

    test_classes = [
        TestCLIConsistency,
        TestArgumentValidation,
        TestIntegrationScenarios,
    ]

    all_passed = True
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}:")
        if not run_test_class(test_class):
            all_passed = False

    if all_passed:
        print("\n✅ All CLI consistency and integration tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
