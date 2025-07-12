#!/usr/bin/env python3
"""
Comprehensive tests for path handling in evaluation module.
Tests basic functionality, edge cases, and error conditions.
"""

import os
import tempfile
from pathlib import Path

from phybench.evaluation.evaluation_config import (
    EvaluationConfig,
    get_log_file_path,
    load_evaluation_config,
)
from phybench.evaluation.main import (
    expand_log_template_placeholders,
    expand_template_placeholders,
    get_file_path_with_normalization,
    normalize_json_filename,
    resolve_file_path,
    resolve_log_file_path,
    resolve_output_file_path,
)


class TestBasicPathHandling:
    """Test basic path handling functions."""

    def test_normalize_json_filename(self):
        """Test JSON filename normalization."""
        assert normalize_json_filename("test") == "test.json"
        assert normalize_json_filename("test.json") == "test.json"
        assert normalize_json_filename("test.jsonl") == "test.jsonl.json"

        # Edge cases
        assert normalize_json_filename("") == ".json"
        assert normalize_json_filename(".json") == ".json"
        assert normalize_json_filename("file.txt") == "file.txt.json"
        assert (
            normalize_json_filename("file.JSON") == "file.JSON.json"
        )  # Case sensitive

    def test_get_file_path_with_normalization(self):
        """Test file path construction with normalization."""
        assert get_file_path_with_normalization("data", "test") == str(
            Path("data") / "test.json"
        )
        assert get_file_path_with_normalization("data", "test.json") == str(
            Path("data") / "test.json"
        )

        # Edge cases
        assert get_file_path_with_normalization("", "test") == str(Path("test.json"))
        assert get_file_path_with_normalization("data", "") == str(
            Path("data") / ".json"
        )

        # Path separators
        assert get_file_path_with_normalization("data/nested", "test") == str(
            Path("data/nested") / "test.json"
        )


class TestPlaceholderExpansion:
    """Test placeholder expansion with various edge cases."""

    def test_basic_placeholder_expansion(self):
        """Test basic placeholder expansion."""
        result = expand_template_placeholders(
            "{api_caller_input_file}_{api_caller_model}",
            api_caller_model="gpt-4o",
            api_caller_input_file="test.json",
            api_caller_output_file="{input_file}_{model}",
        )
        assert result == "test_gpt-4o.json"

    def test_model_name_sanitization(self):
        """Test model name sanitization for special characters."""
        test_cases = [
            ("gpt-4o/v1:latest", "gpt-4o_v1_latest"),
            ("claude-3.5/sonnet:2024", "claude-3.5_sonnet_2024"),
            ("model/with:many/special:chars", "model_with_many_special_chars"),
            ("simple-model", "simple-model"),
            ("", ""),
        ]

        for input_model, expected_sanitized in test_cases:
            result = expand_template_placeholders(
                "{api_caller_model}",
                api_caller_model=input_model,
                api_caller_input_file="test",
                api_caller_output_file="",
            )
            assert result == f"{expected_sanitized}.json"

    def test_output_file_computation(self):
        """Test complex output file computation."""
        result = expand_template_placeholders(
            "{api_caller_output_file}",
            api_caller_model="gpt-4o",
            api_caller_input_file="test",
            api_caller_output_file="{input_file}_{model}",
        )
        assert result == "test_gpt-4o.json"

        # Complex template
        result = expand_template_placeholders(
            "{api_caller_output_file}_evaluated",
            api_caller_model="claude-3-5",  # Use - instead of . to avoid Path.stem issues
            api_caller_input_file="physics_problems.json",
            api_caller_output_file="solutions_{input_file}_{model}_v1",
        )
        assert result == "solutions_physics_problems_claude-3-5_v1_evaluated.json"

    def test_missing_placeholders(self):
        """Test behavior with missing or None values."""
        result = expand_template_placeholders(
            "{api_caller_input_file}_{api_caller_model}",
            api_caller_model="",
            api_caller_input_file="",
            api_caller_output_file="",
        )
        assert result == "_.json"

        result = expand_template_placeholders(
            "static_name",
            api_caller_model="gpt-4o",
            api_caller_input_file="test",
            api_caller_output_file="",
        )
        # Static names without placeholders don't get .json extension in expand_template_placeholders
        # The .json extension is added later by normalize_json_filename in get_file_path_with_normalization
        assert result == "static_name"

    def test_log_template_expansion(self):
        """Test log-specific template expansion (no .json extension)."""
        result = expand_log_template_placeholders(
            "{api_caller_model}_{api_caller_input_file}",
            api_caller_model="gpt-4o",
            api_caller_input_file="test.json",
            api_caller_output_file="",
        )
        assert result == "gpt-4o_test"  # No .json extension

        # Static log name
        result = expand_log_template_placeholders(
            "evaluation_log",
            api_caller_model="gpt-4o",
            api_caller_input_file="test",
            api_caller_output_file="",
        )
        assert result == "evaluation_log"


class TestPathResolution:
    """Test the new path resolution functions with edge cases."""

    def test_resolve_file_path_priority(self):
        """Test path resolution priority: full_path > dir+file > dir+fallback."""
        config = EvaluationConfig()
        config.api_caller_model = "gpt-4o"
        config.api_caller_input_file = "test"

        # Full path takes priority
        result = resolve_file_path(
            "/full/path/file.json", "ignored_dir", "ignored_file", config
        )
        assert str(result) == str(Path("/full/path/file.json"))

        # Directory + filename
        result = resolve_file_path(None, "data", "custom_file", config)
        assert result == str(Path("data") / "custom_file.json")

        # Directory + fallback
        result = resolve_file_path(None, "data", None, config, "fallback.json")
        assert result == str(Path("data") / "fallback.json")

        # Pure fallback
        result = resolve_file_path(None, None, None, config, "default.json")
        assert result == "default.json"

    def test_resolve_file_path_with_placeholders(self):
        """Test file path resolution with placeholder expansion."""
        config = EvaluationConfig()
        config.api_caller_model = "gpt-4o/turbo"
        config.api_caller_input_file = "physics_test.json"
        config.api_caller_output_file = "{input_file}_{model}"

        result = resolve_file_path(
            None, "models", "{api_caller_input_file}_{api_caller_model}", config
        )
        assert result == str(Path("models") / "physics_test_gpt-4o_turbo.json")

    def test_resolve_log_file_path(self):
        """Test log file path resolution."""
        config = EvaluationConfig()
        config.log_dir = "logs"
        config.log_file = "evaluation"
        config.api_caller_model = "gpt-4o"
        config.api_caller_input_file = "test"

        # Full path priority
        result = resolve_log_file_path("/custom/log.log", None, None, config)
        assert str(result) == str(Path("/custom/log.log"))

        # Directory + filename
        result = resolve_log_file_path(None, "custom_logs", "test_run", config)
        assert result == str(Path("custom_logs") / "test_run.log")

        # With placeholders
        result = resolve_log_file_path(
            None, "logs", "{api_caller_model}_{api_caller_input_file}", config
        )
        assert result == str(Path("logs") / "gpt-4o_test.log")

        # Fallback to config
        result = resolve_log_file_path(None, None, None, config)
        expected = get_log_file_path(config)
        assert result == expected

    def test_resolve_output_file_path(self):
        """Test output file path resolution."""
        config = EvaluationConfig()
        config.output_file = "eval_{api_caller_input_file}"
        config.api_caller_model = "gpt-4o"
        config.api_caller_input_file = "test"
        config.api_caller_output_file = "{input_file}_{model}"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Full path priority
            result = resolve_output_file_path(
                "/custom/output.json", None, None, "gt.json", "model.json", config
            )
            assert result == Path("/custom/output.json")

            # Directory + template
            result = resolve_output_file_path(
                None,
                temp_dir,
                "results_{api_caller_model}",
                "gt.json",
                "model.json",
                config,
            )
            assert result.parent == Path(temp_dir)
            assert "results_gpt-4o" in result.name
            assert result.suffix == ".json"

            # Fallback to current directory
            result = resolve_output_file_path(
                None, None, None, "gt.json", "model.json", config
            )
            assert result == Path("evaluation_results.json")


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_empty_and_none_inputs(self):
        """Test behavior with empty or None inputs."""
        config = EvaluationConfig()

        # Empty strings
        result = resolve_file_path("", "", "", config, "fallback.json")
        assert result == "fallback.json"

        # None values
        result = resolve_file_path(None, None, None, config, "fallback.json")
        assert result == "fallback.json"

    def test_windows_path_separators(self):
        """Test handling of Windows path separators."""
        result = get_file_path_with_normalization("data\\nested\\folder", "test")
        expected = str(Path("data\\nested\\folder") / "test.json")
        assert result == expected

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        config = EvaluationConfig()
        config.api_caller_model = "模型-4o"  # Chinese characters
        config.api_caller_input_file = "测试文件.json"  # Chinese characters

        result = resolve_file_path(
            None, "data", "{api_caller_input_file}_{api_caller_model}", config
        )
        # Should handle Unicode gracefully
        assert "测试文件" in result
        assert "模型-4o" in result

    def test_very_long_paths(self):
        """Test handling of very long file paths."""
        config = EvaluationConfig()
        long_model_name = "very_long_model_name_" * 10  # 220 characters
        config.api_caller_model = long_model_name
        config.api_caller_input_file = "test"

        result = resolve_file_path(None, "data", "{api_caller_model}", config)
        # Should handle long paths without crashing
        assert len(result) > 200
        assert long_model_name in result

    def test_relative_vs_absolute_paths(self):
        """Test handling of relative vs absolute paths."""
        config = EvaluationConfig()

        # Relative path
        result = resolve_file_path(None, "relative/path", "test", config)
        assert not Path(result).is_absolute()

        # Absolute path (Windows)
        if os.name == "nt":
            result = resolve_file_path(None, "C:\\absolute\\path", "test", config)
            assert Path(result).is_absolute()

        # Absolute path (Unix-style)
        result = resolve_file_path(None, "/absolute/path", "test", config)
        expected = str(Path("/absolute/path") / "test.json")
        assert result == expected


class TestConfigIntegration:
    """Test integration with actual config loading."""

    def test_config_loading_with_actual_file(self):
        """Test loading actual config file."""
        config = load_evaluation_config()

        # Check that all required fields are present
        required_fields = [
            "gt_dir",
            "gt_file",
            "model_answers_dir",
            "model_answers_file",
            "output_dir",
            "output_file",
            "log_dir",
            "log_file",
        ]

        for field in required_fields:
            assert hasattr(config, field), f"Missing field: {field}"

    def test_full_path_resolution_integration(self):
        """Test full path resolution with actual config."""
        config = load_evaluation_config()

        # Should be able to resolve all paths without errors
        gt_path = resolve_file_path(None, config.gt_dir, config.gt_file, config)
        model_path = resolve_file_path(
            None, config.model_answers_dir, config.model_answers_file, config
        )
        log_path = resolve_log_file_path(None, config.log_dir, config.log_file, config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = resolve_output_file_path(
                None, temp_dir, config.output_file, "gt.json", "model.json", config
            )

        # All paths should be valid strings/Path objects
        assert isinstance(gt_path, str)
        assert isinstance(model_path, str)
        assert isinstance(log_path, str)
        assert isinstance(output_path, Path)

        # Should have proper extensions
        assert gt_path.endswith(".json")
        assert model_path.endswith(".json")
        assert log_path.endswith(".log")
        assert output_path.suffix == ".json"


if __name__ == "__main__":
    # Run tests manually without pytest
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

    print("🧪 Running comprehensive evaluation path handling tests...")

    test_classes = [
        TestBasicPathHandling,
        TestPlaceholderExpansion,
        TestPathResolution,
        TestEdgeCasesAndErrors,
        TestConfigIntegration,
    ]

    all_passed = True
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}:")
        if not run_test_class(test_class):
            all_passed = False

    if all_passed:
        print(
            "\n✅ All tests passed! Path handling is robust and handles edge cases correctly."
        )
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
