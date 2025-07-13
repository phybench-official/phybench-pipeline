from __future__ import annotations

from pathlib import Path


class PathResolver:
    """
    A centralized class to resolve all file paths and placeholders.
    """

    def __init__(
        self,
        model_name: str,
        api_caller_input_dir: str,
        api_caller_input_file: str,
        api_caller_output_dir: str,
        api_caller_output_file_template: str,
        evaluation_gt_dir: str,
        evaluation_gt_file_template: str,
        evaluation_model_answers_dir: str,
        evaluation_model_answers_file_template: str,
        evaluation_output_dir: str,
        evaluation_output_file_template: str,
        log_dir: str,
        log_file_template: str,
    ):
        self.model_name = model_name
        self.sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
        self.api_caller_input_file_base = Path(api_caller_input_file).stem

        self.api_caller_input_dir = api_caller_input_dir
        self.api_caller_output_dir = api_caller_output_dir
        self.api_caller_output_file_template = api_caller_output_file_template
        self.evaluation_gt_dir = evaluation_gt_dir
        self.evaluation_gt_file_template = evaluation_gt_file_template
        self.evaluation_model_answers_dir = evaluation_model_answers_dir
        self.evaluation_model_answers_file_template = (
            evaluation_model_answers_file_template
        )
        self.evaluation_output_dir = evaluation_output_dir
        self.evaluation_output_file_template = evaluation_output_file_template
        self.log_dir = log_dir
        self.log_file_template = log_file_template

        self.api_caller_output_file_base = self.api_caller_output_file_template.replace(
            "{input_file}", self.api_caller_input_file_base
        ).replace("{model}", self.sanitized_model_name)

    def _normalize_filename(self, filename: str, extension: str) -> str:
        """Ensures the filename ends with the specified extension."""
        if not filename.endswith(extension):
            filename += extension
        return filename

    def _resolve_template(self, template: str) -> str:
        """Resolves all known placeholders in a given template string."""
        gt_file_base = Path(self.evaluation_gt_file_template).stem
        return (
            template.replace("{api_caller_input_file}", self.api_caller_input_file_base)
            .replace("{api_caller_model}", self.sanitized_model_name)
            .replace("{api_caller_output_file}", self.api_caller_output_file_base)
            .replace("{gt_file}", gt_file_base)
        )

    def get_api_caller_input_file(self) -> Path:
        input_dir = Path(self.api_caller_input_dir)
        input_filename = self._normalize_filename(
            self.api_caller_input_file_base, ".json"
        )
        return input_dir / input_filename

    def get_api_caller_output_file(self) -> Path:
        output_dir = Path(self.api_caller_output_dir)
        output_filename = self._normalize_filename(
            self.api_caller_output_file_base, ".json"
        )
        return output_dir / output_filename

    def get_evaluation_gt_file(self) -> Path:
        gt_dir = Path(self.evaluation_gt_dir)
        gt_filename = self._resolve_template(self.evaluation_gt_file_template)
        gt_filename = self._normalize_filename(gt_filename, ".json")
        return gt_dir / gt_filename

    def get_evaluation_model_answers_file(self) -> Path:
        model_answers_dir = Path(self.evaluation_model_answers_dir)
        model_answers_filename = self._resolve_template(
            self.evaluation_model_answers_file_template
        )
        model_answers_filename = self._normalize_filename(
            model_answers_filename, ".json"
        )
        return model_answers_dir / model_answers_filename

    def get_evaluation_output_file(self) -> Path:
        output_dir = Path(self.evaluation_output_dir)
        output_filename = self._resolve_template(self.evaluation_output_file_template)
        output_filename = self._normalize_filename(output_filename, ".json")
        return output_dir / output_filename

    def get_log_file(self) -> Path:
        log_dir = Path(self.log_dir)
        log_filename = self._resolve_template(self.log_file_template)
        log_filename = self._normalize_filename(log_filename, ".log")
        return log_dir / log_filename
