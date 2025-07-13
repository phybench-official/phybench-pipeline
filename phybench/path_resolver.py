from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phybench.settings import AppSettings

class PathResolver:
    """
    A centralized class to resolve all file paths and placeholders.
    """

    def __init__(self, settings: AppSettings, model_name: str, input_file: str):
        self.settings = settings
        self.model_name = model_name
        self.sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
        self.input_file_name = Path(input_file).name
        self.input_file_base = Path(input_file).stem

        # Determine the output file from the api_caller to be used in evaluation placeholders
        self.api_caller_output_file_base = (
            self.settings.api_caller.paths.output_file.replace(
                "{input_file}", self.input_file_base
            ).replace("{model}", self.sanitized_model_name)
        )

    def _resolve_template(self, template: str) -> str:
        """Resolves all known placeholders in a given template string."""
        return (
            template.replace("{api_caller_input_file}", self.input_file_base)
            .replace("{api_caller_model}", self.sanitized_model_name)
            .replace("{api_caller_output_file}", self.api_caller_output_file_base)
        )

    def get_api_caller_input_file(self) -> Path:
        return Path(self.settings.api_caller.paths.input_dir) / self.input_file_name

    def get_api_caller_output_file(self) -> Path:
        output_dir = Path(self.settings.api_caller.paths.output_dir)
        output_filename = self.api_caller_output_file_base
        if not output_filename.endswith(".json"):
            output_filename += ".json"
        return output_dir / output_filename

    def get_evaluation_gt_file(self) -> Path:
        gt_dir = Path(self.settings.evaluation.paths.gt_dir)
        gt_filename = self._resolve_template(self.settings.evaluation.paths.gt_file)
        if not gt_filename.endswith(".json"):
            gt_filename += ".json"
        return gt_dir / gt_filename

    def get_evaluation_model_answers_file(self) -> Path:
        model_answers_dir = Path(self.settings.evaluation.paths.model_answers_dir)
        model_answers_filename = self._resolve_template(
            self.settings.evaluation.paths.model_answers_file
        )
        if not model_answers_filename.endswith(".json"):
            model_answers_filename += ".json"
        return model_answers_dir / model_answers_filename

    def get_evaluation_output_file(self) -> Path:
        output_dir = Path(self.settings.evaluation.paths.output_dir)
        output_filename = self._resolve_template(self.settings.evaluation.paths.output_file)
        if not output_filename.endswith(".json"):
            output_filename += ".json"
        return output_dir / output_filename

    def get_log_file(self) -> Path:
        log_dir = Path(self.settings.logging.log_dir)
        log_filename = self._resolve_template(self.settings.logging.log_file)
        if not log_filename.endswith(".log"):
            log_filename += ".log"
        return log_dir / log_filename