from pathlib import Path

from jinja2 import Template

from backend.utils.common_funcs import read_file
from configs.enums import PromptTypeEnum


class PromptEngine:
    """Prompt engineering helper class."""

    def __init__(self, prompt_dir: Path = "prompts"):
        self.prompt_dir = prompt_dir

    def render(self, prompt_type: PromptTypeEnum, version: str, **kwargs) -> tuple[str, dict, dict]:
        """Performs prompt rendering.

        Args:
            prompt_type: The prompt type = feature to render (e.g. "laboratory_test_processor").
            version: The version of the prompt (e.g. "v1").
            **kwargs: Additional keyword arguments to pass to the template.

        Returns:
            tuple[str, dict, dict]: The formatted prompt string, system settings and model settings.
        """
        data = read_file(self.prompt_dir / str(prompt_type.value) / f"{version}.yaml")

        template = Template(data.get("user", ""))
        prompt_template = template.render(**kwargs)

        model_settings = data.get("_meta", {}).get("model_settings", {})
        system_prompt = data.get("system", {})

        return prompt_template, system_prompt, model_settings
