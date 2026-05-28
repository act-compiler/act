"""Template management for backend code generation

Provides a clean interface for loading and using code templates without
mutable global variables.
"""

import os
from typing import Dict


class TemplateLoader:
    """Manages loading and caching of templates for code generation"""

    def __init__(self, template_dir: str):
        """
        Initialize template loader

        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir
        self._cache: Dict[str, str] = {}

    def load(self, template_file: str) -> str:
        """
        Load a template file

        Args:
            template_file: Name of the template file

        Returns:
            Template content as string
        """
        if template_file not in self._cache:
            file_path = os.path.join(self.template_dir, template_file)
            with open(file_path, 'r') as f:
                self._cache[template_file] = f.read()
        return self._cache[template_file]

    def render(self, template_file: str, **kwargs) -> str:
        """
        Load and render a template with the given variables

        Args:
            template_file: Name of the template file
            **kwargs: Template variables

        Returns:
            Rendered template content
        """
        template = self.load(template_file)
        result = template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result


def get_backend_template_loader() -> TemplateLoader:
    """Get a template loader for backend templates"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    return TemplateLoader(template_dir)
