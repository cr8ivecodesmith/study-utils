"""Quizzer packaged template text."""

from __future__ import annotations

# Import tomllib/tomli at call site to avoid cold-start cost.
_TEMPLATE_TEXT = """\
# Quizzer configuration template.
# Generated via `quizzer config init`.

[ai]
model = "gpt-4o-mini"
api_base = "http://localhost:8080/v1"
use_local = true
provider = "local"
temperature = 0.2
max_tokens = 600

# Storage settings -- unchanged by this spec but read by config loader.
[storage]
out_dir = ".quizzer/<name>"
"""


def get_template_text() -> str:
    """Return the bundled template text."""
    return _TEMPLATE_TEXT


# For direct imports: `from .template import TEMPLATE_TEXT`
TEMPLATE_TEXT = _TEMPLATE_TEXT
