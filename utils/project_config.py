"""Project paths supplied through environment variables.

Shell entry points load ``config.sh`` before invoking Python. Group-level
scripts run interactively can use ``source config.sh`` first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Source HCP/2025_release_analysis/config.sh first."
        )
    return Path(value).expanduser()


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the manuscript analyses."""

    work_root: Path
    hippunfold_root: Path
    output_root: Path
    resource_root: Path
    subject_list: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        return cls(
            work_root=_path("HIPPAMYG_WORK_ROOT"),
            hippunfold_root=_path("HIPPUNFOLD_ROOT"),
            output_root=_path("HIPPAMYG_OUTPUT_ROOT"),
            resource_root=_path("HIPPAMYG_RESOURCE_ROOT"),
            subject_list=_path("HIPPAMYG_SUBJECT_LIST"),
        )
