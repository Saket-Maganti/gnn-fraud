"""Resolve project authorities without usernames or private absolute defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


ENVIRONMENT_KEYS = {
    "repo_root": "COREGRAPH_REPO_ROOT",
    "curated_root": "COREGRAPH_CURATED_ROOT",
    "evidence_cache": "COREGRAPH_EVIDENCE_CACHE",
    "data_root": "COREGRAPH_DATA_ROOT",
    "output_root": "COREGRAPH_OUTPUT_ROOT",
}


class PathResolutionError(RuntimeError):
    """Raised when a configured authority is missing or structurally invalid."""


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    curated_root: Path
    evidence_cache: Path
    data_root: Path
    output_root: Path
    local_config: Path | None

    def as_public_dict(self) -> dict[str, str]:
        """Return portable values where repository-local paths stay relative."""

        def display(path: Path) -> str:
            try:
                return path.relative_to(self.repo_root).as_posix() or "."
            except ValueError:
                return f"${{{_environment_name_for(path, self)}}}"

        return {
            "repo_root": ".",
            "curated_root": display(self.curated_root),
            "evidence_cache": display(self.evidence_cache),
            "data_root": display(self.data_root),
            "output_root": display(self.output_root),
        }


def _environment_name_for(path: Path, paths: ProjectPaths) -> str:
    mapping = {
        paths.repo_root: ENVIRONMENT_KEYS["repo_root"],
        paths.curated_root: ENVIRONMENT_KEYS["curated_root"],
        paths.evidence_cache: ENVIRONMENT_KEYS["evidence_cache"],
        paths.data_root: ENVIRONMENT_KEYS["data_root"],
        paths.output_root: ENVIRONMENT_KEYS["output_root"],
    }
    return mapping.get(path, "CONFIGURED_PATH")


def discover_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until the CoReGraph package and project metadata coexist."""

    candidate = (start or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "coregraph").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise PathResolutionError(
        "cannot discover CoReGraph repository root; set COREGRAPH_REPO_ROOT "
        "or run from inside the checkout"
    )


def _read_local_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PathResolutionError(f"cannot read local path config {path}: {exc}") from exc
    values = payload.get("paths", {})
    if not isinstance(values, dict) or any(not isinstance(v, str) for v in values.values()):
        raise PathResolutionError(f"{path} must contain a [paths] table of strings")
    unknown = sorted(set(values) - set(ENVIRONMENT_KEYS))
    if unknown:
        raise PathResolutionError(f"unknown path keys in {path}: {unknown}")
    return values


def _resolve_value(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_paths(
    *,
    overrides: Mapping[str, str | Path] | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
    local_config: Path | None = None,
    validate: bool = False,
) -> ProjectPaths:
    """Resolve CLI overrides, environment, local TOML, then portable defaults.

    ``overrides`` models explicit CLI arguments and has highest precedence.
    The untracked TOML layer is deliberately optional. Validation never creates
    directories and reports every missing authority in one error.
    """

    overrides = dict(overrides or {})
    environ = environ or os.environ
    unknown = sorted(set(overrides) - set(ENVIRONMENT_KEYS))
    if unknown:
        raise PathResolutionError(f"unknown path override keys: {unknown}")

    repo_value = overrides.get("repo_root") or environ.get(ENVIRONMENT_KEYS["repo_root"])
    repo_root = (
        _resolve_value(repo_value, Path.cwd())
        if repo_value is not None
        else discover_repo_root(start)
    )
    config_path = (local_config or repo_root / "config" / "local_paths.toml").resolve()
    configured = _read_local_config(config_path)
    defaults: dict[str, str | Path] = {
        "repo_root": repo_root,
        "curated_root": repo_root.parent / "gnn-fraud-github-curated",
        "evidence_cache": repo_root.parent / "gnn-fraud-local-evidence-cache",
        "data_root": repo_root / "data",
        "output_root": repo_root / "results",
    }

    resolved: dict[str, Path] = {"repo_root": repo_root}
    for key in ("curated_root", "evidence_cache", "data_root", "output_root"):
        value = (
            overrides.get(key)
            or environ.get(ENVIRONMENT_KEYS[key])
            or configured.get(key)
            or defaults[key]
        )
        resolved[key] = _resolve_value(value, repo_root)

    paths = ProjectPaths(
        repo_root=resolved["repo_root"],
        curated_root=resolved["curated_root"],
        evidence_cache=resolved["evidence_cache"],
        data_root=resolved["data_root"],
        output_root=resolved["output_root"],
        local_config=config_path if config_path.exists() else None,
    )
    if validate:
        required = {
            "repo_root": paths.repo_root,
            "curated_root": paths.curated_root,
            "evidence_cache": paths.evidence_cache,
            "data_root": paths.data_root,
            "output_root": paths.output_root,
        }
        missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
        if missing:
            hints = ", ".join(
                f"{name}: {ENVIRONMENT_KEYS[name]}" for name in sorted(required)
            )
            raise PathResolutionError(
                "configured authorities do not exist: " + "; ".join(missing) + ". "
                "Override with " + hints
            )
        if not (paths.repo_root / "coregraph").is_dir():
            raise PathResolutionError(f"repo_root is not a CoReGraph checkout: {paths.repo_root}")
        if not (paths.curated_root / ".git").exists():
            raise PathResolutionError(
                f"curated_root is not an independent Git checkout: {paths.curated_root}"
            )
    return paths
