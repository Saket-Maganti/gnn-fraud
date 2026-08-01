from __future__ import annotations

from pathlib import Path

import pytest

from coregraph.io.path_resolution import PathResolutionError, discover_repo_root, resolve_paths


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "coregraph").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "results").mkdir()
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    return root


def test_discover_and_portable_defaults(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    curated = root.parent / "gnn-fraud-github-curated"
    (curated / ".git").mkdir(parents=True)
    cache = root.parent / "gnn-fraud-local-evidence-cache"
    cache.mkdir()
    assert discover_repo_root(root / "coregraph") == root
    paths = resolve_paths(start=root / "coregraph", environ={})
    assert paths.repo_root == root
    assert paths.curated_root == curated
    assert paths.evidence_cache == cache
    assert paths.as_public_dict()["repo_root"] == "."


def test_precedence_cli_environment_then_local_config(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    config = root / "config" / "local_paths.toml"
    config.parent.mkdir()
    config.write_text("[paths]\nevidence_cache='from-config'\noutput_root='config-output'\n", encoding="utf-8")
    paths = resolve_paths(
        start=root,
        local_config=config,
        environ={"COREGRAPH_EVIDENCE_CACHE": str(root / "from-env")},
        overrides={"output_root": root / "from-cli"},
    )
    assert paths.evidence_cache == root / "from-env"
    assert paths.output_root == root / "from-cli"


def test_validation_reports_every_missing_authority(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(PathResolutionError, match="configured authorities do not exist") as error:
        resolve_paths(start=root, environ={}, validate=True)
    assert "curated_root" in str(error.value)
    assert "evidence_cache" in str(error.value)


def test_bad_config_and_unknown_override_fail_closed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    config = root / "paths.toml"
    config.write_text("[paths]\nsecret='x'\n", encoding="utf-8")
    with pytest.raises(PathResolutionError, match="unknown path keys"):
        resolve_paths(start=root, local_config=config, environ={})
    with pytest.raises(PathResolutionError, match="unknown path override"):
        resolve_paths(start=root, environ={}, overrides={"secret": "x"})
