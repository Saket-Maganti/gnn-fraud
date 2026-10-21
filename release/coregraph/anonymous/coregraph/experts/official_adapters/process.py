"""Subprocess boundary for official repositories without vendoring them."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from coregraph.experts.base import OfficialStatus


@dataclass(frozen=True)
class OfficialProcessAdapter:
    name: str
    repository: str
    commit: str
    licence: str
    checkout_env: str
    entrypoint: str
    status: OfficialStatus

    def validate_checkout(self, checkout: str | Path) -> list[str]:
        root = Path(checkout)
        errors: list[str] = []
        if not root.is_dir():
            return ["checkout_missing"]
        if not (root / self.entrypoint).is_file():
            errors.append("entrypoint_missing")
        git = root / ".git"
        if git.exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or result.stdout.strip() != self.commit:
                errors.append("commit_mismatch")
        else:
            errors.append("git_metadata_missing")
        return errors

    def smoke_command(
        self,
        *,
        input_manifest: str,
        output_path: str,
    ) -> list[str]:
        return [
            "python",
            self.entrypoint,
            "--coregraph-smoke",
            "--input-manifest",
            input_manifest,
            "--output",
            output_path,
        ]

    def run_smoke(
        self,
        checkout: str | Path,
        *,
        input_manifest: str,
        output_path: str,
        timeout_seconds: int = 120,
    ) -> dict[str, object]:
        errors = self.validate_checkout(checkout)
        if errors:
            return {"status": self.status.value, "errors": errors}
        result = subprocess.run(
            self.smoke_command(
                input_manifest=input_manifest,
                output_path=output_path,
            ),
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "status": "SMOKE_PASS" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
