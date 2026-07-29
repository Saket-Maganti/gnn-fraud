"""
Kaggle bootstrap for gnn-fraud RUNS GPU notebooks.

The code dataset may be mounted under any Kaggle input name, may contain a zip,
or may already be extracted several directories deep. Discovery is therefore
content-based and recursive. Do not import this module as ``kaggle.*`` inside
Kaggle notebooks; the official Kaggle package can shadow the local folder. Load
it by absolute file path from /kaggle/working/gnn-fraud instead.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

WORK = Path("/kaggle/working/gnn-fraud")
INPUT_ROOT = Path("/kaggle/input")

CORE_CODE_MARKERS = (
    "kaggle/kaggle_bootstrap.py",
    "scripts/runs_harness_common.py",
)
EXPECTED_CODE_MARKERS = (
    "kaggle/kaggle_bootstrap.py",
    "scripts/runs_harness_common.py",
    "scripts/run_validation_clean_gnn.py",
    "runs_expansion/README.md",
)
ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)

COMMAND_LOG: list[dict] = []


@dataclass(frozen=True)
class CodePayload:
    """A discovered code payload inside /kaggle/input."""

    kind: str
    source: Path
    repo_root: Path | None
    present_markers: tuple[str, ...]
    missing_expected_markers: tuple[str, ...]

    def summary(self) -> str:
        root = f", repo_root={self.repo_root}" if self.repo_root is not None else ""
        missing = (
            f", missing optional expected markers={list(self.missing_expected_markers)}"
            if self.missing_expected_markers
            else ""
        )
        return f"{self.kind}: {self.source}{root}, markers={list(self.present_markers)}{missing}"


def _normalize_zip_name(name: str) -> str | None:
    posix_name = name.replace("\\", "/")
    path = PurePosixPath(posix_name)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    return "/".join(parts) if parts else None


def _zip_marker_presence(zpath: Path) -> tuple[dict[str, bool], str | None]:
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [
                normalized
                for normalized in (_normalize_zip_name(info.filename) for info in zf.infolist())
                if normalized
            ]
    except zipfile.BadZipFile as exc:
        return {}, f"bad zip: {exc}"
    except OSError as exc:
        return {}, f"cannot read zip: {exc}"

    presence = {
        marker: any(name == marker or name.endswith("/" + marker) for name in names)
        for marker in EXPECTED_CODE_MARKERS
    }
    return presence, None


def _path_marker_presence(root: Path) -> dict[str, bool]:
    return {marker: (root / marker).is_file() for marker in EXPECTED_CODE_MARKERS}


def _present_markers(presence: dict[str, bool]) -> tuple[str, ...]:
    return tuple(marker for marker, present in presence.items() if present)


def _missing_expected_markers(presence: dict[str, bool]) -> tuple[str, ...]:
    return tuple(marker for marker in EXPECTED_CODE_MARKERS if not presence.get(marker, False))


def _missing_core_markers(presence: dict[str, bool]) -> tuple[str, ...]:
    return tuple(marker for marker in CORE_CODE_MARKERS if not presence.get(marker, False))


def _has_core_markers(root: Path) -> bool:
    presence = _path_marker_presence(root)
    return not _missing_core_markers(presence)


def _repo_root_from_marker(marker_path: Path, marker: str) -> Path:
    marker_depth = len(PurePosixPath(marker).parts)
    return marker_path.parents[marker_depth - 1]


def _candidate_roots(input_root: Path) -> list[Path]:
    roots: dict[str, Path] = {}
    for marker in CORE_CODE_MARKERS:
        try:
            hits = sorted(input_root.rglob(marker))
        except OSError:
            hits = []
        for hit in hits:
            root = _repo_root_from_marker(hit, marker)
            key = str(root.resolve(strict=False))
            roots.setdefault(key, root)
    return sorted(roots.values(), key=lambda path: str(path))


def _input_debug_listing(input_base: Path | None = None, *, max_items: int = 80) -> str:
    base = input_base or INPUT_ROOT
    if not base.is_dir():
        return f"{base} is not mounted"

    lines: list[str] = ["top-level inputs:"]
    top_entries = sorted(base.iterdir(), key=lambda path: path.name)
    if not top_entries:
        return f"{base} is empty"
    for entry in top_entries[:max_items]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"- {entry.relative_to(base)}{suffix}")
    if len(top_entries) > max_items:
        lines.append(f"- ... {len(top_entries) - max_items} more top-level entries")

    zip_hits = sorted(base.rglob("*.zip"))
    lines.append("recursive zip files:")
    if zip_hits:
        for zpath in zip_hits[:max_items]:
            lines.append(f"- {zpath.relative_to(base)}")
        if len(zip_hits) > max_items:
            lines.append(f"- ... {len(zip_hits) - max_items} more zip files")
    else:
        lines.append("- none")

    marker_hits: list[Path] = []
    for marker in CORE_CODE_MARKERS:
        marker_hits.extend(sorted(base.rglob(marker)))
    lines.append("recursive code marker hits:")
    if marker_hits:
        for hit in marker_hits[:max_items]:
            lines.append(f"- {hit.relative_to(base)}")
        if len(marker_hits) > max_items:
            lines.append(f"- ... {len(marker_hits) - max_items} more marker hits")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_rejections(rejections: list[str]) -> str:
    if not rejections:
        return "- no zip files or extracted marker roots were found"
    return "\n".join(f"- {reason}" for reason in rejections)


def find_code_payload(input_root: Path = INPUT_ROOT, *, verbose: bool = True) -> CodePayload:
    """Find the uploaded code payload by recursive content markers, not name."""
    input_root = Path(input_root)
    rejections: list[str] = []

    if verbose:
        print("[bootstrap] /kaggle/input discovery listing:")
        print(_input_debug_listing(input_root))

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Kaggle input root not found: {input_root}\n"
            f"Expected code markers: {', '.join(EXPECTED_CODE_MARKERS)}"
        )

    valid_zips: list[CodePayload] = []
    zip_paths = sorted(input_root.rglob("*.zip"))
    if verbose:
        print("[bootstrap] candidate zips:")
    if not zip_paths and verbose:
        print("  - none")
    for zpath in zip_paths:
        presence, error = _zip_marker_presence(zpath)
        if error is not None:
            reason = f"{zpath}: rejected ({error})"
            rejections.append(reason)
            if verbose:
                print(f"  - {reason}")
            continue
        missing_core = _missing_core_markers(presence)
        present = _present_markers(presence)
        if missing_core:
            reason = f"{zpath}: rejected (missing core markers: {list(missing_core)})"
            rejections.append(reason)
            if verbose:
                print(f"  - {reason}; present={list(present)}")
            continue
        payload = CodePayload(
            kind="zip",
            source=zpath,
            repo_root=None,
            present_markers=present,
            missing_expected_markers=_missing_expected_markers(presence),
        )
        valid_zips.append(payload)
        if verbose:
            print(f"  - accepted {payload.summary()}")

    valid_roots: list[CodePayload] = []
    roots = _candidate_roots(input_root)
    if verbose:
        print("[bootstrap] candidate extracted roots:")
    if not roots and verbose:
        print("  - none")
    for root in roots:
        presence = _path_marker_presence(root)
        missing_core = _missing_core_markers(presence)
        present = _present_markers(presence)
        if missing_core:
            reason = f"{root}: rejected (missing core markers: {list(missing_core)})"
            rejections.append(reason)
            if verbose:
                print(f"  - {reason}; present={list(present)}")
            continue
        payload = CodePayload(
            kind="extracted",
            source=root,
            repo_root=root,
            present_markers=present,
            missing_expected_markers=_missing_expected_markers(presence),
        )
        valid_roots.append(payload)
        if verbose:
            print(f"  - accepted {payload.summary()}")

    selected = valid_zips[0] if valid_zips else (valid_roots[0] if valid_roots else None)
    if selected is not None:
        if verbose:
            print(f"[bootstrap] selected code source: {selected.summary()}")
        return selected

    raise FileNotFoundError(
        "No valid gnn-fraud code payload found under "
        f"{input_root}. Expected at least these core markers: "
        f"{', '.join(CORE_CODE_MARKERS)}. Full bundle markers: "
        f"{', '.join(EXPECTED_CODE_MARKERS)}.\n\nDetected inputs:\n"
        f"{_input_debug_listing(input_root)}\n\nRejected candidates:\n"
        f"{_format_rejections(rejections)}"
    )


def _zip_has_code_markers(zpath: Path) -> bool:
    presence, error = _zip_marker_presence(zpath)
    return error is None and not _missing_core_markers(presence)


def _find_code_input(input_base: Path | None = None) -> Path | None:
    try:
        return find_code_payload(input_base or INPUT_ROOT, verbose=False).source
    except FileNotFoundError:
        return None


def _safe_extract_zip(zpath: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    dst_root = dst.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            normalized = _normalize_zip_name(info.filename)
            if normalized is None:
                raise ValueError(f"Unsafe zip member path in {zpath}: {info.filename!r}")
            if not normalized:
                continue
            target = (dst / normalized).resolve()
            if target != dst_root and dst_root not in target.parents:
                raise ValueError(f"Unsafe zip member path in {zpath}: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _find_repo_root_under(root: Path) -> Path:
    if _has_core_markers(root):
        return root
    for candidate in _candidate_roots(root):
        if _has_core_markers(candidate):
            return candidate
    raise FileNotFoundError(
        f"Extracted payload under {root} does not contain core markers: "
        f"{', '.join(CORE_CODE_MARKERS)}"
    )


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def stage_code_payload(
    input_root: Path = INPUT_ROOT,
    work_dir: Path = WORK,
    *,
    force: bool = False,
) -> Path:
    """Stage a valid code payload into ``work_dir``.

    Supports both preserved zip uploads and Kaggle-extracted nested folders.
    Dataset names are ignored; only marker files are used.
    """
    input_root = Path(input_root)
    work_dir = Path(work_dir)
    if not force and _has_core_markers(work_dir):
        print(f"[bootstrap] code already staged at {work_dir}")
        missing = _missing_expected_markers(_path_marker_presence(work_dir))
        if missing:
            print(f"[bootstrap] staged code is missing optional expected markers: {list(missing)}")
        return work_dir

    payload = find_code_payload(input_root, verbose=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    if payload.kind == "zip":
        extract_dir = work_dir.parent / ".gnn_fraud_code_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[bootstrap] extracting code zip: {payload.source} -> {extract_dir}")
        _safe_extract_zip(payload.source, extract_dir)
        repo_root = _find_repo_root_under(extract_dir)
        print(f"[bootstrap] extracted repo root: {repo_root}")
        _copy_tree(repo_root, work_dir)
        shutil.rmtree(extract_dir)
    elif payload.repo_root is not None:
        print(f"[bootstrap] copying extracted code root: {payload.repo_root} -> {work_dir}")
        _copy_tree(payload.repo_root, work_dir)
    else:
        raise FileNotFoundError(f"Invalid code payload object: {payload}")

    if not _has_core_markers(work_dir):
        raise FileNotFoundError(
            f"Invalid staged code bundle at {work_dir}; missing core markers: "
            f"{', '.join(CORE_CODE_MARKERS)}"
        )

    print(f"[bootstrap] code staged at {work_dir}")
    missing = _missing_expected_markers(_path_marker_presence(work_dir))
    if missing:
        print(f"[bootstrap] staged code is missing optional expected markers: {list(missing)}")
    return work_dir


def stage_code(work: Path = WORK, input_base: Path | None = None) -> Path:
    """Backward-compatible wrapper for older local tests."""
    return stage_code_payload(input_root=input_base or INPUT_ROOT, work_dir=work)


def _module_status(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        return f"ERROR importing {module_name}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return str(version)


def _print_numpy_repair_suggestion(error: BaseException) -> None:
    print(f"[bootstrap] NumPy import appears broken: {error}")
    print("[bootstrap] Do not downgrade NumPy in a live Kaggle kernel.")
    print("[bootstrap] Suggested repair command, then restart the kernel:")
    repair_cmd = [
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        '"numpy>=2.0,<2.4"',
    ]
    print("[bootstrap] " + " ".join(repair_cmd))


def print_environment_versions() -> None:
    """Print dependency versions without modifying the environment."""
    print(f"[bootstrap] Python: {sys.version.split()[0]}")
    try:
        import numpy as np

        print(f"[bootstrap] NumPy: {np.__version__}")
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        _print_numpy_repair_suggestion(exc)

    try:
        import torch

        print(f"[bootstrap] torch: {torch.__version__}")
        print(f"[bootstrap] torch cuda: {torch.version.cuda}")
        print(f"[bootstrap] cuda available: {torch.cuda.is_available()}")
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        print(f"[bootstrap] ERROR importing torch: {exc}")

    if importlib.util.find_spec("torch_geometric") is None:
        print("[bootstrap] torch_geometric: not installed")
    else:
        print(f"[bootstrap] torch_geometric: {_module_status('torch_geometric')}")


def _install_missing_light_packages() -> None:
    missing = [
        pkg
        for pkg, mod in (("networkx", "networkx"),)
        if importlib.util.find_spec(mod) is None
    ]
    if not missing:
        return
    print(f"[bootstrap] installing missing light packages with --no-deps: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *missing])


def install_pyg() -> None:
    """Conservative optional PyG install path for fresh Kaggle kernels."""
    print_environment_versions()
    print("[bootstrap] Do not downgrade NumPy in a live Kaggle kernel.")

    import torch

    if importlib.util.find_spec("torch_geometric") is not None:
        print(f"[bootstrap] PyG already available; cuda_available={torch.cuda.is_available()}")
        _install_missing_light_packages()
        return

    torch_ver = torch.__version__.split("+")[0]
    cuda_ver = torch.version.cuda or "12.1"
    cuda_tag = "cu" + cuda_ver.replace(".", "")
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    print(f"[bootstrap] installing PyG wheels with --no-deps")
    print(f"[bootstrap] torch={torch.__version__} cuda={cuda_ver} wheels={wheel_url}")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
            "-f",
            wheel_url,
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.5.3",
        ]
    )
    _install_missing_light_packages()

    import torch_geometric  # noqa: F401

    print(f"[bootstrap] PyG OK; cuda_available={torch.cuda.is_available()}")


def _find_csvs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in ELLIPTIC_FILES:
        hits = list(root.rglob(name))
        if hits:
            found[name] = hits[0]
    return found


def stage_elliptic(work: Path = WORK, input_base: Path | None = None) -> None:
    """Stage the three Elliptic CSVs from any attached dataset, by filename."""
    raw = work / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if all((raw / n).is_file() for n in ELLIPTIC_FILES):
        print("[bootstrap] Elliptic CSVs already in data/raw/")
        return

    base = input_base or INPUT_ROOT
    csvs: dict[str, Path] = {}
    if base.is_dir():
        for name in ELLIPTIC_FILES:
            hits = list(base.rglob(name))
            if hits:
                csvs[name] = hits[0]
    missing = [n for n in ELLIPTIC_FILES if n not in csvs]
    if missing:
        raise FileNotFoundError(
            "Elliptic CSVs not found in any attached dataset: "
            f"{missing}. Attach the Elliptic Bitcoin dataset under any name "
            f"(expected files: {', '.join(ELLIPTIC_FILES)}).\n\nDetected inputs:\n"
            f"{_input_debug_listing(base)}"
        )
    for name, src in csvs.items():
        shutil.copy2(src, raw / name)
    print(f"[bootstrap] Elliptic CSVs copied to {raw}")


# Canonical fraud-dataset NPZ signature. Distinguishing key: feature dimension
# (DGraphFin = 17, T-Finance = 10) plus DGraphFin's optional edge_type/mask keys.
# Lets the bootstrap stage an NPZ uploaded under ANY filename, by content.
_FRAUD_NPZ_REQUIRED_KEYS = frozenset({"x", "y", "edge_index", "edge_timestamp"})


def _peek_npy_shape(npz_path: Path, key: str):
    """Read one NPZ member's shape from its .npy header (no full array load)."""
    try:
        import numpy.lib.format as nf
        with zipfile.ZipFile(npz_path) as zf:
            member = next(
                (n for n in zf.namelist() if n in (key + ".npy", key)), None
            )
            if member is None:
                return None
            with zf.open(member) as fh:
                version = nf.read_magic(fh)
                if version == (1, 0):
                    shape, _f, _d = nf.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _f, _d = nf.read_array_header_2_0(fh)
                else:
                    shape, _f, _d = nf._read_array_header(fh, version)
            return shape
    except Exception:
        return None


def _classify_fraud_npz(npz_path: Path) -> str | None:
    """Identify a fraud-dataset NPZ by CONTENT, returning its canonical filename.

    Name-agnostic: an archive uploaded as ``anything.npz`` still resolves to
    ``dgraphfin.npz`` or ``tfinance.npz`` from its keys + feature dimension, so
    Kaggle dataset *and* file names are both irrelevant.
    """
    try:
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as arch:
            keys = set(arch.files)
    except Exception:
        return None
    if not _FRAUD_NPZ_REQUIRED_KEYS.issubset(keys):
        return None
    shape = _peek_npy_shape(npz_path, "x")
    if shape is None:  # header peek failed — fall back to a full load of x only
        try:
            import numpy as np
            with np.load(npz_path, allow_pickle=False) as arch:
                shape = arch["x"].shape
        except Exception:
            shape = None
    feat = shape[1] if shape and len(shape) >= 2 else None
    if "edge_type" in keys or feat == 17:
        return "dgraphfin.npz"
    if feat == 10:
        return "tfinance.npz"
    return None


def stage_optional_npz(work: Path = WORK, input_base: Path | None = None) -> None:
    """Stage DGraphFin / T-Finance NPZ archives — by filename, then by content.

    Both the Kaggle dataset name and the NPZ filename are irrelevant: canonical
    names are matched first (fast path), then any still-missing target is
    resolved by scanning every ``*.npz`` and classifying it by content.
    """
    raw = work / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    base = input_base or INPUT_ROOT
    if not base.is_dir():
        return

    targets = ("dgraphfin.npz", "tfinance.npz")

    # 1) Fast path: match by canonical filename anywhere under the inputs.
    for fname in targets:
        if (raw / fname).is_file():
            continue
        hits = list(base.rglob(fname))
        if hits:
            shutil.copy2(hits[0], raw / fname)
            print(f"[bootstrap] staged {fname} from {hits[0]} (by name)")

    # 2) Name-agnostic path: classify any *.npz by content for what's still missing.
    missing = [f for f in targets if not (raw / f).is_file()]
    if not missing:
        return
    for npz in sorted(base.rglob("*.npz")):
        target = _classify_fraud_npz(npz)
        if target in missing and not (raw / target).is_file():
            shutil.copy2(npz, raw / target)
            print(f"[bootstrap] staged {target} from {npz} (by content: name-agnostic)")
            missing = [f for f in targets if not (raw / f).is_file()]
            if not missing:
                return


def stage_cpu_results(work: Path = WORK, input_base: Path | None = None) -> None:
    """Optional: reuse prior validation_clean runs from any attached dataset."""
    base = input_base or INPUT_ROOT
    if not base.is_dir():
        return
    hits = list(base.rglob("validation_clean/runs.csv"))
    if not hits:
        return
    src = hits[0].parent
    dst = work / "results" / "runs" / "validation_clean"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[bootstrap] restored CPU validation_clean from {src}")


def bootstrap(
    *,
    install_deps: bool = False,
    work: Path = WORK,
    input_base: Path | None = None,
) -> Path:
    """Stage code/data and prepare imports. Dependency installs are opt-in."""
    print_environment_versions()
    if install_deps:
        print("[bootstrap] install_deps requested; using conservative --no-deps installs only.")
        install_pyg()
    else:
        print("[bootstrap] install_deps=False; using the existing Kaggle environment.")
        print("[bootstrap] Do not downgrade NumPy in a live Kaggle kernel.")

    base = input_base or INPUT_ROOT
    stage_code_payload(input_root=base, work_dir=work)
    stage_elliptic(work, base)
    stage_optional_npz(work, base)
    stage_cpu_results(work, base)
    os.chdir(work)
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    return work


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cmd_to_string(cmd) -> str:
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(part) for part in cmd)
    return str(cmd)


def run_cmd_logged(cmd, log_path=None, cwd=None, check=True, env: dict | None = None) -> dict:
    """Run a command with streamed output and structured metadata."""
    cwd_path = Path(cwd) if cwd is not None else WORK
    started = _utc_now()
    command_text = _cmd_to_string(cmd)
    print(f"[run] {command_text}", flush=True)

    log_handle = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"started_at_utc={started}\n")
        log_handle.write(f"cwd={cwd_path}\n")
        log_handle.write(f"cmd={command_text}\n\n")
        log_handle.flush()

    shell = isinstance(cmd, str)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd_path,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log_handle is not None:
            log_handle.write(line)
            log_handle.flush()
    return_code = proc.wait()
    finished = _utc_now()
    metadata = {
        "cmd": command_text,
        "cwd": str(cwd_path),
        "log_path": str(log_path) if log_path is not None else "",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "return_code": int(return_code),
    }
    if log_handle is not None:
        log_handle.write(f"\nfinished_at_utc={finished}\n")
        log_handle.write(f"return_code={return_code}\n")
        log_handle.close()
    COMMAND_LOG.append(metadata)
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return metadata


def run_cmd(args: list[str], *, env: dict | None = None) -> dict:
    merged = os.environ.copy()
    # Stream child output live; Kaggle pipes can otherwise make long runs look frozen.
    merged.setdefault("PYTHONUNBUFFERED", "1")
    merged.setdefault("RUNS_PROGRESS", "1")
    if env:
        merged.update(env)
    log_dir = Path("/kaggle/working/runs_outputs/logs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    script = Path(args[0]).name if args else "python"
    log_path = log_dir / f"{stamp}__{script}.log"
    return run_cmd_logged([sys.executable, "-u"] + args, log_path=log_path, cwd=WORK, check=True, env=merged)
