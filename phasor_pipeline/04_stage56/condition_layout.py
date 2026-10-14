"""Locate per-contrast clustering results for Fourier and sustained/transient stages."""

import os
from pathlib import Path


def resolve_cluster_dir(condition_dir: Path) -> Path:
    """Return the folder that contains Stage4_Visualization for this contrast.

    A contrast folder that already holds Stage4 is returned as-is.
    A PHASOR classification folder such as ``04_fourier/c90`` is resolved to
    ``02_processing/filtered/c90``.
    """
    condition_dir = Path(condition_dir)
    marker = condition_dir / "Stage4_Visualization" / "X_processed.npy"
    if marker.exists():
        return condition_dir

    name = condition_dir.name
    candidates = []
    if condition_dir.parent.name == "04_fourier":
        run = condition_dir.parent.parent
        candidates.append(run / "02_processing" / "filtered" / name)
        candidates.append(run / "Filtered_Data" / name)
    candidates.append(condition_dir.parent / "filtered" / name)

    tried = []
    for cand in candidates:
        tried.append(str(cand))
        if (cand / "Stage4_Visualization" / "X_processed.npy").exists():
            return cand

    looked = ", ".join(tried) if tried else str(condition_dir)
    raise FileNotFoundError(
        f"No clustering results for {condition_dir}. "
        f"Expected Stage4_Visualization/X_processed.npy in that folder or in {looked}. "
        "Run Filter by contrast, then UMAP + OPTICS clustering."
    )


def _digits(value) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def contrast_names(contrast) -> list[str]:
    text = str(contrast)
    digits = _digits(text)
    names = [text]
    if digits:
        names.extend([
            f"c{digits}",
            f"C{digits}",
            f"c{digits}_f2Hz",
            f"c{digits}_f4Hz",
            f"C-{digits}",
            f"c-{digits}",
        ])
    ordered = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def run_roots(root: Path) -> list[Path]:
    """Folders that may hold this recording's filtered contrasts."""
    root = Path(root)
    found = []

    def add(path: Path) -> None:
        if path not in found:
            found.append(path)

    add(root)
    parts = root.parts
    if "example_data" in parts:
        index = parts.index("example_data")
        if index + 1 < len(parts):
            recording = parts[index + 1]
            repo = Path(*parts[:index])
            add(repo / "phasor_output" / "runs" / recording)
            output_root = os.environ.get("PHASOR_OUTPUT_ROOT")
            if output_root:
                add(Path(output_root) / "runs" / recording)
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            add(Path(*parts[: index + 2]))
    organized = _organized_root(root)
    if organized is not None:
        add(organized)
    if root.name in {"06_analysis", "05_hmm", "03_labels", "04_fourier", "01_preprocessing", "02_processing", "filtered"}:
        add(root.parent)
    return found


def _has_contrast_data(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "Stage4_Visualization").is_dir():
        return True
    return any(path.glob("*filtered*.npz"))


def contrast_dir(root: Path, contrast) -> Path:
    """Return the contrast folder, including c0.

    The website writes Filter output at ``02_processing/filtered/c0``.
    Older runs use ``Filtered_Data/c0`` or ``clustering/filtered/c0_f2Hz``.
    """
    names = contrast_names(contrast)
    digits = _digits(contrast)
    label = f"c{digits}" if digits else str(contrast)
    tried = []
    relative = (
        ("02_processing", "filtered"),
        ("Filtered_Data",),
        ("Filtered Data",),
        ("Filtered data",),
        ("clustering", "filtered"),
    )
    for base in run_roots(root):
        if base.name in names and _has_contrast_data(base):
            return base
        if base.name in {"filtered", "Filtered_Data", "Filtered Data", "Filtered data"}:
            for name in names:
                cand = base / name
                tried.append(cand)
                if cand.is_dir():
                    return cand
        for parts in relative:
            folder = base.joinpath(*parts)
            for name in names:
                cand = folder / name
                tried.append(cand)
                if cand.is_dir():
                    return cand
    expected = []
    for base in run_roots(root):
        expected.append(str(base / "02_processing" / "filtered" / label))
        expected.append(str(base / "Filtered_Data" / label))
    looked = "; ".join(expected)
    raise FileNotFoundError(f"No {label} folder under {root}. Looked for {looked}")


def _organized_root(path: Path) -> Path | None:
    markers = {"01_preprocessing", "02_processing", "03_labels", "04_fourier", "05_hmm", "06_analysis"}
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in markers and index > 0:
            return Path(*parts[:index])
    return None


def _workspace(path: Path) -> Path | None:
    parts = Path(path).parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            return Path(*parts[: index + 2])
    return _organized_root(path)


def _first_file(paths) -> Path | None:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def stage5_csv(condition_dir: Path) -> Path | None:
    condition_dir = Path(condition_dir)
    names = contrast_names(condition_dir.name)
    candidates = [condition_dir / "Stage5_Cell_Fourier_Analysis" / "cell_fourier_results.csv"]
    workspace = _workspace(condition_dir)
    if workspace is not None:
        for name in names:
            candidates.append(workspace / "04_fourier" / name / "Stage5_Cell_Fourier_Analysis" / "cell_fourier_results.csv")
    return _first_file(candidates)


def stage6_csv(condition_dir: Path) -> Path | None:
    condition_dir = Path(condition_dir)
    folders = ("Stage6_Cell_SustainedTransient", "Stage6_SustainedTransient_CellLevel")
    files = (
        "cell_level_sustained_transient_KEPT.csv",
        "cell_level_sustained_transient.csv",
        "cell_level_sustained_transient_ALL.csv",
    )
    names = contrast_names(condition_dir.name)
    candidates = []
    for folder in folders:
        for name in files:
            candidates.append(condition_dir / folder / name)
    workspace = _workspace(condition_dir)
    if workspace is not None:
        for contrast_name in names:
            for folder in folders:
                for name in files:
                    candidates.append(workspace / "04_fourier" / contrast_name / folder / name)
    return _first_file(candidates)


def label_map_csv(root: Path, condition_name) -> Path | None:
    digits = _digits(condition_name)
    filenames = []
    if digits:
        filenames.append(f"c{digits}_txtcells_label_index_map.csv")
    filenames.append(f"{condition_name}_txtcells_label_index_map.csv")
    candidates = []
    for base in run_roots(root):
        for folder in (
            base / "label_index_maps",
            base / "03_labels" / "label_index_maps",
            base / "03_labels",
        ):
            for filename in filenames:
                candidates.append(folder / filename)
    return _first_file(candidates)


def sibling_c0(cluster_dir: Path) -> Path | None:
    """Return a clustered c0 folder next to this contrast, when one exists."""
    cluster_dir = Path(cluster_dir)
    if cluster_dir.name.startswith("c0"):
        return None
    for name in ("c0", "c0_f2Hz"):
        cand = cluster_dir.parent / name
        if (cand / "Stage4_Visualization" / "X_processed.npy").exists():
            return cand
    return None
