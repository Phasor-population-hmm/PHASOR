"""Locate per-contrast clustering results for Fourier and sustained/transient stages."""

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
