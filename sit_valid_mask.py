from __future__ import annotations
from typing import Iterable, Mapping, Tuple
import numpy as np

SIC_THRESHOLD = 0.10


def one_cell_coastal_buffer(land_mask: np.ndarray, *, diagonal: bool = True) -> np.ndarray:
    """Return land plus its one-cell neighbourhood.

    Parameters
    ----------
    land_mask
        Boolean array whose True values identify land cells on the native
        PIOMAS grid.
    diagonal
        If True (default), all eight neighbouring cells are included.  Set to
        False to use only the four edge-sharing neighbours.
    """
    land = np.asarray(land_mask, dtype=bool)
    if land.ndim != 2:
        raise ValueError("land_mask must have dimensions (y, x).")

    padded = np.pad(land, 1, mode="constant", constant_values=False)
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    buffered = np.zeros_like(land, dtype=bool)
    ny, nx = land.shape
    for dy, dx in offsets:
        buffered |= padded[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
    return buffered


def make_valid_mask(
    land_mask: np.ndarray,
    sic: np.ndarray,
    *sit_fields: np.ndarray,
    sic_threshold: float = SIC_THRESHOLD,
    diagonal_coastal_buffer: bool = True,
) -> np.ndarray:
    """Construct the common validity mask for SIT samples.

    ``sic`` and each SIT field can have shape ``(y, x)`` or leading temporal
    dimensions followed by ``(y, x)``.  The output is broadcast to the common
    sample shape.  All supplied fields must be finite at a valid location.
    """
    if not sit_fields:
        raise ValueError("Provide at least one SIT field when building a valid mask.")

    land_and_coast = one_cell_coastal_buffer(
        land_mask, diagonal=diagonal_coastal_buffer
    )
    sic_array = np.asarray(sic, dtype=float)
    valid = (~land_and_coast) & np.isfinite(sic_array) & (sic_array >= sic_threshold)

    for field in sit_fields:
        valid = valid & np.isfinite(np.asarray(field, dtype=float))
    return np.asarray(valid, dtype=bool)


def fill_invalid_for_tensor(field: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace invalid values by zero for storage/tensor construction only."""
    values = np.asarray(field, dtype=np.float32).copy()
    values[~np.asarray(valid_mask, dtype=bool)] = 0.0
    return values


def prepare_paired_sit_fields(
    piomas_sit: np.ndarray,
    reference_sit: np.ndarray,
    sic: np.ndarray,
    land_mask: np.ndarray,
    *,
    sic_threshold: float = SIC_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zero-filled PIOMAS/reference tensors and their common mask.

    The mask, not the fill value, determines which cells contribute to loss or
    metrics.
    """
    valid_mask = make_valid_mask(
        land_mask,
        sic,
        piomas_sit,
        reference_sit,
        sic_threshold=sic_threshold,
    )
    return (
        fill_invalid_for_tensor(piomas_sit, valid_mask),
        fill_invalid_for_tensor(reference_sit, valid_mask),
        valid_mask,
    )


def masked_l1_loss(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> float:
    """Calculate masked L1 loss; invalid cells make no contribution."""
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("The valid mask contains no retained cells.")
    return float(np.mean(np.abs(np.asarray(prediction)[mask] - np.asarray(target)[mask])))


def masked_metrics(prediction: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray) -> Mapping[str, float]:
    """Calculate RMSE, MAE, signed bias, R2, and thick-ice MAE (reference >2 m)."""
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("The valid mask contains no retained cells.")

    pred = np.asarray(prediction, dtype=float)[mask]
    ref = np.asarray(reference, dtype=float)[mask]
    residual = pred - ref
    sse = np.sum(residual**2)
    sst = np.sum((ref - ref.mean()) ** 2)
    thick = ref > 2.0

    return {
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "r2": float(1.0 - sse / sst) if sst > 0.0 else np.nan,
        "mae2_m": float(np.mean(np.abs(residual[thick]))) if np.any(thick) else np.nan,
        "n_valid": int(mask.sum()),
        "n_thick_ice": int(thick.sum()),
    }

