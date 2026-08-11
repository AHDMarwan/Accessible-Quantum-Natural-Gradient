"""Safe serialization helpers for calibrated AQNG optimizer state.

The archive format is a ZIP containing JSON metadata and NumPy ``.npy`` arrays.
No pickle payloads are written or loaded.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Mapping

import numpy as np

from aqng_readouts import ReadoutDesign

_STATE_VERSION = 1


def _write_array(zf: zipfile.ZipFile, name: str, value) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    zf.writestr(name, buffer.getvalue())


def _read_array(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as handle:
        return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def save_optimizer_state(
    path,
    *,
    config: Mapping[str, object],
    readout: str,
    designs: Mapping[str, ReadoutDesign] | None,
) -> Path:
    """Save optimizer configuration and calibrated readout designs.

    Objective/probability callables and cached metric tensors are intentionally
    excluded. A restored optimizer must bind a compatible probability callable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "aqng-optimizer-state",
        "version": _STATE_VERSION,
        "config": dict(config),
        "readout": str(readout),
        "design_names": [] if designs is None else sorted(designs.keys()),
    }

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
        if designs is not None:
            for name, design in designs.items():
                prefix = f"designs/{name}"
                info = {
                    "name": design.name,
                    "rank": int(design.rank),
                    "centered_dimension": int(design.centered_dimension),
                }
                zf.writestr(f"{prefix}/metadata.json", json.dumps(info, sort_keys=True))
                _write_array(zf, f"{prefix}/support_indices.npy", design.support_indices)
                _write_array(
                    zf,
                    f"{prefix}/reference_probabilities.npy",
                    design.reference_probabilities,
                )
                _write_array(zf, f"{prefix}/basis.npy", design.basis)
                _write_array(zf, f"{prefix}/outcome_features.npy", design.outcome_features)
    return target


def load_optimizer_state(path) -> tuple[dict, dict[str, ReadoutDesign]]:
    """Load a state archive written by :func:`save_optimizer_state`."""
    source = Path(path)
    with zipfile.ZipFile(source, "r") as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        if metadata.get("format") != "aqng-optimizer-state":
            raise ValueError("not an AQNG optimizer state archive")
        if int(metadata.get("version", -1)) != _STATE_VERSION:
            raise ValueError(
                f"unsupported AQNG state version {metadata.get('version')!r}; "
                f"expected {_STATE_VERSION}"
            )

        designs: dict[str, ReadoutDesign] = {}
        for key in metadata.get("design_names", []):
            prefix = f"designs/{key}"
            info = json.loads(zf.read(f"{prefix}/metadata.json").decode("utf-8"))
            designs[key] = ReadoutDesign(
                name=str(info["name"]),
                rank=int(info["rank"]),
                centered_dimension=int(info["centered_dimension"]),
                support_indices=_read_array(zf, f"{prefix}/support_indices.npy"),
                reference_probabilities=_read_array(
                    zf, f"{prefix}/reference_probabilities.npy"
                ),
                basis=_read_array(zf, f"{prefix}/basis.npy"),
                outcome_features=_read_array(zf, f"{prefix}/outcome_features.npy"),
            )
    return metadata, designs
