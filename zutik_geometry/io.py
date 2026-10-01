"""Small, explicit file adapters for point clouds and pose results."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .registration import validate_point_cloud
from .types import ModelCatalogEntry, PoseHypothesis


_UNIT_TO_METRES = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "cm": 1e-2}


def _convert_units(points: np.ndarray, units: str) -> np.ndarray:
    key = units.lower()
    if key not in _UNIT_TO_METRES:
        raise ValueError(f"unsupported point-cloud units {units!r}; use m, cm, or mm")
    return points * _UNIT_TO_METRES[key]


def _load_ply(path: Path, *, include_normals: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as stream:
        header_lines: list[bytes] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: PLY header has no end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = [line.decode("ascii", errors="strict").strip().split() for line in header_lines]
        if not header or header[0] != ["ply"]:
            raise ValueError(f"{path}: not a PLY file")
        format_tokens = next((line[1:] for line in header if line and line[0] == "format"), None)
        if format_tokens is None:
            raise ValueError(f"{path}: PLY format is missing")
        vertex_index = next((i for i, line in enumerate(header) if line[:2] == ["element", "vertex"]), None)
        if vertex_index is None:
            raise ValueError(f"{path}: PLY vertex element is missing")
        vertex_count = int(header[vertex_index][2])
        property_lines: list[list[str]] = []
        for line in header[vertex_index + 1 :]:
            if line and line[0] == "element":
                break
            if line and line[0] == "property" and len(line) >= 3 and line[1] != "list":
                property_lines.append(line[1:3])
        names = [item[1] for item in property_lines]
        if not all(name in names for name in ("x", "y", "z")):
            raise ValueError(f"{path}: PLY vertices need x, y, and z properties")
        positions = [names.index(name) for name in ("x", "y", "z")]
        normal_names = ("nx", "ny", "nz")
        normal_positions = [names.index(name) for name in normal_names] if all(
            name in names for name in normal_names
        ) else None
        if format_tokens[0] == "ascii":
            rows: list[list[float]] = []
            normal_rows: list[list[float]] = []
            for _ in range(vertex_count):
                values = stream.readline().decode("ascii").split()
                if len(values) < len(names):
                    raise ValueError(f"{path}: truncated ASCII PLY vertex")
                rows.append([float(values[index]) for index in positions])
                if normal_positions is not None:
                    normal_rows.append([float(values[index]) for index in normal_positions])
            points = np.asarray(rows, dtype=float)
            if include_normals:
                normals = None if normal_positions is None else np.asarray(normal_rows, dtype=float)
                return points, normals
            return points
        if format_tokens[0] not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"{path}: unsupported PLY format {format_tokens[0]!r}")
        scalar_formats = {
            "char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
            "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
            "float": "f", "float32": "f", "double": "d", "float64": "d",
        }
        property_types = [item[0] for item in property_lines]
        if any(item not in scalar_formats for item in property_types):
            raise ValueError(f"{path}: unsupported PLY scalar property type")
        endian = "<" if format_tokens[0] == "binary_little_endian" else ">"
        record_format = endian + "".join(scalar_formats[item] for item in property_types)
        record_size = struct.calcsize(record_format)
        unpacker = struct.Struct(record_format)
        rows = []
        normal_rows = []
        for _ in range(vertex_count):
            values = unpacker.unpack(stream.read(record_size))
            rows.append([float(values[index]) for index in positions])
            if normal_positions is not None:
                normal_rows.append([float(values[index]) for index in normal_positions])
        points = np.asarray(rows, dtype=float)
        if include_normals:
            normals = None if normal_positions is None else np.asarray(normal_rows, dtype=float)
            return points, normals
        return points


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load XYZ points from NPY/NPZ/XYZ/CSV/OBJ or ASCII/binary PLY."""

    point_path = Path(path)
    if not point_path.is_file():
        raise FileNotFoundError(f"point-cloud file does not exist: {point_path}")
    suffix = point_path.suffix.lower()
    if suffix == ".npy":
        points = np.load(point_path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(point_path, allow_pickle=False)
        if "points" not in archive:
            raise ValueError(f"{point_path}: NPZ must contain a 'points' array")
        points = archive["points"]
    elif suffix == ".ply":
        points = _load_ply(point_path)
    elif suffix == ".obj":
        vertices = []
        for line in point_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if fields and fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(value) for value in fields[1:4]])
        points = np.asarray(vertices, dtype=float)
    elif suffix in {".xyz", ".txt", ".csv"}:
        points = np.loadtxt(point_path, delimiter="," if suffix == ".csv" else None, ndmin=2)
    else:
        raise ValueError(f"unsupported point-cloud format: {point_path.suffix}")
    return validate_point_cloud(points, str(point_path))


def load_model_catalog_entry(
    path: str | Path,
    *,
    model_id: str | None = None,
    semantic_label: str | None = None,
    units: str = "m",
    representation: str = "custom",
) -> ModelCatalogEntry:
    """Load a CAD/model point sample and convert its declared units to metres."""

    model_path = Path(path)
    normals = None
    if model_path.suffix.lower() == ".ply":
        loaded = _load_ply(model_path, include_normals=True)
        raw_points, normals = loaded
        points = _convert_units(raw_points, units)
    else:
        points = _convert_units(load_point_cloud(model_path), units)
    return ModelCatalogEntry(
        model_id=model_id or model_path.stem,
        points=points,
        semantic_label=semantic_label,
        representation=representation,
        normals=normals,
    )


def load_transform(path: str | Path) -> np.ndarray:
    transform_path = Path(path)
    if transform_path.suffix.lower() == ".npy":
        matrix = np.load(transform_path, allow_pickle=False)
    elif transform_path.suffix.lower() == ".json":
        matrix = np.asarray(json.loads(transform_path.read_text(encoding="utf-8")), dtype=float)
    else:
        matrix = np.loadtxt(transform_path)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{transform_path}: expected a 4x4 transform")
    return matrix


def load_depth_array(path: str | Path) -> np.ndarray:
    depth_path = Path(path)
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path, allow_pickle=False)
    elif depth_path.suffix.lower() == ".npz":
        archive = np.load(depth_path, allow_pickle=False)
        if "depth" not in archive:
            raise ValueError(f"{depth_path}: NPZ must contain a 'depth' array")
        depth = archive["depth"]
    else:
        raise ValueError("the core CLI accepts .npy/.npz depth; BOP PNG uses the BOP adapter")
    if np.asarray(depth).ndim != 2:
        raise ValueError("depth array must be 2-D")
    return np.asarray(depth)


def write_hypotheses(path: str | Path, hypotheses: list[PoseHypothesis]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "frame_convention": "T_A_B maps coordinates from frame B into frame A",
        "translation_units": "m",
        "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
