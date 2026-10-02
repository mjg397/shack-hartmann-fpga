"""Strict loading, staging, and provenance for the RTL ROM images."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config as C


class ROMValidationError(ValueError):
    """A required ROM is absent or does not match the RTL declaration."""


@dataclass(frozen=True)
class ROMSpec:
    words: int
    width_bits: int
    signed: bool = False
    encoded_bits: int | None = None


# These dimensions are the declarations in the authoritative modules under rtl/.
REQUIRED_ROMS: dict[str, ROMSpec] = {
    "reciprocal_u16_q18.mem": ROMSpec(1 << C.SI_BITS, C.RECIP_OUT_BITS),
    "slopes_ref_x.mem": ROMSpec(C.NUM_SUBAPS_GRID, C.SLOPE_BITS, True, 28),
    "slopes_ref_y.mem": ROMSpec(C.NUM_SUBAPS_GRID, C.SLOPE_BITS, True, 28),
    "subaperture_bitmap.mem": ROMSpec(1, C.NUM_SUBAPS_GRID),
    "e_matrix_x.mem": ROMSpec(C.NUM_MODES * C.NUM_SUBAPS_VALID, C.EMATRIX_BITS, True, 20),
    "e_matrix_y.mem": ROMSpec(C.NUM_MODES * C.NUM_SUBAPS_VALID, C.EMATRIX_BITS, True, 20),
}

_HEX = re.compile(r"^[0-9a-fA-F]+$")


def read_hex_words(path: Path, spec: ROMSpec) -> list[int]:
    """Read one plain $readmemh image and require its exact declared shape."""
    path = Path(path)
    if not path.is_file():
        raise ROMValidationError(f"required ROM is missing: {path}")
    tokens = path.read_text(encoding="ascii").split()
    if len(tokens) != spec.words:
        raise ROMValidationError(
            f"{path}: expected {spec.words} hex words, found {len(tokens)}")
    limit = 1 << spec.width_bits
    encoded_bits = spec.encoded_bits or spec.width_bits
    encoded_limit = 1 << encoded_bits
    values: list[int] = []
    for i, token in enumerate(tokens):
        if not _HEX.fullmatch(token):
            raise ROMValidationError(
                f"{path}: malformed hex word {token!r} at index {i}")
        value = int(token, 16)
        if value >= encoded_limit:
            raise ROMValidationError(
                f"{path}: word {i} (0x{value:X}) exceeds its {encoded_bits}-bit encoding")
        if value >= limit:
            low = value & (limit - 1)
            sign_set = bool(low & (1 << (spec.width_bits - 1)))
            extension = value >> spec.width_bits
            expected_extension = (1 << (encoded_bits - spec.width_bits)) - 1
            if not (spec.signed and sign_set and extension == expected_extension):
                raise ROMValidationError(
                    f"{path}: word {i} (0x{value:X}) is not valid "
                    f"{spec.width_bits}-bit sign extension")
            value = low
        values.append(value)
    return values


def validate_rom_set(source_dir: Path = C.RTL_DIR) -> dict[str, dict]:
    """Validate all DUT ROMs and return a byte-level provenance manifest."""
    source_dir = Path(source_dir).resolve()
    manifest: dict[str, dict] = {}
    for name, spec in REQUIRED_ROMS.items():
        path = source_dir / name
        values = read_hex_words(path, spec)
        data = path.read_bytes()
        manifest[name] = {
            "source": str(path),
            "words": len(values),
            "width_bits": spec.width_bits,
            "signed": spec.signed,
            "encoded_bits": spec.encoded_bits or spec.width_bits,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return manifest


def stage_rom_set(destination: Path, source_dir: Path = C.RTL_DIR) -> dict[str, dict]:
    """Validate and copy exactly the ROMs read by the DUT into its run CWD."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = validate_rom_set(source_dir)
    for name, record in manifest.items():
        target = destination / name
        shutil.copyfile(record["source"], target)
        staged = target.read_bytes()
        staged_hash = hashlib.sha256(staged).hexdigest()
        if staged_hash != record["sha256"]:
            raise ROMValidationError(f"staged ROM hash mismatch: {target}")
        record["staged_path"] = str(target)
        record["staged_sha256"] = staged_hash
    return manifest
