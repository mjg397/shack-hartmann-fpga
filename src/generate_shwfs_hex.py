#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shwfs_utils import (
    generate_shwfs_case,
    parse_zernike_coefficient_file,
    run_hcipy_estimation,
    write_shwfs_hex,
)


def write_named_coefficients(path: Path, coeffs: np.ndarray, mode_labels: list[str], header_prefix: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"mode {header_prefix}_coeff\n")
        for label, coeff in zip(mode_labels, coeffs):
            handle.write(f"{label} {coeff:.10f}\n")


def write_comparison(path: Path, true_coeffs: np.ndarray, estimated_coeffs: np.ndarray, mode_labels: list[str]) -> None:
    comparison = np.column_stack((true_coeffs, estimated_coeffs, estimated_coeffs - true_coeffs))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("mode true_coeff estimated_coeff error\n")
        for label, true_coeff, estimated_coeff, error in zip(
            mode_labels,
            comparison[:, 0],
            comparison[:, 1],
            comparison[:, 2],
        ):
            handle.write(
                f"{label} {true_coeff:.10f} {estimated_coeff:.10f} {error:.10f}\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an image_rotating.hex-style SHWFS frame from HCIPy Zernike coefficients."
        )
    )
    parser.add_argument(
        "coefficients",
        type=Path,
        help="Path to a coefficient file. Each line may be a single float or a named coefficient ending in a float.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hcipy_shwfs.hex"),
        help="Output path for the newline-delimited hex byte stream.",
    )
    parser.add_argument(
        "--num-lenslets",
        type=int,
        default=16,
        help="Lenslet grid size used by the SHWFS model.",
    )
    parser.add_argument(
        "--demo-image-path",
        type=Path,
        default=None,
        help="Optional path for the rendered demo image. Omit to skip image output.",
    )
    parser.add_argument(
        "--estimated-output",
        type=Path,
        default=Path("hcipy_estimated_zernikes_named.txt"),
        help="Output path for HCIPy-estimated Zernike coefficients.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("hcipy_zernike_comparison.txt"),
        help="Output path for true-vs-estimated coefficient comparison.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    coeffs = parse_zernike_coefficient_file(args.coefficients)
    simulation = generate_shwfs_case(
        true_coeffs=coeffs,
        num_lenslets=args.num_lenslets,
        num_zernike=int(coeffs.size),
        demo_image_path=None if args.demo_image_path is None else str(args.demo_image_path),
    )
    quantized = write_shwfs_hex(args.output, simulation["image_aber"])

    estimation = run_hcipy_estimation(
        image=simulation["image_aber"],
        estimator=simulation["estimator"],
        reference_slopes=simulation["slopes_ref"],
        reconstruction_matrix=simulation["reconstruction_matrix"],
        zernike_modes=simulation["zernike_modes"],
        aperture=simulation["aperture"],
        measured_opd_field=simulation["input_opd_field"],
        shwfs=simulation["shwfs"],
    )

    write_named_coefficients(
        args.estimated_output,
        np.asarray(estimation["estimated_coeffs"], dtype=np.float64),
        list(simulation["mode_labels"]),
        "hcipy_estimated",
    )
    write_comparison(
        args.comparison_output,
        np.asarray(simulation["true_coeffs"], dtype=np.float64),
        np.asarray(estimation["estimated_coeffs"], dtype=np.float64),
        list(simulation["mode_labels"]),
    )

    print(f"Wrote {quantized.size} bytes to {args.output}")
    print(f"Image shape: {quantized.shape}")
    print(f"Value range: [{int(quantized.min())}, {int(quantized.max())}]")
    print(f"Wrote HCIPy estimates to {args.estimated_output}")
    print(f"Wrote comparison to {args.comparison_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())