#!/usr/bin/env python
# /// script
# dependencies = [
#   "astropy",
#   "numpy",
# ]
# ///
"""End-to-end TLISI/IQA FITS pipeline.

For 3D TLISI cubes this runs iqa_harfft_cuda, finds the most significant
tile in the 2D harmonic-summed score matrix, uses that tile's TLISI time
series on FITS axis 3 to choose one difference image, then localises all
detected boxes on that difference image.

For 2D IQA/result matrices this skips CUDA, uses the 2D input directly as r,
and requires --image-fits for source localisation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


DEFAULT_SLICE_SCRIPT = Path(__file__).with_name("FIP_prototype_slice.py")
SKA_PYPI_INDEX = "https://artefact.skao.int/repository/pypi-internal/simple"
SLICE_VENV_NAME = ".fip-slice-venv"
SLICE_PYTHON_VERSION = "3.12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TLISI/IQA detection through CUDA, SDP/PFL imaging, and RA/Dec analysis."
    )
    parser.add_argument("iqa_fits", help="Input TLISI/IQA FITS cube, or existing 2D result FITS.")
    parser.add_argument("--ms", default=None, help="Measurement Set used by FIP_prototype_slice.py for 3D inputs.")
    parser.add_argument("--result-fits", default="tlisi_result.fits", help="2D CUDA/result FITS path.")
    parser.add_argument("--image-fits", default=None, help="Image FITS used for source localization.")
    parser.add_argument(
        "--difference-base-fits",
        default=None,
        help="Optional base FITS to subtract from --image-fits before localizing 2D-input sources.",
    )
    parser.add_argument(
        "--difference-fits",
        default="difference_image.fits",
        help="Output FITS path for the localization difference image.",
    )
    parser.add_argument("--cuda-bin", default="./iqa_harfft_cuda", help="Compiled iqa_harfft_cuda executable.")
    parser.add_argument("--fip-bin", default="./FastImagingPipe/build/Meson/src/fip", help="Compiled fip executable.")
    parser.add_argument("--max-orientation-shift", type=int, default=16, help="Max pixel shift allowed when matching FIP image to prototype image.")
    parser.add_argument("--slice-script", default=str(DEFAULT_SLICE_SCRIPT), help="FIP_prototype_slice.py path.")
    parser.add_argument("--uv", default="uv", help="uv executable used to run FIP_prototype_slice.py.")
    parser.add_argument("--python", default="python", help="Python executable when --no-uv is set.")
    parser.add_argument("--no-uv", action="store_true", help="Run slice script with --python instead of uv run python.")
    parser.add_argument("--uv-cache-dir", default="/tmp/uv-cache", help="UV_CACHE_DIR for subprocess calls.")
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        default=4096,
        help="Image size for FIP_prototype_slice.py.",
    )
    parser.add_argument(
        "--pixel_size_arcsec",
        type=float,
        default=0.8820000686134000789,
        help="Pixel size for FIP_prototype_slice.py.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=None,
        help="Cell size in radians. Overrides --pixel_size_arcsec when supplied.",
    )
    parser.add_argument("--weighting", default="natural", choices=["natural", "uniform"])
    parser.add_argument("--epsilon", default=None, help="Optional epsilon forwarded to FIP_prototype_slice.py.")
    parser.add_argument("--do_single", default="True", help="Optional do_single value forwarded to FIP_prototype_slice.py.")
    parser.add_argument("--do_w_stacking", default=None, help="Optional do_w_stacking value forwarded to FIP_prototype_slice.py.")
    parser.add_argument(
        "--cupy-package",
        default="cupy-cuda12x<14",
        help="CuPy package spec installed into the automatic slice environment.",
    )
    parser.add_argument(
        "--skip-slice-env-setup",
        action="store_true",
        help="Skip uv venv / uv pip install before automatic 3D snapshot FITS generation.",
    )
    parser.add_argument("--time-axis", type=int, choices=[1, 2, 3], default=3, help="TLISI cube time axis.")
    parser.add_argument(
        "--non-periodic",
        action="store_true",
        help="Skip FFT/harmonic summing and detect tiles from sum(1 - TLISI) over time.",
    )
    parser.add_argument("--max-harmonic", type=int, default=19, help="Maximum harmonic for CUDA summing.")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size used by analysis.m.")
    parser.add_argument("--sigma-threshold", type=float, default=5.0, help="Sigma threshold used by analysis.m.")
    parser.add_argument(
        "--tlisi-prominence-frac",
        type=float,
        default=0.25,
        help="Required local-large prominence as a fraction of the strongest tile TLISI range.",
    )
    parser.add_argument(
        "--local-source-sigma",
        type=float,
        default=None,
        help=(
            "Local image S/N threshold used to make a source-pixel mask inside each TLISI tile box. "
            "Defaults to 3.0 for effective 2D inputs and 5.0 for effective 3D inputs."
        ),
    )
    parser.add_argument(
        "--min-gaussian-corr",
        type=float,
        default=0.6,
        help="Minimum correlation with a moment-fitted elliptical Gaussian inside each source crop.",
    )
    parser.add_argument(
        "--r-orientation",
        default="transpose",
        choices=[
            "auto",
            "none",
            "transpose",
            "flipud",
            "fliplr",
            "fliplr-flipud",
            "flipud-transpose",
            "fliplr-transpose",
            "transpose-flipud",
            "transpose-fliplr",
            "transpose-fliplr-flipud",
            "matlab-flipfliplr-transpose",
        ],
        help=(
            "Orientation applied to the 2D result before 1-r. "
            "Default matches analysis.m r'. "
            "matlab-flipfliplr-transpose matches flip(flip(r,2)')."
        ),
    )
    args = parser.parse_args()
    if args.cell_size is not None:
        args.pixel_size_arcsec = args.cell_size * 180.0 / np.pi * 3600.0
    return args


def fits_effective_naxis(path: str | Path) -> tuple[int, tuple[int, ...]]:
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        naxis = int(header["NAXIS"])
        axis_lengths = tuple(int(header[f"NAXIS{i}"]) for i in range(1, naxis + 1))
        effective_naxis = sum(length > 1 for length in axis_lengths)
        return effective_naxis, axis_lengths


def load_iqa_cube_tyx(path: str | Path, time_axis_fits: int, r_orientation: str) -> np.ndarray:
    data = np.asarray(fits.getdata(path, memmap=True), dtype=np.float64)
    if data.ndim != 3:
        raise RuntimeError(f"Expected a 3D TLISI cube in {path}, got shape {data.shape}")

    time_axis = time_axis_fits - 1
    other_axes = [axis for axis in range(3) if axis != time_axis]
    y_axis, x_axis = other_axes
    numpy_axes = [2 - time_axis, 2 - y_axis, 2 - x_axis]
    cube = np.transpose(data, numpy_axes)
    return np.stack([orient_result(frame, r_orientation) for frame in cube], axis=0)


def load_iqa_cube_layout(path: str | Path, time_axis_fits: int) -> np.ndarray:
    data = np.asarray(fits.getdata(path, memmap=True), dtype=np.float64)
    if data.ndim != 3:
        raise RuntimeError(f"Expected a 3D TLISI cube in {path}, got shape {data.shape}")

    time_axis = time_axis_fits - 1
    other_axes = [axis for axis in range(3) if axis != time_axis]
    y_axis, x_axis = other_axes
    numpy_axes = [2 - time_axis, 2 - y_axis, 2 - x_axis]
    return np.transpose(data, numpy_axes)


def write_python_cuda_fallback(args: argparse.Namespace, result_path: Path) -> Path:
    cube = load_iqa_cube_layout(args.iqa_fits, args.time_axis)
    n_time = cube.shape[0]
    if n_time < 2:
        raise RuntimeError("Need at least two TLISI snapshots for harmonic analysis")

    finite = np.isfinite(cube)
    valid_counts = np.count_nonzero(finite, axis=0)
    nonfinite_samples = int(cube.size - np.count_nonzero(finite))
    all_nonfinite_cells = int(np.count_nonzero(valid_counts == 0))
    print(
        "Masking non-finite TLISI samples before FFT: "
        f"{nonfinite_samples}/{cube.size} samples are NaN/Inf; "
        f"{all_nonfinite_cells}/{valid_counts.size} cells have no finite samples.",
        flush=True,
    )
    sums = np.nansum(np.where(finite, cube, 0.0), axis=0)
    means = np.divide(sums, valid_counts, out=np.zeros_like(sums), where=valid_counts > 0)
    sanitized = np.where(finite, cube, means[None, :, :])
    window = np.hamming(n_time).astype(np.float64)[:, None, None]
    fft = np.fft.rfft((sanitized - means[None, :, :]) * window, axis=0)
    power = np.abs(fft / float(n_time)) ** 2
    if power.shape[0] > 2:
        power[1:-1, :, :] *= 2.0

    result = np.full(cube.shape[1:], -np.inf, dtype=np.float64)
    length = power.shape[0]
    x_new = np.linspace(0.0, 1.0, length) if length > 1 else np.zeros(1, dtype=np.float64)
    for k_pos in x_new:
        p_sum = np.zeros(cube.shape[1:], dtype=np.float64)
        for harmonic in range(1, args.max_harmonic + 1):
            comp_len = length // harmonic
            if comp_len <= 0:
                continue
            pos = k_pos * float(comp_len - 1) if comp_len > 1 else 0.0
            lo = int(np.floor(pos))
            hi = min(lo + 1, comp_len - 1)
            frac = pos - float(lo)
            vlo = power[lo * harmonic]
            vhi = power[hi * harmonic]
            p_sum += vlo + frac * (vhi - vlo)
        result = np.maximum(result, p_sum)

    result = np.where(valid_counts > 0, result, np.nan).astype(np.float32)
    min_idx = np.nanargmin(np.where(finite, cube, np.inf), axis=0)
    finite_result = np.where(np.isfinite(result), result, -np.inf)
    max_cell = int(np.argmax(finite_result))
    max_y, max_x = np.unravel_index(max_cell, result.shape)
    max_eta = float(result[max_y, max_x])
    header = fits.Header()
    header["MAXX"] = (int(max_x + 1), "1-based x index of max result cell")
    header["MAXY"] = (int(max_y + 1), "1-based y index of max result cell")
    header["MINSNAP"] = (int(min_idx[max_y, max_x] + 1), "1-based min-IQA snapshot at max cell")
    if np.isfinite(max_eta):
        header["MAXETA"] = (max_eta, "maximum harmonic-summed score")
    fits.PrimaryHDU(data=result, header=header).writeto(result_path, overwrite=True)
    print(
        "Wrote 2D harmonic detection matrix: "
        f"{result_path} shape={result.shape} finite={int(np.isfinite(result).sum())}/{result.size} "
        f"maxEta={max_eta:.9g} maxCell=({max_y + 1}, {max_x + 1})",
        flush=True,
    )
    return result_path


def fits_has_nonfinite_data(path: str | Path) -> bool:
    data = np.asarray(fits.getdata(path, memmap=True), dtype=np.float64)
    return bool(np.any(~np.isfinite(data)))


def run_cuda(args: argparse.Namespace) -> Path:
    result_path = Path(args.result_fits)
    if fits_has_nonfinite_data(args.iqa_fits):
        print(
            "Input cube contains non-finite values; running harmonic FFT in Python with non-finite values masked.",
            flush=True,
        )
        return write_python_cuda_fallback(args, result_path)

    cmd = [
        args.cuda_bin,
        args.iqa_fits,
        str(result_path),
        "--max-harmonic",
        str(args.max_harmonic),
    ]
    if args.time_axis is not None:
        cmd.extend(["--time-axis", str(args.time_axis)])

    print("Running CUDA harmonic FFT:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("CUDA harmonic FFT failed; retrying harmonic analysis in Python with non-finite values masked.", flush=True)
        return write_python_cuda_fallback(args, result_path)

    return result_path


def default_image_fits(ms: str, im_size: int, time_idx: int, weighting: str) -> str:
    ms_name = os.path.basename(ms.rstrip("/"))
    stem = ms_name[:-3] if ms_name.endswith(".ms") else ms_name
    return f"{stem}_{im_size}p_t{time_idx}-{time_idx}_{weighting}.fits"


def slice_workdir(args: argparse.Namespace) -> Path:
    return Path(args.slice_script).resolve().parent


def slice_venv(workdir: Path) -> Path:
    return workdir / SLICE_VENV_NAME


def slice_python(workdir: Path) -> Path:
    return slice_venv(workdir) / "bin" / "python"


def slice_env_has_dependencies(workdir: Path, env: dict[str, str]) -> bool:
    python = slice_python(workdir)
    if not python.exists():
        return False
    check = [
        str(python),
        "-c",
        "import astropy, cupy, ska_sdp_func; import casacore.tables",
    ]
    return subprocess.run(check, cwd=workdir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_slice_env(args: argparse.Namespace) -> None:
    if args.no_uv or args.skip_slice_env_setup or getattr(args, "_slice_env_ready", False):
        return

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", args.uv_cache_dir)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    workdir = slice_workdir(args)

    if slice_env_has_dependencies(workdir, env):
        print(f"Using existing SDP/PFL imaging environment: {slice_python(workdir)}", flush=True)
        args._slice_env_ready = True
        return

    commands = []
    if not slice_python(workdir).exists():
        commands.append([args.uv, "venv", str(slice_venv(workdir)), "--python", SLICE_PYTHON_VERSION])
    commands.extend(
        [
            [
                args.uv,
                "pip",
                "install",
                "--python",
                str(slice_python(workdir)),
                "--extra-index-url",
                SKA_PYPI_INDEX,
                "ska-sdp-func",
            ],
            [
                args.uv,
                "pip",
                "install",
                "--python",
                str(slice_python(workdir)),
                args.cupy_package,
                "python-casacore",
                "astropy",
            ],
        ]
    )
    for cmd in commands:
        print("Preparing SDP/PFL imaging environment:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=workdir, env=env)
    if not slice_env_has_dependencies(workdir, env):
        raise RuntimeError(f"SDP/PFL imaging environment is missing dependencies after setup: {slice_python(workdir)}")
    args._slice_env_ready = True


def run_slice(args: argparse.Namespace, time_idx: int, image_fits: Path | None = None) -> Path:
    if not args.ms:
        raise RuntimeError("A Measurement Set argument is required for 3D IQA cube inputs")

    ensure_slice_env(args)

    image_fits = image_fits or Path(default_image_fits(args.ms, args.image_size, time_idx, args.weighting))
    if not image_fits.is_absolute():
        image_fits = slice_workdir(args) / image_fits
    runner = [args.python] if args.no_uv else [str(slice_python(slice_workdir(args)))]
    script_path = Path(args.slice_script).resolve()
    cmd = [
        *runner,
        str(script_path),
        f"--im_size={args.image_size}",
        f"--pixel_size_arcsec={args.pixel_size_arcsec}",
        f"--do_single={args.do_single}",
        "--time_idx_start",
        str(time_idx),
        "--time_idx_end",
        str(time_idx),
        args.ms,
        "--weighting",
        args.weighting,
        "--fitsfile",
        str(image_fits),
    ]
    if args.epsilon is not None:
        cmd.append(f"--epsilon={args.epsilon}")
    if args.do_w_stacking is not None:
        cmd.append(f"--do_w_stacking={args.do_w_stacking}")

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", args.uv_cache_dir)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    print("Running SDP/PFL imaging:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=slice_workdir(args), env=env)
    return image_fits


def connected_boxes(mask: np.ndarray, tile_size: int) -> list[dict[str, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    boxes: list[dict[str, int]] = []
    nrow, ncol = mask.shape

    for start_r, start_c in zip(*np.nonzero(mask)):
        if visited[start_r, start_c]:
            continue

        rmin = rmax = int(start_r)
        cmin = cmax = int(start_c)
        visited[start_r, start_c] = True
        queue: deque[tuple[int, int]] = deque([(int(start_r), int(start_c))])

        while queue:
            r, c = queue.popleft()
            rmin = min(rmin, r)
            rmax = max(rmax, r)
            cmin = min(cmin, c)
            cmax = max(cmax, c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < nrow and 0 <= cc < ncol and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        queue.append((rr, cc))

        boxes.append(
            {
                "cell_rmin": rmin,
                "cell_rmax": rmax,
                "cell_cmin": cmin,
                "cell_cmax": cmax,
                "xmin": cmin * tile_size + 1,
                "xmax": (cmax + 1) * tile_size,
                "ymin": rmin * tile_size + 1,
                "ymax": (rmax + 1) * tile_size,
            }
        )

    return boxes


def robust_rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0.0 and np.isfinite(mad):
        return 1.4826 * mad
    std = float(np.std(finite))
    return std if np.isfinite(std) else 0.0


def connected_component(mask: np.ndarray, start_row: int, start_col: int) -> np.ndarray:
    component = np.zeros(mask.shape, dtype=bool)
    if not mask[start_row, start_col]:
        component[start_row, start_col] = True
        return component

    nrow, ncol = mask.shape
    component[start_row, start_col] = True
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])

    while queue:
        row, col = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < nrow and 0 <= cc < ncol and mask[rr, cc] and not component[rr, cc]:
                    component[rr, cc] = True
                    queue.append((rr, cc))

    return component


def gaussian_like_fit(
    img2: np.ndarray,
    component: np.ndarray,
    peak: float,
    min_gaussian_corr: float,
) -> dict[str, float] | None:
    weights = np.where(component, img2, 0.0)
    weights[weights < 0] = 0.0
    weight_sum = float(np.nansum(weights))
    if weight_sum <= 0.0 or np.count_nonzero(component) < 3:
        return None

    yy, xx = np.mgrid[: img2.shape[0], : img2.shape[1]]
    x0 = float(np.nansum(xx * weights) / weight_sum)
    y0 = float(np.nansum(yy * weights) / weight_sum)
    dx = xx - x0
    dy = yy - y0
    var_x = float(np.nansum(weights * dx * dx) / weight_sum)
    var_y = float(np.nansum(weights * dy * dy) / weight_sum)
    cov_xy = float(np.nansum(weights * dx * dy) / weight_sum)
    covariance = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(eigvals)) or np.any(eigvals <= 0.0):
        return None

    sigma_minor, sigma_major = (float(v) for v in np.sqrt(eigvals))
    if sigma_minor < 0.5 or sigma_major > max(img2.shape):
        return None

    try:
        inv_cov = np.linalg.inv(covariance + np.eye(2) * 1e-6)
    except np.linalg.LinAlgError:
        return None

    q = inv_cov[0, 0] * dx * dx + 2.0 * inv_cov[0, 1] * dx * dy + inv_cov[1, 1] * dy * dy
    model = peak * np.exp(-0.5 * q)
    fit_region = component | (model > 0.1 * peak)
    observed = img2[fit_region]
    expected = model[fit_region]
    if observed.size < 4 or float(np.nanstd(observed)) == 0.0 or float(np.nanstd(expected)) == 0.0:
        return None

    corr = float(np.corrcoef(observed, expected)[0, 1])
    if not np.isfinite(corr) or corr < min_gaussian_corr:
        return None

    return {
        "gaussianCorr": corr,
        "gaussianSigmaMinor": sigma_minor,
        "gaussianSigmaMajor": sigma_major,
    }


def find_source_position(
    crop: np.ndarray,
    local_source_sigma: float,
    min_gaussian_corr: float = 0.6,
) -> dict[str, float] | None:
    img = np.asarray(crop, dtype=np.float64)
    img2 = img - np.nanmedian(img)
    peak_index = int(np.nanargmax(img2))
    row, col = np.unravel_index(peak_index, img2.shape)
    peak = float(img2[row, col])

    rms = robust_rms(img2)
    if rms > 0.0 and np.isfinite(rms):
        source_mask = img2 >= local_source_sigma * rms
        if not source_mask[row, col]:
            return None
    else:
        source_mask = img2 > 0.0
        if peak <= 0.0 or not np.isfinite(peak):
            return None
    component = connected_component(source_mask, int(row), int(col))
    gaussian_fit = gaussian_like_fit(img2, component, peak, min_gaussian_corr)
    if gaussian_fit is None:
        return None

    weights = np.where(component, img2, 0.0)
    weights[weights < 0] = 0.0

    weight_sum = float(np.nansum(weights))
    if weight_sum <= 0:
        x_centroid = float(col + 1)
        y_centroid = float(row + 1)
    else:
        yy, xx = np.mgrid[: img2.shape[0], : img2.shape[1]]
        x_centroid = float(np.nansum((xx + 1.0) * weights) / weight_sum)
        y_centroid = float(np.nansum((yy + 1.0) * weights) / weight_sum)

    return {
        "xCentroid": x_centroid,
        "yCentroid": y_centroid,
        "Peak": peak,
        "localRms": rms,
        "localSnr": peak / rms if rms > 0.0 and np.isfinite(rms) else np.inf,
        "nSourcePix": float(np.count_nonzero(component)),
        **gaussian_fit,
    }


def find_source_position_2d(crop: np.ndarray, local_source_sigma: float) -> dict[str, float]:
    img = np.asarray(crop, dtype=np.float64)
    img2 = img - np.nanmedian(img)
    peak_index = int(np.nanargmax(img2))
    row, col = np.unravel_index(peak_index, img2.shape)
    peak = float(img2[row, col])

    rms = robust_rms(img2)
    if rms > 0.0 and np.isfinite(rms):
        source_mask = img2 >= local_source_sigma * rms
    else:
        source_mask = img2 > 0.0
    component = connected_component(source_mask, int(row), int(col))
    weights = np.where(component, img2, 0.0)
    weights[weights < 0] = 0.0

    weight_sum = float(np.nansum(weights))
    if weight_sum <= 0:
        x_centroid = float(col + 1)
        y_centroid = float(row + 1)
    else:
        yy, xx = np.mgrid[: img2.shape[0], : img2.shape[1]]
        x_centroid = float(np.nansum((xx + 1.0) * weights) / weight_sum)
        y_centroid = float(np.nansum((yy + 1.0) * weights) / weight_sum)

    return {
        "xCentroid": x_centroid,
        "yCentroid": y_centroid,
        "Peak": peak,
        "localRms": rms,
        "nSourcePix": float(np.count_nonzero(component)),
    }


def orient_result(result: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "none":
        return result
    if orientation == "transpose":
        return result.T
    if orientation == "flipud":
        return np.flipud(result)
    if orientation == "fliplr":
        return np.fliplr(result)
    if orientation == "fliplr-flipud":
        return np.flipud(np.fliplr(result))
    if orientation == "flipud-transpose":
        return np.flipud(result).T
    if orientation == "fliplr-transpose":
        return np.fliplr(result).T
    if orientation == "transpose-flipud":
        return np.flipud(result.T)
    if orientation == "transpose-fliplr":
        return np.fliplr(result.T)
    if orientation == "transpose-fliplr-flipud":
        return np.flipud(np.fliplr(result.T))
    if orientation == "matlab-flipfliplr-transpose":
        return np.flipud(np.fliplr(result).T)
    raise ValueError(f"Unknown r orientation: {orientation}")

def preprocess_orientation_image(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise RuntimeError("Orientation image has no finite pixels")

    x = image.copy()
    x[~finite] = 0.0
    lo = float(np.nanpercentile(x[finite], 1.0))
    hi = float(np.nanpercentile(x[finite], 99.9))
    if hi <= lo:
        hi = float(np.nanmax(x[finite]))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)

    x = np.clip(x, lo, hi) - lo
    x = np.log1p(1000.0 * x / (hi - lo)) / np.log1p(1000.0)

    step = max(1, int(np.ceil(max(x.shape) / max_size)))
    return x[::step, ::step]


def shifted_corr(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, int, int]:
    if a.shape != b.shape:
        raise RuntimeError(f"Orientation comparison shape mismatch: {a.shape} vs {b.shape}")

    best_score = -np.inf
    best_dx = 0
    best_dy = 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ay0 = max(0, dy)
            ay1 = min(a.shape[0], a.shape[0] + dy)
            ax0 = max(0, dx)
            ax1 = min(a.shape[1], a.shape[1] + dx)

            by0 = max(0, -dy)
            by1 = min(b.shape[0], b.shape[0] - dy)
            bx0 = max(0, -dx)
            bx1 = min(b.shape[1], b.shape[1] - dx)

            aa = a[ay0:ay1, ax0:ax1]
            bb = b[by0:by1, bx0:bx1]

            mask = np.isfinite(aa) & np.isfinite(bb)
            if np.count_nonzero(mask) < 100:
                continue

            av = aa[mask] - np.nanmedian(aa[mask])
            bv = bb[mask] - np.nanmedian(bb[mask])
            denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
            if denom <= 0:
                continue

            score = float(np.sum(av * bv) / denom)
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    return best_score, best_dx, best_dy


def run_fip_image(args: argparse.Namespace, time_idx: int) -> Path:
    out = Path(f"fip_image_t{time_idx}.fits")
    cmd = [
        args.fip_bin,
        "image",
        "-i",
        args.ms,
        "-o",
        str(out),
        "--snapshot",
        str(time_idx),
        "--image-size",
        str(args.image_size),
    ]
    if args.cell_size is not None:
        cmd.extend(["--cell-size", str(args.cell_size)])

    print("Running FastImagingPipe image:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def choose_auto_orientation(args: argparse.Namespace, time_idx: int) -> tuple[str, Path]:
    prototype_fits = run_slice(args, time_idx)
    fip_fits = run_fip_image(args, time_idx)

    prototype = np.squeeze(fits.getdata(prototype_fits)).astype(np.float64)
    fip_image = np.squeeze(fits.getdata(fip_fits)).astype(np.float64)

    prototype_small = preprocess_orientation_image(prototype)
    scale = max(1.0, max(fip_image.shape) / max(prototype_small.shape))
    max_shift_small = max(1, int(round(args.max_orientation_shift / scale)))

    candidates = [
        "none",
        "transpose",
        "flipud",
        "fliplr",
        "fliplr-flipud",
        "flipud-transpose",
        "fliplr-transpose",
        "transpose-flipud",
        "transpose-fliplr",
        "transpose-fliplr-flipud",
        "matlab-flipfliplr-transpose",
    ]

    scores = []
    for orientation in candidates:
        oriented = orient_result(fip_image, orientation)
        oriented_small = preprocess_orientation_image(oriented)
        if oriented_small.shape != prototype_small.shape:
            continue
        score, dx, dy = shifted_corr(prototype_small, oriented_small, max_shift_small)
        scores.append((score, orientation, dx, dy))

    if not scores:
        raise RuntimeError("Could not auto-select orientation: no comparable orientation candidates")

    score, orientation, dx, dy = max(scores)
    print(
        f"Auto-selected r orientation: {orientation} "
        f"score={score:.6g} dx={dx} dy={dy} compared_snapshot={time_idx}",
        flush=True,
    )
    return orientation, prototype_fits

def detect_boxes(
    result_fits: Path,
    tile_size: int,
    sigma_threshold: float,
    r_orientation: str,
    invert_score: bool = True,
) -> list[dict[str, int]]:
    result = np.squeeze(fits.getdata(result_fits)).astype(np.float64)
    if result.ndim != 2:
        raise RuntimeError(f"Expected 2D result matrix after squeezing {result_fits}, got shape {result.shape}")

    # analysis.m did r = r'; this is now controlled by --r-orientation.
    oriented = orient_result(result, r_orientation)
    score = 1.0 - oriented if invert_score else oriented
    score_std = float(np.nanstd(score))
    if score_std == 0.0 or not np.isfinite(score_std):
        return []
    sigma = (score - np.nanmean(score)) / score_std
    mask = sigma >= sigma_threshold
    boxes = connected_boxes(mask, tile_size)
    for box in boxes:
        cell_region = np.s_[
            box["cell_rmin"] : box["cell_rmax"] + 1,
            box["cell_cmin"] : box["cell_cmax"] + 1,
        ]
        local_sigma = sigma[cell_region]
        local_peak = np.unravel_index(int(np.nanargmax(local_sigma)), local_sigma.shape)
        box["maxSigma"] = float(local_sigma[local_peak])
        box["peakCellRow"] = box["cell_rmin"] + int(local_peak[0])
        box["peakCellCol"] = box["cell_cmin"] + int(local_peak[1])
    return boxes


def detect_boxes_from_zscore(
    zscore: np.ndarray,
    tile_size: int,
    sigma_threshold: float,
) -> list[dict[str, int]]:
    if zscore.ndim != 2:
        raise RuntimeError(f"Expected a 2D z-score matrix, got shape {zscore.shape}")

    mask = zscore >= sigma_threshold
    boxes = connected_boxes(mask, tile_size)
    for box in boxes:
        cell_region = np.s_[
            box["cell_rmin"] : box["cell_rmax"] + 1,
            box["cell_cmin"] : box["cell_cmax"] + 1,
        ]
        local_zscore = zscore[cell_region]
        local_peak = np.unravel_index(int(np.nanargmax(local_zscore)), local_zscore.shape)
        box["maxSigma"] = float(local_zscore[local_peak])
        box["peakCellRow"] = box["cell_rmin"] + int(local_peak[0])
        box["peakCellCol"] = box["cell_cmin"] + int(local_peak[1])
    return boxes


def non_periodic_tlisi_zscore(args: argparse.Namespace, r_orientation: str) -> np.ndarray:
    cube_tyx = load_iqa_cube_tyx(args.iqa_fits, args.time_axis, r_orientation)
    anomaly_sum = np.sum(1.0 - cube_tyx, axis=0)
    mean = float(np.mean(anomaly_sum))
    std = float(np.std(anomaly_sum, ddof=0))
    if std == 0.0 or not np.isfinite(std):
        raise RuntimeError("Cannot z-score non-periodic TLISI map because std is zero or non-finite")
    return (anomaly_sum - mean) / std


def write_non_periodic_result_fits(result_fits: Path, zscore: np.ndarray) -> None:
    header = fits.Header()
    header["METHOD"] = ("NONPER", "Non-periodic TLISI scoring")
    header["FORMULA"] = ("ZSUM1MT", "zscore(sum(1 - TLISI, time))")
    fits.PrimaryHDU(data=np.asarray(zscore, dtype=np.float32), header=header).writeto(result_fits, overwrite=True)
    print(f"Wrote non-periodic TLISI z-score map: {result_fits}", flush=True)


def load_localization_image(image_fits: Path, difference_base_fits: Path | None) -> tuple[np.ndarray, fits.Header]:
    with fits.open(image_fits, memmap=True) as hdul:
        image_data = np.squeeze(hdul[0].data).astype(np.float64)
        header = hdul[0].header.copy()
    if image_data.ndim != 2:
        raise RuntimeError(f"Expected 2D image after squeezing {image_fits}, got shape {image_data.shape}")

    if difference_base_fits is not None:
        base_data = np.squeeze(fits.getdata(difference_base_fits, memmap=True)).astype(np.float64)
        if base_data.ndim != 2:
            raise RuntimeError(
                f"Expected 2D base image after squeezing {difference_base_fits}, got shape {base_data.shape}"
            )
        if base_data.shape != image_data.shape:
            raise RuntimeError(
                f"Difference image shape mismatch: {image_fits} has {image_data.shape}, "
                f"{difference_base_fits} has {base_data.shape}"
            )
        image_data = np.abs(image_data - base_data)

    return image_data, header


def load_localization_image_2d(image_fits: Path, difference_base_fits: Path | None) -> tuple[np.ndarray, fits.Header]:
    with fits.open(image_fits, memmap=True) as hdul:
        image_data = np.squeeze(hdul[0].data).astype(np.float64)
        header = hdul[0].header
    if image_data.ndim != 2:
        raise RuntimeError(f"Expected 2D image after squeezing {image_fits}, got shape {image_data.shape}")

    if difference_base_fits is not None:
        base_data = np.squeeze(fits.getdata(difference_base_fits, memmap=True)).astype(np.float64)
        if base_data.ndim != 2:
            raise RuntimeError(
                f"Expected 2D base image after squeezing {difference_base_fits}, got shape {base_data.shape}"
            )
        if base_data.shape != image_data.shape:
            raise RuntimeError(
                f"Difference image shape mismatch: {image_fits} has {image_data.shape}, "
                f"{difference_base_fits} has {base_data.shape}"
            )
        image_data = image_data - base_data

    return image_data, header


def write_difference_fits(
    difference_fits: Path,
    image_data: np.ndarray,
    header: fits.Header,
    minuend_time_idx: int | None = None,
    subtrahend_time_idx: int | None = None,
    min_time_idx: int | None = None,
    transient_time_idx: int | None = None,
) -> None:
    out_header = header.copy()
    out_header["DIFFOP"] = ("ABS(MINUEND-SUBTRAHEND)", "Difference image operation")
    if minuend_time_idx is not None:
        out_header["MINUEND"] = (int(minuend_time_idx), "Snapshot index used as minuend")
    if subtrahend_time_idx is not None:
        out_header["SUBTRAH"] = (int(subtrahend_time_idx), "Snapshot index used as subtrahend")
    if min_time_idx is not None:
        out_header["MINTIME"] = (int(min_time_idx), "Selected minimum TLISI snapshot index")
    if transient_time_idx is not None:
        out_header["TRTIME"] = (int(transient_time_idx), "Selected transient TLISI snapshot index")
    fits.PrimaryHDU(data=np.asarray(image_data, dtype=np.float32), header=out_header).writeto(
        difference_fits, overwrite=True
    )
    print(f"Wrote difference image FITS: {difference_fits}", flush=True)


def localize_boxes(
    boxes: list[dict[str, int]],
    image_data: np.ndarray,
    header: fits.Header,
    local_source_sigma: float,
    min_gaussian_corr: float = 0.6,
) -> list[dict[str, float]]:
    wcs = WCS(header).celestial

    rows: list[dict[str, float]] = []
    for source_id, box in enumerate(boxes, start=1):
        xmin = box["xmin"]
        xmax = box["xmax"]
        ymin = box["ymin"]
        ymax = box["ymax"]
        xmin_clip = max(1, xmin)
        xmax_clip = min(image_data.shape[1], xmax)
        ymin_clip = max(1, ymin)
        ymax_clip = min(image_data.shape[0], ymax)
        crop = image_data[ymin_clip - 1 : ymax_clip, xmin_clip - 1 : xmax_clip]
        if crop.size == 0:
            continue

        position = find_source_position(crop, local_source_sigma, min_gaussian_corr)
        if position is None:
            continue
        xcentroid = xmin_clip + position["xCentroid"] - 1
        ycentroid = ymin_clip + position["yCentroid"] - 1

        ra_centroid, dec_centroid = (float(v) for v in wcs.all_pix2world(xcentroid, ycentroid, 1))
        rows.append(
            {
                "xCentroid": xcentroid,
                "yCentroid": ycentroid,
                "Peak": position["Peak"],
                "localRms": position["localRms"],
                "localSnr": position["localSnr"],
                "nSourcePix": position["nSourcePix"],
                "gaussianCorr": position["gaussianCorr"],
                "gaussianSigmaMinor": position["gaussianSigmaMinor"],
                "gaussianSigmaMajor": position["gaussianSigmaMajor"],
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "xminSearch": xmin_clip,
                "xmaxSearch": xmax_clip,
                "yminSearch": ymin_clip,
                "ymaxSearch": ymax_clip,
                "RA_centroid_deg": ra_centroid,
                "Dec_centroid_deg": dec_centroid,
            }
        )

    return rows


def localize_boxes_2d(
    boxes: list[dict[str, int]],
    image_data: np.ndarray,
    header: fits.Header,
    local_source_sigma: float,
) -> list[dict[str, float]]:
    wcs = WCS(header).celestial

    rows: list[dict[str, float]] = []
    for source_id, box in enumerate(boxes, start=1):
        xmin = box["xmin"]
        xmax = box["xmax"]
        ymin = box["ymin"]
        ymax = box["ymax"]
        xmin_clip = max(1, xmin)
        xmax_clip = min(image_data.shape[1], xmax)
        ymin_clip = max(1, ymin)
        ymax_clip = min(image_data.shape[0], ymax)
        crop = image_data[ymin_clip - 1 : ymax_clip, xmin_clip - 1 : xmax_clip]
        if crop.size == 0:
            continue

        position = find_source_position_2d(crop, local_source_sigma)
        xcentroid = xmin_clip + position["xCentroid"] - 1
        ycentroid = ymin_clip + position["yCentroid"] - 1

        ra_centroid, dec_centroid = (float(v) for v in wcs.all_pix2world(xcentroid, ycentroid, 1))
        rows.append(
            {
                "xCentroid": xcentroid,
                "yCentroid": ycentroid,
                "Peak": position["Peak"],
                "localRms": position["localRms"],
                "nSourcePix": position["nSourcePix"],
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "xminSearch": xmin_clip,
                "xmaxSearch": xmax_clip,
                "yminSearch": ymin_clip,
                "ymaxSearch": ymax_clip,
                "RA_centroid_deg": ra_centroid,
                "Dec_centroid_deg": dec_centroid,
            }
        )

    return rows


def analyze_result(
    result_fits: Path,
    image_fits: Path,
    difference_base_fits: Path | None,
    tile_size: int,
    sigma_threshold: float,
    r_orientation: str,
    local_source_sigma: float,
) -> list[dict[str, float]]:
    boxes = detect_boxes(result_fits, tile_size, sigma_threshold, r_orientation)
    image_data, header = load_localization_image_2d(image_fits, difference_base_fits)
    return localize_boxes_2d(boxes, image_data, header, local_source_sigma)


def select_difference_pair_from_series(series: np.ndarray, prominence_frac: float) -> tuple[int, int, int, int]:
    if series.ndim != 1:
        raise RuntimeError(f"Expected a 1D TLISI time series, got shape {series.shape}")
    if not np.any(np.isfinite(series)):
        raise RuntimeError("No finite TLISI values in strongest tile time series")
    if series.size < 2:
        raise RuntimeError("Need at least two TLISI snapshots to form a difference image")

    series_range = float(np.nanmax(series) - np.nanmin(series))
    prominence_threshold = prominence_frac * series_range

    def is_large_between_smalls(i: int) -> bool:
        if i < 1 or i + 1 >= series.size:
            return False
        if not np.isfinite(series[i - 1]) or not np.isfinite(series[i]) or not np.isfinite(series[i + 1]):
            return False
        prominence = float(series[i] - max(series[i - 1], series[i + 1]))
        return prominence >= prominence_threshold and series[i] > series[i - 1] and series[i] > series[i + 1]

    def pair_from_transient(k: int, i: int) -> tuple[int, int, int, int] | None:
        if not is_large_between_smalls(i):
            return None
        if series[i - 1] < series[i + 1]:
            return k, i, i + 1, i
        if i + 1 == series.size - 1:
            return k, i, i + 1, i
        if i + 2 < series.size:
            return k, i, i + 1, i + 2
        return None

    def nearest_finite(start: int, step: int) -> int | None:
        idx = start + step
        while 0 <= idx < series.size:
            if np.isfinite(series[idx]):
                return idx
            idx += step
        return None

    def adjacent_high_pair(k: int) -> tuple[int, int, int, int] | None:
        candidates = []
        for idx in (nearest_finite(k, -1), nearest_finite(k, 1)):
            if idx is not None and series[idx] > series[k]:
                candidates.append((float(series[idx] - series[k]), -abs(idx - k), idx))
        if not candidates:
            return None
        transient_time_idx = max(candidates)[2]
        return k, transient_time_idx, transient_time_idx, k

    finite_indices = [int(idx) for idx in np.argsort(series) if np.isfinite(series[int(idx)])]
    for k in finite_indices[:2]:
        if k == 0:
            edge_pair = pair_from_transient(k, 1)
            if edge_pair is not None:
                return edge_pair
            fallback_pair = adjacent_high_pair(k)
            if fallback_pair is not None:
                return fallback_pair
            return k, 0, 0, 1
        if k == series.size - 1:
            edge_pair = pair_from_transient(k, series.size - 2)
            if edge_pair is not None:
                return edge_pair
            fallback_pair = adjacent_high_pair(k)
            if fallback_pair is not None:
                return fallback_pair
            return k, series.size - 1, series.size - 1, series.size - 2

        peak_candidates: list[tuple[float, int]] = []
        centre_min = max(1, k - 2)
        centre_max = min(series.size - 2, k + 2)
        for i in range(centre_min, centre_max + 1):
            if is_large_between_smalls(i):
                prominence = float(series[i] - max(series[i - 1], series[i + 1]))
                peak_candidates.append((prominence, i))

        if peak_candidates:
            transient_time_idx = max(peak_candidates)[1]
            pair = pair_from_transient(k, transient_time_idx)
            if pair is None:
                continue
            return pair

        fallback_pair = adjacent_high_pair(k)
        if fallback_pair is not None:
            return fallback_pair

    raise RuntimeError("Cannot find a valid TLISI difference pair from the first two minima")


def analyze_3d_result(
    args: argparse.Namespace,
    result_fits: Path,
    tile_size: int,
    sigma_threshold: float,
    r_orientation: str,
    local_source_sigma: float,
    min_gaussian_corr: float,
    tlisi_prominence_frac: float,
) -> list[dict[str, float]]:
    initial_orientation = "none" if r_orientation == "auto" else r_orientation
    boxes = detect_boxes(result_fits, tile_size, sigma_threshold, initial_orientation, invert_score=False)
    if not boxes:
        return []

    strongest_box = max(boxes, key=lambda box: box["maxSigma"])
    cube_tyx = load_iqa_cube_tyx(args.iqa_fits, args.time_axis, initial_orientation)
    row = int(strongest_box["peakCellRow"])
    col = int(strongest_box["peakCellCol"])
    series = cube_tyx[:, row, col]
    min_time_idx, transient_time_idx, minuend_time_idx, subtrahend_time_idx = select_difference_pair_from_series(
        series, tlisi_prominence_frac
    )
    print(
        "Selected TLISI snapshots: "
        f"min={min_time_idx}, transient={transient_time_idx}, "
        f"minuend={minuend_time_idx}, subtrahend={subtrahend_time_idx}",
        flush=True,
    )
    minuend_fits = None
    if r_orientation == "auto":
        r_orientation, minuend_fits = choose_auto_orientation(args, minuend_time_idx)
        boxes = detect_boxes(result_fits, tile_size, sigma_threshold, r_orientation, invert_score=False)
        if not boxes:
            return []

    if minuend_fits is None:
        minuend_fits = run_slice(args, minuend_time_idx)
    subtrahend_fits = run_slice(args, subtrahend_time_idx)
    minuend_image, header = load_localization_image(minuend_fits, None)
    subtrahend_image, _ = load_localization_image(subtrahend_fits, None)
    difference_image = np.abs(minuend_image - subtrahend_image)
    write_difference_fits(
        Path(args.difference_fits),
        difference_image,
        header,
        minuend_time_idx=minuend_time_idx,
        subtrahend_time_idx=subtrahend_time_idx,
        min_time_idx=min_time_idx,
        transient_time_idx=transient_time_idx,
    )
    rows = localize_boxes(boxes, difference_image, header, local_source_sigma, min_gaussian_corr)
    for row in rows:
        row["minTimeIdx"] = float(min_time_idx)
        row["transientTimeIdx"] = float(transient_time_idx)
        row["minuendTimeIdx"] = float(minuend_time_idx)
        row["subtrahendTimeIdx"] = float(subtrahend_time_idx)
    return rows


def analyze_3d_non_periodic_result(
    args: argparse.Namespace,
    tile_size: int,
    sigma_threshold: float,
    r_orientation: str,
    local_source_sigma: float,
    min_gaussian_corr: float,
    tlisi_prominence_frac: float,
) -> list[dict[str, float]]:
    initial_orientation = "none" if r_orientation == "auto" else r_orientation
    zscore = non_periodic_tlisi_zscore(args, initial_orientation)
    write_non_periodic_result_fits(Path(args.result_fits), zscore)
    boxes = detect_boxes_from_zscore(zscore, tile_size, sigma_threshold)
    if not boxes:
        return []

    strongest_box = max(boxes, key=lambda box: box["maxSigma"])
    cube_tyx = load_iqa_cube_tyx(args.iqa_fits, args.time_axis, initial_orientation)
    row = int(strongest_box["peakCellRow"])
    col = int(strongest_box["peakCellCol"])
    series = cube_tyx[:, row, col]
    min_time_idx, transient_time_idx, minuend_time_idx, subtrahend_time_idx = select_difference_pair_from_series(
        series, tlisi_prominence_frac
    )
    print(
        "Selected TLISI snapshots: "
        f"min={min_time_idx}, transient={transient_time_idx}, "
        f"minuend={minuend_time_idx}, subtrahend={subtrahend_time_idx}",
        flush=True,
    )
    minuend_fits = None
    if r_orientation == "auto":
        r_orientation, minuend_fits = choose_auto_orientation(args, minuend_time_idx)
        zscore = non_periodic_tlisi_zscore(args, r_orientation)
        write_non_periodic_result_fits(Path(args.result_fits), zscore)
        boxes = detect_boxes_from_zscore(zscore, tile_size, sigma_threshold)
        if not boxes:
            return []

    if minuend_fits is None:
        minuend_fits = run_slice(args, minuend_time_idx)
    subtrahend_fits = run_slice(args, subtrahend_time_idx)
    minuend_image, header = load_localization_image(minuend_fits, None)
    subtrahend_image, _ = load_localization_image(subtrahend_fits, None)
    difference_image = np.abs(minuend_image - subtrahend_image)
    write_difference_fits(
        Path(args.difference_fits),
        difference_image,
        header,
        minuend_time_idx=minuend_time_idx,
        subtrahend_time_idx=subtrahend_time_idx,
        min_time_idx=min_time_idx,
        transient_time_idx=transient_time_idx,
    )
    rows = localize_boxes(boxes, difference_image, header, local_source_sigma, min_gaussian_corr)
    for row in rows:
        row["minTimeIdx"] = float(min_time_idx)
        row["transientTimeIdx"] = float(transient_time_idx)
        row["minuendTimeIdx"] = float(minuend_time_idx)
        row["subtrahendTimeIdx"] = float(subtrahend_time_idx)
    return rows


def print_rows(rows: list[dict[str, float]]) -> None:
    if not rows:
        print("No sources passed the sigma threshold.")
        return

    headers = [
        "RA_centroid_deg",
        "Dec_centroid_deg",
    ]
    if any("localSnr" in row for row in rows):
        headers.append("localSnr")
    if any("minuendTimeIdx" in row for row in rows):
        headers.extend(["minTimeIdx", "transientTimeIdx", "minuendTimeIdx", "subtrahendTimeIdx"])
    print("\t".join(headers))
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if key in {
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "xminSearch",
                "xmaxSearch",
                "yminSearch",
                "ymaxSearch",
                "nSourcePix",
                "minTimeIdx",
                "transientTimeIdx",
                "minuendTimeIdx",
                "subtrahendTimeIdx",
            }:
                values.append(str(int(value)))
            else:
                values.append(f"{value:.15g}")
        print("\t".join(values))


def main() -> int:
    args = parse_args()
    effective_naxis, axis_lengths = fits_effective_naxis(args.iqa_fits)
    difference_base_fits = Path(args.difference_base_fits) if args.difference_base_fits else None

    if effective_naxis == 3:
        local_source_sigma = args.local_source_sigma if args.local_source_sigma is not None else 5.0
        if args.image_fits or args.difference_base_fits:
            raise RuntimeError(
                "Effective 3D input runs harmonic detection to make a 2D result and then selects "
                "the two localization snapshots automatically. Do not pass --image-fits or "
                "--difference-base-fits for 3D input."
            )
        if args.non_periodic:
            rows = analyze_3d_non_periodic_result(
                args,
                args.tile_size,
                4.0,
                args.r_orientation,
                local_source_sigma,
                args.min_gaussian_corr,
                args.tlisi_prominence_frac,
            )
        else:
            result_fits = run_cuda(args)
            rows = analyze_3d_result(
                args,
                result_fits,
                args.tile_size,
                args.sigma_threshold,
                args.r_orientation,
                local_source_sigma,
                args.min_gaussian_corr,
                args.tlisi_prominence_frac,
            )
    elif effective_naxis == 2:
        local_source_sigma = args.local_source_sigma if args.local_source_sigma is not None else 3.0
        result_fits = Path(args.iqa_fits)
        if not args.image_fits:
            raise RuntimeError(
                "Effective 2D input skips CUDA and requires --image-fits. "
                "Supply the localization image explicitly, e.g. t2-2, t1-1, or another x-x image."
            )
        image_fits = Path(args.image_fits)
        rows = analyze_result(
            result_fits,
            image_fits,
            difference_base_fits,
            args.tile_size,
            args.sigma_threshold,
            args.r_orientation,
            local_source_sigma,
        )
    else:
        raise RuntimeError(
            f"Expected effectively 2D or 3D FITS image, got effective NAXIS={effective_naxis} "
            f"from axis lengths {axis_lengths}"
        )

    print_rows(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
