import argparse
import os
import random
from multiprocessing.pool import ThreadPool
import dask.array as da
import numpy as np
import tifffile


def _lazy_load(path):
    tif = tifffile.TiffFile(path)
    store = tif.series[0].levels[0].aszarr()
    return da.from_zarr(store), tif


def _load_channels(markers_path):
    with open(markers_path) as f:
        return [line.strip() for line in f if line.strip()]


def _get_2d_mask(mask_arr, mask_axes):
    mask = mask_arr.compute()
    if mask.ndim == 2:
        return mask
    slices = [0] * mask.ndim
    slices[mask_axes.index('Y')] = slice(None)
    slices[mask_axes.index('X')] = slice(None)
    return mask[tuple(slices)]


def _compute_crops_from_mask(mask_arr, mask_axes, target_cells, n_crops, probe_size=256):
    """Sample on-tissue centers and derive a local crop size per center via a probe window.

    Returns a list of (cy, cx, size_y, size_x) tuples with centers guaranteed to keep
    the crop window inside the image.
    """
    mask_2d = _get_2d_mask(mask_arr, mask_axes)
    h, w = mask_2d.shape
    half_probe = probe_size // 2

    # direct lookup of all tissue pixel coordinates — no scanning needed
    tissue_coords = np.argwhere(mask_2d > 0)
    if len(tissue_coords) == 0:
        raise ValueError("Mask contains no labelled cells.")

    # prefer positions where the full probe fits inside the image
    valid = tissue_coords[
        (tissue_coords[:, 0] >= half_probe) & (tissue_coords[:, 0] < h - half_probe) &
        (tissue_coords[:, 1] >= half_probe) & (tissue_coords[:, 1] < w - half_probe)
    ]
    if len(valid) == 0:
        print("Warning: image is smaller than probe_size in at least one dimension; "
              "probe windows will be clipped to image bounds.")
        valid = tissue_coords

    if n_crops > len(valid):
        print(
            f"Warning: requested {n_crops} crops but only {len(valid)} unique tissue "
            f"positions are available — some crops will overlap."
        )

    chosen_idx = np.random.choice(len(valid), size=n_crops, replace=n_crops > len(valid))
    chosen = valid[chosen_idx]

    results = []
    for cy, cx in chosen:
        cy, cx = int(cy), int(cx)

        # clamp probe to image bounds (guards against border tissue pixels in fallback)
        probe = mask_2d[
            max(0, cy - half_probe):min(h, cy + half_probe),
            max(0, cx - half_probe):min(w, cx + half_probe),
        ]
        n_local = int(np.sum(np.unique(probe) != 0))

        if n_local == 0:
            print(f"  Warning: probe at ({cy},{cx}) contains no cells; "
                  f"defaulting crop side to probe_size ({probe_size}px).")
            side = probe_size
        else:
            local_density = n_local / probe.size  # cells per pixel within probe
            side = int(np.sqrt(target_cells / local_density))

        if side > min(h, w):
            print(f"  Warning: derived crop side {side}px exceeds image bounds "
                  f"({h}×{w}); clamping.")
            side = min(h, w)

        # shift center if needed so the crop window stays fully inside the image
        half_side = side // 2
        cy = int(np.clip(cy, half_side, h - half_side))
        cx = int(np.clip(cx, half_side, w - half_side))

        results.append((cy, cx, side, side))
        print(f"  Center ({cy},{cx}): {n_local} cells in probe → crop side {side}px "
              f"for ~{target_cells} cells")

    return results


def _get_grid_centers(image_height, image_width, size_y, size_x):
    centers = []
    for row in range(image_height // size_y):
        for col in range(image_width // size_x):
            cy = size_y // 2 + row * size_y
            cx = size_x // 2 + col * size_x
            centers.append((cy, cx))
    return centers


def _write_single_crop(i, cy, cx, arr, axes, mask_arr, mask_axes,
                       size_y, size_x, channel_names, output_dir, image_name):
    y0, y1 = cy - size_y // 2, cy + size_y // 2
    x0, x1 = cx - size_x // 2, cx + size_x // 2

    img_slices = [slice(None)] * len(axes)
    img_slices[axes.index('Y')] = slice(y0, y1)
    img_slices[axes.index('X')] = slice(x0, x1)
    tifffile.imwrite(
        os.path.join(output_dir, f"crop_{image_name}_{i:04d}.ome.tiff"),
        arr[tuple(img_slices)].compute(),
        metadata={"Channel": {"Name": channel_names}},
        photometric="minisblack",
        compression="zlib",
    )

    mask_slices = [slice(None)] * len(mask_axes)
    mask_slices[mask_axes.index('Y')] = slice(y0, y1)
    mask_slices[mask_axes.index('X')] = slice(x0, x1)
    tifffile.imwrite(
        os.path.join(output_dir, f"crop_{image_name}_{i:04d}_mask.tiff"),
        mask_arr[tuple(mask_slices)].compute(),
        photometric="minisblack",
        compression="zlib",
    )


def crop_images(tiff_path, mask_path, markers_path, crop_size, n_crops,
                output_dir, n_workers, target_cells=None, probe_size=256):
    channel_names = _load_channels(markers_path)

    arr, tif = _lazy_load(tiff_path)
    axes = tif.series[0].axes
    image_height = arr.shape[axes.index('Y')]
    image_width = arr.shape[axes.index('X')]

    mask_arr, mask_tif = _lazy_load(mask_path)
    mask_axes = mask_tif.series[0].axes

    if target_cells is not None:
        crops = _compute_crops_from_mask(
            mask_arr, mask_axes, target_cells, n_crops, probe_size=probe_size
        )
    else:
        size_y, size_x = crop_size
        centers = _get_grid_centers(image_height, image_width, size_y, size_x)
        if not centers:
            raise ValueError(
                f"Crop size ({size_y}×{size_x}) is larger than the image "
                f"({image_height}×{image_width}); no crops fit."
            )
        if n_crops > len(centers):
            print(f"Warning: requested {n_crops} crops but only {len(centers)} "
                  f"non-overlapping crops fit. Using {len(centers)}.")
            n_crops = len(centers)
        crops = [(cy, cx, size_y, size_x) for cy, cx in random.sample(centers, n_crops)]

    n_crops = len(crops)
    image_name = os.path.basename(tiff_path).replace(".ome.tiff", "").replace(".tiff", "")
    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (i, cy, cx, arr, axes, mask_arr, mask_axes, sy, sx, channel_names, output_dir, image_name)
        for i, (cy, cx, sy, sx) in enumerate(crops)
    ]
    with ThreadPool(processes=min(n_workers, n_crops)) as pool:
        pool.starmap(_write_single_crop, args_list)

    tif.close()
    mask_tif.close()
    print(f"Saved {n_crops} crops to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Produce QC crops from TIFF images and masks.")
    parser.add_argument("--tiff_path", required=True,
                        help="Path to the input OME-TIFF image.")
    parser.add_argument("--mask_path", required=True,
                        help="Path to the input mask TIFF.")
    parser.add_argument("--markers_path", required=True,
                        help="Path to the markers.txt file.")
    parser.add_argument("--crop_size", nargs=2, type=int, default=[256, 256],
                        metavar=("H", "W"),
                        help="Crop size as height width (default: 256 256). "
                             "Ignored when --target_cells is set.")
    parser.add_argument("--target_cells", type=int, default=None,
                        help="Target number of cells per crop. Crop size is derived "
                             "from local mask density around each sampled center. "
                             "Mutually exclusive with --crop_size.")
    parser.add_argument("--probe_size", type=int, default=256,
                        help="Side length (px) of the square probe window used to "
                             "estimate local cell density when --target_cells is set "
                             "(default: 256).")
    parser.add_argument("--n_crops", type=int, required=True,
                        help="Number of crops to produce.")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of worker threads for parallel I/O (default: 4).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save crops.")
    args = parser.parse_args()

    crop_images(
        tiff_path=args.tiff_path,
        mask_path=args.mask_path,
        markers_path=args.markers_path,
        crop_size=tuple(args.crop_size),
        n_crops=args.n_crops,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        target_cells=args.target_cells,
        probe_size=args.probe_size,
    )


if __name__ == "__main__":
    main()
