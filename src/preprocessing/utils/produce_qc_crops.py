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


def _count_cells_in_window(mask_2d, cy, cx, side):
    h, w = mask_2d.shape
    half = side // 2
    window = mask_2d[max(0, cy - half):min(h, cy + half),
                     max(0, cx - half):min(w, cx + half)]
    return int(np.count_nonzero(np.unique(window)))


def _binary_search_crop_size(mask_2d, cy, cx, target_cells, max_iter=25):
    """Find the smallest square side length centred on (cy, cx) that contains
    at least target_cells labelled cells, via binary search on the mask."""
    h, w = mask_2d.shape
    lo, hi = 1, min(h, w)

    # check whether even the maximum window has enough cells
    if _count_cells_in_window(mask_2d, cy, cx, hi) < target_cells:
        print(f"  Warning: fewer than {target_cells} cells reachable from ({cy},{cx}); "
              f"using maximum crop side ({hi}px).")
        return hi

    for _ in range(max_iter):
        mid = (lo + hi) // 2
        if _count_cells_in_window(mask_2d, cy, cx, mid) < target_cells:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1:
            break

    return hi


def _compute_crops_from_mask(mask_arr, mask_axes, target_cells, n_crops):
    """Sample on-tissue centers and find the exact crop size per center via binary
    search on the mask.

    Returns a list of (cy, cx, size_y, size_x) tuples with centers guaranteed to keep
    the crop window inside the image.
    """
    mask_2d = _get_2d_mask(mask_arr, mask_axes)
    h, w = mask_2d.shape

    tissue_coords = np.argwhere(mask_2d > 0)
    if len(tissue_coords) == 0:
        raise ValueError("Mask contains no labelled cells.")

    if n_crops > len(tissue_coords):
        print(
            f"Warning: requested {n_crops} crops but only {len(tissue_coords)} unique "
            f"tissue positions are available — some crops will overlap."
        )

    chosen_idx = np.random.choice(len(tissue_coords), size=n_crops,
                                  replace=n_crops > len(tissue_coords))
    chosen = tissue_coords[chosen_idx]

    results = []
    for cy, cx in chosen:
        cy, cx = int(cy), int(cx)

        # binary search needs a fixed center — use a preliminary size estimate to
        # clamp first, then search from the clamped position
        # initial clamp uses a conservative half=1 so any tissue pixel is valid
        # the real clamp happens after we know the side
        side = _binary_search_crop_size(mask_2d, cy, cx, target_cells)

        # clamp center so the derived crop fits inside the image, then re-verify
        half_side = side // 2
        cy = int(np.clip(cy, half_side, h - half_side))
        cx = int(np.clip(cx, half_side, w - half_side))

        # re-run search from clamped position (cheap — usually converges in 1-2 iters
        # since the clamped center is typically only a few pixels away)
        side = _binary_search_crop_size(mask_2d, cy, cx, target_cells)

        n_actual = _count_cells_in_window(mask_2d, cy, cx, side)
        results.append((cy, cx, side, side))
        print(f"  Center ({cy},{cx}): crop side {side}px contains {n_actual} cells "
              f"(target {target_cells})")

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
                output_dir, n_workers, target_cells=None):
    channel_names = _load_channels(markers_path)

    arr, tif = _lazy_load(tiff_path)
    axes = tif.series[0].axes
    image_height = arr.shape[axes.index('Y')]
    image_width = arr.shape[axes.index('X')]

    mask_arr, mask_tif = _lazy_load(mask_path)
    mask_axes = mask_tif.series[0].axes

    if target_cells is not None:
        crops = _compute_crops_from_mask(mask_arr, mask_axes, target_cells, n_crops)
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
    )


if __name__ == "__main__":
    main()
