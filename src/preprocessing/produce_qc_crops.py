import argparse
import tifffile
import dask.array as da
import os
import random
from multiprocessing.pool import ThreadPool


def _lazy_load(path):
    tif = tifffile.TiffFile(path)
    store = tif.series[0].levels[0].aszarr()
    return da.from_zarr(store), tif


def _load_channels(markers_path):
    with open(markers_path) as f:
        return [line.strip() for line in f if line.strip()]


def _get_grid_centers(image_height, image_width, size_y, size_x):
    n_rows = image_height // size_y
    n_cols = image_width // size_x
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            cy = size_y // 2 + row * size_y
            cx = size_x // 2 + col * size_x
            centers.append((cy, cx))
    return centers

def _write_single_crop(i, cy, cx, arr, axes, mask_arr, mask_axes, size_y, size_x, channel_names, output_dir):
    y0 = cy - size_y // 2
    y1 = cy + size_y // 2
    x0 = cx - size_x // 2
    x1 = cx + size_x // 2

    # crop image
    slices = [slice(None)] * len(axes)
    slices[axes.index('Y')] = slice(y0, y1)
    slices[axes.index('X')] = slice(x0, x1)
    crop = arr[tuple(slices)].compute()
    tifffile.imwrite(
        os.path.join(output_dir, f"crop_{i:04d}.ome.tiff"),
        crop,
        metadata={"Channel": {"Name": channel_names}},
        photometric="minisblack",
        compression="zlib",
    )

    # crop mask
    mask_slices = [slice(None)] * len(mask_axes)
    mask_slices[mask_axes.index('Y')] = slice(y0, y1)
    mask_slices[mask_axes.index('X')] = slice(x0, x1)
    mask_crop = mask_arr[tuple(mask_slices)].compute()
    tifffile.imwrite(
        os.path.join(output_dir, f"crop_{i:04d}_mask.tiff"),
        mask_crop,
        photometric="minisblack",
        compression="zlib",
    )


def crop_images(tiff_path, mask_path, markers_path, crop_size, n_crops, output_dir, n_workers):
    size_y, size_x = crop_size
    channel_names = _load_channels(markers_path)

    arr, tif = _lazy_load(tiff_path)
    axes = tif.series[0].axes
    image_height = arr.shape[axes.index('Y')]
    image_width = arr.shape[axes.index('X')]

    mask_arr, mask_tif = _lazy_load(mask_path)
    mask_axes = mask_tif.series[0].axes

    centers = _get_grid_centers(image_height, image_width, size_y, size_x)
    max_crops = len(centers)

    if n_crops > max_crops:
        print(f"Warning: requested {n_crops} crops but only {max_crops} non-overlapping crops fit. Using {max_crops}.")
        n_crops = max_crops

    selected = random.sample(centers, n_crops)
    n_workers = min(n_workers, n_crops)
    os.makedirs(output_dir, exist_ok=True)
    args_list = [
        (i, cy, cx, arr, axes, mask_arr, mask_axes, size_y, size_x, channel_names, output_dir) for i, (cy, cx) in enumerate(selected)
        ]
    with ThreadPool(processes=n_workers) as pool:
        pool.starmap(_write_single_crop, args_list)

    tif.close()
    mask_tif.close()

    print(f"Saved {n_crops} crops to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Produce QC crops from TIFF images and masks.")
    parser.add_argument("--tiff_path", required=True, help="Path to the input OME-TIFF image.")
    parser.add_argument("--mask_path", required=True, help="Path to the input mask TIFF.")
    parser.add_argument("--markers_path", required=True, help="Path to the markers.txt file.")
    parser.add_argument("--crop_size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"), help="Crop size as height width (default: 256 256).")
    parser.add_argument("--n_crops", type=int, required=True, help="Number of crops to produce.")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of worker threads for parallel processing (default: 4).")
    parser.add_argument("--output_dir", required=True, help="Directory to save crops.")
    args = parser.parse_args()

    crop_images(
        tiff_path=args.tiff_path,
        mask_path=args.mask_path,
        markers_path=args.markers_path,
        crop_size=tuple(args.crop_size),
        n_crops=args.n_crops,
        output_dir=args.output_dir,
        n_workers=args.n_workers
    )


if __name__ == "__main__":
    main()
