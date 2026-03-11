"""
Check shape of each OME-TIFF in a directory without loading pixel data into RAM.
Usage: python check_image_shapes.py <directory>
"""

import sys
from pathlib import Path
import tifffile


def check_shapes(directory: str):
    dirpath = Path(directory)
    tiffs = sorted(dirpath.glob("*.tiff")) + sorted(dirpath.glob("*.tif"))

    if not tiffs:
        print(f"No .tiff/.tif files found in {dirpath}")
        return

    print(f"{'File':<50} {'Shape':>20}")
    print("-" * 72)
    for path in tiffs:
        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                shape = series.shape
                axes = series.axes  # e.g. 'CYX', 'ZYX', 'CZYX'
            print(f"{path.name:<50} {str(shape):>20}  axes={axes}")
        except Exception as e:
            print(f"{path.name:<50} ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_image_shapes.py <directory>")
        sys.exit(1)
    check_shapes(sys.argv[1])
