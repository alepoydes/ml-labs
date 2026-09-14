from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from coursekit.common import CourseError

SIZE = 128
COLUMNS = 32
PER_SHEET = COLUMNS**2


def image_location(index: int) -> tuple[str, int, int]:
    """PNG filename and one-based row/column for a zero-based image index."""
    sheet, tile = divmod(index, PER_SHEET)
    row, column = divmod(tile, COLUMNS)
    return f"images-{sheet:04d}.png", row + 1, column + 1


def write_images(directory: Path, images: np.ndarray) -> None:
    if images.dtype != np.uint8 or images.shape[1:] != (SIZE, SIZE, 3):
        raise CourseError("Images must have shape (N,128,128,3) and dtype uint8.")
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(images), PER_SHEET):
        batch = images[start : start + PER_SHEET]
        rows = (len(batch) + COLUMNS - 1) // COLUMNS
        tiles = np.full((rows * COLUMNS, SIZE, SIZE, 3), 255, np.uint8)
        tiles[: len(batch)] = batch
        sheet = tiles.reshape(rows, COLUMNS, SIZE, SIZE, 3).transpose(0, 2, 1, 3, 4)
        path = directory / f"images-{start // PER_SHEET:04d}.png"
        Image.fromarray(sheet.reshape(rows * SIZE, COLUMNS * SIZE, 3)).save(path)
        with Image.open(path) as stored:
            decoded = np.array(stored).reshape(rows, SIZE, COLUMNS, SIZE, 3)
        decoded = decoded.transpose(0, 2, 1, 3, 4).reshape(-1, SIZE, SIZE, 3)
        if not np.array_equal(decoded[: len(batch)], batch):
            raise CourseError(f"PNG round trip changed pixels: {path}")


class ImageSet:
    """At most two decoded sheets per split; indices never include padding."""

    def __init__(self, directory: Path, count: int):
        self.directory, self.count = directory, count
        self._sheet = lru_cache(maxsize=2)(self._read_sheet)
        expected = {f"images-{i:04d}.png" for i in range((count + 1023) // 1024)}
        if {p.name for p in directory.glob("images-*.png")} != expected:
            raise CourseError(f"Missing or unexpected image sheets: {directory}")

    def _read_sheet(self, number: int) -> np.ndarray:
        count = min(PER_SHEET, self.count - number * PER_SHEET)
        rows = (count + COLUMNS - 1) // COLUMNS
        with Image.open(self.directory / f"images-{number:04d}.png") as image:
            if image.mode != "RGB" or image.size != (COLUMNS * SIZE, rows * SIZE):
                raise CourseError("Invalid RGB sheet dimensions.")
            pixels = np.array(image).reshape(rows, SIZE, COLUMNS, SIZE, 3)
        tiles = pixels.transpose(0, 2, 1, 3, 4).reshape(-1, SIZE, SIZE, 3)
        if np.any(tiles[count:] != 255):
            raise CourseError("Unused image cells must be white.")
        return tiles[:count]

    def __len__(self):
        return self.count

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        if (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= self.count)
        ):
            raise CourseError("Image index is outside the dataset.")
        result = [
            self._sheet(int(i) // PER_SHEET)[int(i) % PER_SHEET] for i in indices.flat
        ]
        return np.asarray(result, dtype=np.uint8).reshape(*indices.shape, SIZE, SIZE, 3)
