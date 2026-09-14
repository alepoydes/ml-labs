import numpy as np
from PIL import Image


def pixels(images, resolution):
    small = [
        np.asarray(
            Image.fromarray(image).resize(
                (resolution, resolution), Image.Resampling.BILINEAR
            )
        )
        for image in images
    ]
    # White padding becomes zero, so it does not dominate the first affine layer.
    return np.asarray(small, dtype=np.float32).reshape(len(images), -1) / 255 - 1
