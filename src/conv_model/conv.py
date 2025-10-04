"""Module that convert RGb to HSV and change colors."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from loguru import logger
from PIL import Image

from src.models.color_models import RGB


def get_image(filename: str) -> npt.NDArray[np.uint8]:
    """Extract the numpy array from the main image."""
    logger.debug(f"🏁 Starting the extract the file: {filename} into a numpy matrix")
    matrix: npt.NDArray[np.uint8] = np.asarray(Image.open(fp=filename))
    return matrix


def conv_sub_matrix_rgb_to_hsv(
    matrix: npt.NDArray[np.uint8],
    mat_hsv: npt.NDArray[np.uint8],
    line: int,
    col: int,
    size: int = 3,
) -> None:
    """Conversion for sub_matrix."""
    number_line, number_col, _ = matrix.shape
    for curr_l in range(size):
        for curr_c in range(size):
            pos_l = line + curr_l
            pos_c = col + curr_c
            if pos_l >= number_line:
                continue
            if pos_c >= number_col:
                continue
            R, G, B = matrix[pos_l, pos_c]
            rgb_value = RGB(R, G, B)
            hsv_val = rgb_value.to_HSV()
            mat_hsv[pos_l, pos_c] = [hsv_val.H, hsv_val.S, hsv_val.V]


def conv_rgb_to_hsv_threads(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert with optimizations."""
    # Normalisation [0, 1]
    new_mat = matrix.astype(np.float32) / 255.0
    r, g, b = new_mat[..., 0], new_mat[..., 1], new_mat[..., 2]

    maxc = np.max(new_mat, axis=-1)
    minc = np.min(new_mat, axis=-1)
    delta = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    mask = delta != 0

    # Pour éviter les divisions par zéro, on divise uniquement là où delta != 0
    np.divide((g - b), delta, out=h, where=(mask & (maxc == r)))
    h[mask & (maxc == r)] = (60 * h[mask & (maxc == r)]) % 360

    np.divide((b - r), delta, out=h, where=(mask & (maxc == g)))
    h[mask & (maxc == g)] = 60 * h[mask & (maxc == g)] + 120

    np.divide((r - g), delta, out=h, where=(mask & (maxc == b)))
    h[mask & (maxc == b)] = 60 * h[mask & (maxc == b)] + 240

    s = np.zeros_like(maxc)
    np.divide(delta, maxc, out=s, where=maxc != 0)

    v = maxc

    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def conv_hsv_to_rgb(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert HSV matrix to RGB ones."""
    new_mat = matrix.astype(np.float32)
    h, s, v = new_mat[..., 0], new_mat[..., 1], new_mat[..., 2]

    C = v * s
    X = C * (1 - np.abs(((h / 60) % 2) - 1))
    m = v - C

    R_prime = np.zeros_like(h)
    G_prime = np.zeros_like(h)
    B_prime = np.zeros_like(h)

    mask = (h >= 0) & (h < 60)
    R_prime[mask], G_prime[mask], B_prime[mask] = C[mask], X[mask], 0

    mask = (h >= 60) & (h < 120)
    R_prime[mask], G_prime[mask], B_prime[mask] = X[mask], C[mask], 0

    mask = (h >= 120) & (h < 180)
    R_prime[mask], G_prime[mask], B_prime[mask] = 0, C[mask], X[mask]

    mask = (h >= 180) & (h < 240)
    R_prime[mask], G_prime[mask], B_prime[mask] = 0, X[mask], C[mask]

    mask = (h >= 240) & (h < 300)
    R_prime[mask], G_prime[mask], B_prime[mask] = X[mask], 0, C[mask]

    mask = (h >= 300) & (h < 360)
    R_prime[mask], G_prime[mask], B_prime[mask] = C[mask], 0, X[mask]

    R = np.clip((R_prime + m) * 255, 0, 255).astype(np.uint8)
    G = np.clip((G_prime + m) * 255, 0, 255).astype(np.uint8)
    B = np.clip((B_prime + m) * 255, 0, 255).astype(np.uint8)

    rgb = np.stack([R, G, B], axis=-1)
    return rgb


def conv_image_rgb_to_hsv(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert image from rgb to hsv."""
    shape = matrix.shape
    mat_hsv = np.zeros(shape, dtype=np.float32)
    number_line, number_col, _ = shape
    for line in range(number_line):
        for col in range(number_col):
            R, G, B = matrix[line, col]
            rgb_value = RGB(R, G, B)
            hsv_val = rgb_value.to_HSV()
            mat_hsv[line, col] = [hsv_val.H, hsv_val.S, hsv_val.V]
    return mat_hsv


def create_mask(
    mat: npt.NDArray[np.float32], init_values: list[int], end_values: list[int]
) -> npt.NDArray[np.uint8] | None:
    """Create a mask according to a list of angle."""
    logger.debug("🏁 Starting to create a mask for the matrix")

    if len(init_values) != len(end_values):
        logger.error(
            f"Not the same size between init values and end values. "
            f"Init: {len(init_values)}, End: {len(end_values)}"
        )
        return None  # safer than returning `mat`

    # Start with all True
    mask = np.ones_like(mat, dtype=bool) * False

    for i in range(len(init_values)):
        mask = (mat >= init_values[i]) & (mat <= end_values[i]) | mask

    return mask


def erase_values(
    matrix: npt.NDArray[np.float32], init_values: list[int], end_values: list[int]
) -> npt.NDArray[np.float32]:
    """Erase values on the HSV representation."""
    matrix = matrix.copy()
    h, _, v = matrix[..., 0], matrix[..., 1], matrix[..., 2]

    mask = create_mask(h, init_values, end_values)
    v[mask] = 0
    return matrix


def display_matrix(matrix: npt.NDArray[np.uint8]) -> None:
    """Display the RGB matrix."""
    plt.imshow(matrix)
    plt.show()


def main() -> None:
    """Main function that launch the intersting functions."""
    logger.info("hello")
    file = "data/test.jpg"
    init_mat = get_image(file)
    hsv_image = conv_rgb_to_hsv_threads(init_mat)
    hsv_image = erase_values(hsv_image, [20], [100])
    rgb_image = conv_hsv_to_rgb(hsv_image)
    display_matrix(rgb_image)


if __name__ == "__main__":
    main()
