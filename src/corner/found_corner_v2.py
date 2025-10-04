"""Hello world example to show some best coding and testing practices."""

import typing

import cv2 as cv
import numpy as np
import numpy.typing as npt
from loguru import logger

from src.params import FILE


def show_image(filename: str) -> None:
    """Display the current image."""
    logger.debug("🏁 Starting to show image")
    image = cv.imread(filename=filename)
    if image is None:
        logger.warning("Image is None")
        return
    cv.imshow("Main image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_matrix(matrix: npt.NDArray[np.uint8]) -> None:
    """Display the current matrix."""
    logger.debug("🏁 Starting to show image")
    cv.imshow("Main image", matrix)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_numpy_matrix(filename: str) -> npt.NDArray[np.uint8] | None:
    """Get the current matrix from the file."""
    logger.debug("🏁 Extracting array from the file")
    image = cv.imread(filename=filename)
    if image is None:
        logger.warning("Image could not be loaded")
        return None
    return image.astype(np.uint8)


def update_matrix(
    matrix: npt.NDArray[np.uint8],
    corners: list[tuple[int, int, float]],
    threshold: float = 0.1,
    display_points: int = 300,
) -> tuple[npt.NDArray[np.uint8], list[tuple[int, int]]]:
    """Update the current matrix wth the cloud points."""
    result_matrix = matrix.copy()
    points: list[tuple[int, int]] = []
    count = 0
    for corner in corners:
        if corner[2] < threshold or count > display_points:
            continue
        x, y = corner[0], corner[1]
        for di in range(-2, 3):
            for dj in range(-2, 3):
                row = y + di
                col = x + dj
                if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]:
                    result_matrix[row, col] = [0, 0, 255]
                    points.append((row, col))
        count += 1
    return result_matrix, points


def get_gradients(
    matrix: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Get the graidents from the image."""
    gray = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY).astype(np.float32)
    Ix = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3).astype(np.float32)
    Iy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3).astype(np.float32)
    return Ix, Iy


@typing.no_type_check
def compute_structure_matrix(
    Ix: npt.NDArray[np.float32], Iy: npt.NDArray[np.float32], window_size: int = 3
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Apply filter to the current matrix."""
    Ix2: npt.NDArray[np.float32] = (Ix * Ix).astype(np.float32)
    Iy2: npt.NDArray[np.float32] = (Iy * Iy).astype(np.float32)
    IxIy: npt.NDArray[np.float32] = (Ix * Iy).astype(np.float32)

    kernel: npt.NDArray[np.float32] = np.ones(
        (window_size, window_size), dtype=np.float32
    )

    M11: npt.NDArray[np.float32] = cv.filter2D(Ix2, -1, kernel).astype(np.float32)
    M22: npt.NDArray[np.float32] = cv.filter2D(Iy2, -1, kernel).astype(np.float32)
    M12: npt.NDArray[np.float32] = cv.filter2D(IxIy, -1, kernel).astype(np.float32)

    return M11, M12, M22


def harris_compute(
    M11: npt.NDArray[np.float32],
    M22: npt.NDArray[np.float32],
    M12: npt.NDArray[np.float32],
    alpha: float = 0.04,
) -> npt.NDArray[np.float32]:
    """Compute the harris method."""
    det_m: npt.NDArray[np.float32] = (M11 * M22 - M12**2).astype(np.float32)
    trace_m: npt.NDArray[np.float32] = (M11 + M22).astype(np.float32)
    R: npt.NDArray[np.float32] = (det_m - alpha * (trace_m**2)).astype(np.float32)
    return R


def simple_maximum_filter(
    matrix: npt.NDArray[np.float32], distance: int
) -> npt.NDArray[np.float32]:
    """Found the local maximum."""
    h, w = matrix.shape
    result = np.zeros_like(matrix)
    half_size = distance // 2
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - half_size)
            i_max = min(h, i + half_size + 1)
            j_min = max(0, j - half_size)
            j_max = min(w, j + half_size + 1)
            result[i, j] = np.max(matrix[i_min:i_max, j_min:j_max])
    return result


def get_harris_values_filtered(
    matrix: npt.NDArray[np.float32], threshold: float, min_distance: int = 5
) -> list[tuple[int, int, float]]:
    """Filter the matrix using masks."""
    response_mat = matrix.copy()
    mini, maxi = np.min(matrix), np.max(matrix)
    if maxi > mini:
        response_mat = (matrix - mini) / (maxi - mini)
    mask = response_mat > threshold
    local_max = mask & (
        response_mat == simple_maximum_filter(response_mat, min_distance)
    )
    y_mor, x_mor = np.where(local_max)
    filtered_matrix = response_mat[local_max]
    corners: list[tuple[int, int, float]] = [
        (int(x), int(y), float(score))
        for x, y, score in zip(x_mor, y_mor, filtered_matrix, strict=False)
    ]
    corners.sort(key=lambda x: x[2], reverse=True)
    return corners


def moravec_optimized_harris(
    matrix: npt.NDArray[np.uint8], window_size: int = 3, threshold: float = 0.01
) -> list[tuple[int, int, float]]:
    """Moravec calculus optimization."""
    Ix, Iy = get_gradients(matrix)
    M11, M12, M22 = compute_structure_matrix(Ix, Iy, window_size)
    harris_values = harris_compute(M11, M22, M12)
    corners = get_harris_values_filtered(harris_values, threshold)
    return corners


def main() -> list[tuple[int, int, float]]:
    """Main function that launch structured funcitons."""
    matrix = get_numpy_matrix(FILE)
    if matrix is None:
        return []
    corners_harris = moravec_optimized_harris(matrix, window_size=3, threshold=0.01)
    matrix, _ = update_matrix(matrix, corners_harris)
    show_matrix(matrix)
    return corners_harris
