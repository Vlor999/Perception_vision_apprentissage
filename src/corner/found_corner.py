"""Hello world example to show some best coding and testing practices."""

import typing

import cv2 as cv
import numpy as np
import numpy.typing as npt
from loguru import logger

from src.params import FILE


def show_image(filename: str) -> None:
    """Display image thanks to filename."""
    logger.debug("🏁 Starting to show image")
    image = cv.imread(filename)
    logger.info("🤔 to quite tap on : q")
    if image is None:
        logger.warning("Image is None")
        return
    cv.imshow("Main image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_matrix(matrix: npt.NDArray[np.uint8]) -> None:
    """Display image thanks to matrix."""
    logger.debug("🏁 Starting to show image")
    logger.info("🤔 to quite tap on : q")
    cv.imshow("Main image", matrix)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_numpy_matrix(filename: str) -> npt.NDArray[np.uint8]:
    """Return matrix from filename."""
    logger.debug("🏁 Extracting array form the file")
    image = cv.imread(filename)
    if image is None:
        raise ValueError("Image could not be loaded.")
    return image.astype(np.uint8)


def compute_gradients(
    matrix: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calcule les gradients Ix et Iy de l'image en niveaux de gris."""
    logger.debug("🏁 Computing image gradients")
    gray = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY).astype(np.float32)
    Ix = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3).astype(np.float32)
    Iy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3).astype(np.float32)
    logger.debug(f"Gradients computed: Ix shape {Ix.shape}, Iy shape {Iy.shape}")
    return Ix, Iy


def compute_structure_matrix(
    Ix: npt.NDArray[np.float32], Iy: npt.NDArray[np.float32], window_size: int = 3
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calcule la matrice de structure M pour chaque pixel."""
    logger.debug("🏁 Computing structure matrix")
    Ix2 = (Ix * Ix).astype(np.float32)
    Iy2 = (Iy * Iy).astype(np.float32)
    IxIy = (Ix * Iy).astype(np.float32)
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    M11 = cv.filter2D(Ix2, -1, kernel).astype(np.float32)
    M22 = cv.filter2D(Iy2, -1, kernel).astype(np.float32)
    M12 = cv.filter2D(IxIy, -1, kernel).astype(np.float32)
    logger.debug("Structure matrix computed")
    return M11, M12, M22


@typing.no_type_check
def harris_corner_response(
    M11: npt.NDArray[np.float32],
    M12: npt.NDArray[np.float32],
    M22: npt.NDArray[np.float32],
    k: float = 0.04,
) -> npt.NDArray[np.float32]:
    """Calcule la réponse Harris pour la détection de coins."""
    logger.debug("🏁 Computing Harris corner response")
    det_M = M11 * M22 - M12 * M12
    trace_M = M11 + M22
    R = det_M - k * (trace_M**2)
    logger.debug(f"Harris response computed, shape: {R.shape}")
    return R.astype(np.float32)


def moravec_optimized_harris(
    matrix: npt.NDArray[np.uint8], window_size: int = 3, threshold: float = 0.01
) -> list[tuple[int, int, float]]:
    """Détection de coins optimisée avec la simplification de Harris."""
    logger.info("🏁 Starting optimized Moravec detection with Harris simplification")
    logger.info(f"📏 Input image shape: {matrix.shape}")
    Ix, Iy = compute_gradients(matrix)
    logger.info(f"📐 Gradients shape: Ix={Ix.shape}, Iy={Iy.shape}")
    M11, M12, M22 = compute_structure_matrix(Ix, Iy, window_size)
    logger.info(f"🏗️ Structure matrix shape: M11={M11.shape}")
    harris_response = harris_corner_response(M11, M12, M22)
    logger.info(f"📊 Harris response shape: {harris_response.shape}")
    logger.info(
        f"📊 Harris response range: min={np.min(harris_response):.6f}, max={np.max(harris_response):.6f}"
    )
    corners = extract_corners(harris_response, threshold)
    logger.info(f"Detected {len(corners)} corner candidates")
    if corners:
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        logger.info(
            f"📍 Corner distribution - X: [{min(x_coords)}-{max(x_coords)}], Y: [{min(y_coords)}-{max(y_coords)}]"
        )
        logger.info(
            f"📍 Image dimensions: width={matrix.shape[1]}, height={matrix.shape[0]}"
        )
    return corners


def simple_maximum_filter(
    image: npt.NDArray[np.float32], size: int
) -> npt.NDArray[np.float32]:
    """Alternative simple au maximum_filter de scipy."""
    h, w = image.shape
    result = np.zeros_like(image)
    half_size = size // 2
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - half_size)
            i_max = min(h, i + half_size + 1)
            j_min = max(0, j - half_size)
            j_max = min(w, j + half_size + 1)
            result[i, j] = np.max(image[i_min:i_max, j_min:j_max])
    return result


def extract_corners(
    response: npt.NDArray[np.float32], threshold: float, min_distance: int = 5
) -> list[tuple[int, int, float]]:
    """Extrait les coins en trouvant les maxima locaux au-dessus du seuil."""
    logger.debug("🏁 Extracting corners from response matrix")
    response_norm = response.copy()
    min_val, max_val = np.min(response), np.max(response)
    if max_val > min_val:
        response_norm = (response - min_val) / (max_val - min_val)
    mask = response_norm > threshold
    local_maxima = (
        response_norm == simple_maximum_filter(response_norm, min_distance)
    ) & mask
    y_coords, x_coords = np.where(local_maxima)
    scores = response_norm[local_maxima]
    corners: list[tuple[int, int, float]] = [
        (int(x), int(y), float(score))
        for x, y, score in zip(x_coords, y_coords, scores, strict=False)
    ]
    corners.sort(key=lambda x: x[2], reverse=True)
    logger.debug(f"Extracted {len(corners)} corners")
    return corners


def update_matrix(
    matrix: npt.NDArray[np.uint8],
    corners: list[tuple[int, int, float]],
    threshold: float = 0.1,
) -> npt.NDArray[np.uint8]:
    """Marque les coins détectés sur la matrice en rouge."""
    logger.debug(f"🎨 Marking {len(corners)} corners on image")
    result_matrix = matrix.copy()
    count = 0
    for corner in corners:
        if corner[2] < threshold or count > 300:
            continue
        x, y = corner[0], corner[1]
        for di in range(-2, 3):
            for dj in range(-2, 3):
                row = y + di
                col = x + dj
                if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]:
                    result_matrix[row, col] = [0, 0, 255]
        count += 1
    logger.debug(f"✅ Marked corners on image of shape {matrix.shape}")
    return result_matrix


def main() -> list[tuple[int, int, float]]:
    """Détection de coins avec l'algorithme de Moravec optimisé avec Harris."""
    logger.info("🏁 Starting corner detection with optimized Moravec algorithm")
    matrix = get_numpy_matrix(FILE)
    logger.info("🚀 Using Harris optimization for faster computation")
    corners_harris = moravec_optimized_harris(matrix, window_size=3, threshold=0.01)
    logger.info("🔍 Top 10 corner candidates (Harris optimization):")
    for i, (x, y, score) in enumerate(corners_harris[:50]):
        logger.info(f"Corner {i + 1}: position ({x}, {y}), Score: {score:.4f}")
    matrix = update_matrix(matrix, corners_harris)
    show_matrix(matrix)
    return corners_harris
