"""Hello world example to show some best coding and testing practices."""

from typing import Any

import cv2 as cv
import numpy as np
from loguru import logger

from src.params import FILE


def show_image(filename: str) -> None:
    """Display image thanks to filename."""
    logger.debug("🏁 Starting to show image")
    image = cv.imread(filename=filename)
    logger.info("🤔 to quite tap on : q")
    if image is None:
        logger.warning("Image is None")
        return
    cv.imshow("Main image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_matrix(matrix: np.ndarray) -> None:
    """Display image thanks to matrix."""
    logger.debug("🏁 Starting to show image")
    logger.info("🤔 to quite tap on : q")
    cv.imshow("Main image", matrix)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_numpy_matrix(filename: str):
    """Return matrix from filename."""
    logger.debug("🏁 Extracting array form the file")
    image: np.ndarray | None = cv.imread(filename=filename)
    if type(image) is np.ndarray:
        logger.debug("✅ The file has been properly extrcated into numpy array")
    else:
        logger.warning(f"The type of the matrix is {type(image)}")
    return image


def compute_gradients(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les gradients Ix et Iy de l'image en niveaux de gris.

    Args:
        matrix: image en couleur (H, W, 3)

    Returns:
        Ix, Iy: gradients horizontaux et verticaux
    """
    logger.debug("🏁 Computing image gradients")

    # Convertir en niveaux de gris
    gray = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY).astype(np.float32)

    # Calcul des gradients avec les opérateurs de Sobel
    Ix = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    Iy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

    logger.debug(f"Gradients computed: Ix shape {Ix.shape}, Iy shape {Iy.shape}")
    return Ix, Iy


def compute_structure_matrix(
    Ix: np.ndarray, Iy: np.ndarray, window_size: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule la matrice de structure M pour chaque pixel.

    M = [Ix² IxIy]
        [IxIy Iy²]

    Args:
        Ix: gradients de l'image
        Iy: gradients de l'image
        window_size: taille de la fenêtre pour la sommation

    Returns:
        M11, M12, M22: composantes de la matrice de structure
    """
    logger.debug("🏁 Computing structure matrix")

    # Calcul des produits des gradients
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # Création du noyau de convolution pour la sommation dans la fenêtre
    kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Sommation dans la fenêtre locale
    M11 = cv.filter2D(Ix2, -1, kernel)  # Σ(Ix²)
    M22 = cv.filter2D(Iy2, -1, kernel)  # Σ(Iy²)
    M12 = cv.filter2D(IxIy, -1, kernel)  # Σ(IxIy)

    logger.debug("Structure matrix computed")
    return M11, M12, M22


def harris_corner_response(
    M11: np.ndarray, M12: np.ndarray, M22: np.ndarray, k: float = 0.04
) -> np.ndarray:
    """Calcule la réponse Harris pour la détection de coins.

    Args:
        M11: composantes de la matrice de structure
        M12: composantes de la matrice de structure
        M22: composantes de la matrice de structure
        k: paramètre de Harris (typiquement 0.04-0.06)

    Returns:
        R: réponse Harris pour chaque pixel
    """
    logger.debug("🏁 Computing Harris corner response")

    # Déterminant et trace de la matrice M
    det_M = M11 * M22 - M12 * M12
    trace_M = M11 + M22

    # Réponse Harris: R = det(M) - k * trace(M)²
    R = det_M - k * (trace_M**2)

    logger.debug(f"Harris response computed, shape: {R.shape}")
    return R


def moravec_optimized_harris(
    matrix: np.ndarray, window_size: int = 3, threshold: float = 0.01
) -> list[tuple]:
    """Détection de coins optimisée avec la simplification de Harris.

    Cette méthode est beaucoup plus rapide que l'approche directionnelle de Moravec.

    Args:
        matrix: image en couleur
        window_size: taille de la fenêtre pour la matrice de structure
        threshold: seuil pour la détection des coins

    Returns:
        Liste des coins détectés avec leurs coordonnées et valeurs de réponse
    """
    logger.info("🏁 Starting optimized Moravec detection with Harris simplification")
    logger.info(f"📏 Input image shape: {matrix.shape}")

    # 1. Calculer les gradients
    Ix, Iy = compute_gradients(matrix)
    logger.info(f"📐 Gradients shape: Ix={Ix.shape}, Iy={Iy.shape}")

    # 2. Calculer la matrice de structure
    M11, M12, M22 = compute_structure_matrix(Ix, Iy, window_size)
    logger.info(f"🏗️ Structure matrix shape: M11={M11.shape}")

    # 3. Calculer la réponse Harris
    harris_response = harris_corner_response(M11, M12, M22)
    logger.info(f"📊 Harris response shape: {harris_response.shape}")
    logger.info(
        f"📊 Harris response range: min={np.min(harris_response):.6f}, max={np.max(harris_response):.6f}"
    )

    # 4. Seuiller et extraire les maxima locaux
    corners = extract_corners(harris_response, threshold)

    logger.info(f"Detected {len(corners)} corner candidates")

    # Vérifier la distribution des coins détectés
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


def simple_maximum_filter(image: np.ndarray, size: int) -> np.ndarray:
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
    response: np.ndarray, threshold: float, min_distance: int = 5
) -> list[tuple]:
    """Extrait les coins en trouvant les maxima locaux au-dessus du seuil.

    Args:
        response: matrice de réponse (Harris ou Moravec)
        threshold: seuil minimum pour considérer un point comme coin
        min_distance: distance minimale entre les coins

    Returns:
        Liste des coins [(x, y, score), ...]
    """
    logger.debug("🏁 Extracting corners from response matrix")

    # Normaliser la réponse
    response_norm = response.copy()
    min_val, max_val = np.min(response), np.max(response)
    if max_val > min_val:
        response_norm = (response - min_val) / (max_val - min_val)

    # Seuiller
    mask = response_norm > threshold

    local_maxima = (
        response_norm == simple_maximum_filter(response_norm, min_distance)
    ) & mask

    # Extraire les coordonnées et scores
    y_coords, x_coords = np.where(local_maxima)
    scores = response_norm[local_maxima]

    # Créer la liste des coins
    corners = [
        (int(x), int(y), float(score))
        for x, y, score in zip(x_coords, y_coords, scores, strict=False)
    ]

    # Trier par score décroissant
    corners.sort(key=lambda x: x[2], reverse=True)

    logger.debug(f"Extracted {len(corners)} corners")
    return corners


def update_matrix(matrix, corners, threshold: float = 0.1):
    """Marque les coins détectés sur la matrice en rouge.

    Args:
        matrix: image originale
        corners: liste des coins [(x, y, score), ...]
        threshold: seuil minimum pour marquer un coin
    """
    logger.debug(f"🎨 Marking {len(corners)} corners on image")
    result_matrix = matrix.copy()

    count = 0

    for corner in corners:
        if corner[2] < threshold or count > 300:
            continue
        x, y = corner[0], corner[1]  # x = colonne, y = ligne

        # Marquer un carré de 5x5 pixels autour du coin
        for di in range(-2, 3):  # de -2 à +2
            for dj in range(-2, 3):
                row = y + di  # y correspond à la ligne (axe vertical)
                col = x + dj  # x correspond à la colonne (axe horizontal)

                # Vérifier les limites
                if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]:
                    result_matrix[row, col] = [0, 0, 255]  # Rouge en BGR
        count += 1

    logger.debug(f"✅ Marked corners on image of shape {matrix.shape}")
    return result_matrix


def main() -> list[tuple[Any]]:
    """Détection de coins avec l'algorithme de Moravec optimisé avec Harris."""
    logger.info("🏁 Starting corner detection with optimized Moravec algorithm")

    matrix = get_numpy_matrix(FILE)

    # Version optimisée avec Harris
    logger.info("🚀 Using Harris optimization for faster computation")
    corners_harris = moravec_optimized_harris(matrix, window_size=3, threshold=0.01)

    # Afficher les meilleurs coins
    logger.info("🔍 Top 10 corner candidates (Harris optimization):")
    for i, (x, y, score) in enumerate(corners_harris[:50]):
        logger.info(f"Corner {i + 1}: position ({x}, {y}), Score: {score:.4f}")

    matrix = update_matrix(matrix, corners_harris)
    show_matrix(matrix)

    return corners_harris
