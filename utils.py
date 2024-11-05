import numpy as np
import cv2
from scipy.stats import zscore


def check_dimensional_consistency(image, expected_shape=(28, 28)):
    return image.shape == expected_shape


def apply_gaussian_filter(image):
    image = (
        (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    )
    return cv2.GaussianBlur(image.reshape(28, 28), (5, 5), 0)


def apply_median_filter(image):
    image = (
        (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    )
    return cv2.medianBlur(image.reshape(28, 28), 5)


def improved_binarization(image):
    image = (
        (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    )
    return cv2.adaptiveThreshold(
        image.reshape(28, 28),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    standardized_image = (image - mean) / std
    return standardized_image


# Full preprocessing pipeline
def pre_process_image(image):
    if not check_dimensional_consistency(image):
        raise ValueError("Image dimensions are incorrect")

    # Step 1: Standardization (skip Z-score normalization to avoid redundancy)
    image = standardize_image(image)

    # Step 2: Noise reduction (choose one filter)
    image = apply_gaussian_filter(image)  # Optionally use apply_median_filter(image)

    # Step 3: Binarization
    image = improved_binarization(image)

    # Ensure final shape for model compatibility (1 channel for CNNs)
    return image.reshape(28, 28, 1)
