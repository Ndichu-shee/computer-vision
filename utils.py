import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Function to display images
def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step 1: Initial Normalization (if needed)
def normalize_image(image):
    if image.max() > 1.0:
        image = image / 255.0
    return image

# Step 2: Apply Z-score Normalization
def zscore_normalize(image):
    reshaped_image = image.reshape(-1)
    normalized_image = zscore(reshaped_image, axis=0)
    return normalized_image.reshape(28, 28)

# Step 3: Detect and Remove Outliers (if any)
def detect_and_remove_outliers(image, threshold=3):
    is_outlier = np.abs(image) > threshold
    image[is_outlier] = np.median(image)
    return image

# Step 4: Histogram Equalization
def histogram_equalization(image):
    image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    equalized_image = cv2.equalizeHist(image_uint8)
    return equalized_image

# Step 5: Centering the Image
def center_image(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image  # No non-zero pixels, return the image as is
    
    y1, y2 = y_indices[[0, -1]]
    x1, x2 = x_indices[[0, -1]]
    digit = image[y1:y2 + 1, x1:x2 + 1]

    if digit.shape[0] > 28 or digit.shape[1] > 28:
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    centered_image = np.zeros((28, 28), dtype=np.uint8)
    center_y = (28 - digit.shape[0]) // 2
    center_x = (28 - digit.shape[1]) // 2
    centered_image[center_y:center_y + digit.shape[0], center_x:center_x + digit.shape[1]] = digit
    return centered_image

# Apply Gaussian filter for noise reduction
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Standardize image (for model input only)
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

# Rescale image to 0-255 range for display purposes
def rescale_for_display(image):
    min_val, max_val = image.min(), image.max()
    if max_val - min_val > 0:  # Avoid division by zero
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        image = np.zeros(image.shape, dtype=np.uint8)
    return image

# Preprocessing function that calls each step sequentially
def process_image(image, for_display=False):
    # Step 1: Normalize
    image = normalize_image(image)
    
    # Step 2: Z-score normalization
    image = zscore_normalize(image)
    
    # Step 3: Outlier detection and removal
    image = detect_and_remove_outliers(image)
    
    # Step 4: Histogram equalization
    image = histogram_equalization(image)
    
    # Step 5: Centering the image
    image = center_image(image)
    
    # Step 6: Gaussian filtering
    image = apply_gaussian_filter(image)

    # Step 7: Standardization (for model input)
    image = standardize_image(image)
    
    # Step 8: Rescale for display, if specified
    if for_display:
        image = rescale_for_display(image)
    
    return image