import cv2
import numpy as np

img = cv2.imread("../datasets/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Binary thresholding ---
# Pixels >= 127 become white (255), else black (0)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --- Inverse thresholding ---
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# --- Adaptive thresholding ---
# Threshold computed per-pixel based on the mean of a local neighborhood (blockSize x blockSize)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
)

# --- Otsu's method ---
# cv2.THRESH_OTSU automatically computes the optimal threshold value
otsu_thresh_value, otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"Otsu automatically chose threshold: {otsu_thresh_value}")

cv2.imwrite("../outputs/thresh_binary.jpg", binary)
cv2.imwrite("../outputs/thresh_binary_inv.jpg", binary_inv)
cv2.imwrite("../outputs/thresh_adaptive.jpg", adaptive)
cv2.imwrite("../outputs/thresh_otsu.jpg", otsu)

print("Saved thresh_binary.jpg, thresh_binary_inv.jpg, thresh_adaptive.jpg, thresh_otsu.jpg to outputs/")

# --- Morphological operations: clean up the binary mask ---
# We'll work with Otsu's result since it's automatically well-tuned
kernel = np.ones((3, 3), np.uint8)

# Opening = erosion followed by dilation -> removes small white noise specks
opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing = dilation followed by erosion -> fills small black holes inside white blobs
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

cv2.imwrite("../outputs/morph_cleaned.jpg", cleaned)

# --- Connected components (on cleaned mask) ---
# connectivity=8 means diagonal neighbors count as "connected" too (not just up/down/left/right)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    cleaned, connectivity=8
)

print(f"\n[With cleanup] Found {num_labels - 1} connected components (excluding background)")

for label_id in range(1, num_labels):
    x, y, w, h, area = stats[label_id]
    cx, cy = centroids[label_id]

    if area < 50:
        continue  # skip tiny noise blobs

    print(f"Component {label_id}: area={area}, bbox=({x},{y},{w},{h}), centroid=({cx:.1f},{cy:.1f})")

# --- Visualize: color each component differently ---
output_colored = np.zeros((*cleaned.shape, 3), dtype=np.uint8)
for label_id in range(1, num_labels):
    mask = labels == label_id
    color = np.random.randint(0, 255, size=3)
    output_colored[mask] = color

cv2.imwrite("../outputs/connected_components.jpg", output_colored)
print("Saved morph_cleaned.jpg and connected_components.jpg to outputs/")

# --- Comparison: connected components WITHOUT morphological cleanup ---
num_labels_raw, labels_raw, stats_raw, centroids_raw = cv2.connectedComponentsWithStats(
    otsu, connectivity=8
)
print(f"\n[Without cleanup] Found {num_labels_raw - 1} connected components (excluding background)")