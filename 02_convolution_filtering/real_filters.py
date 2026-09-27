import cv2
import numpy as np

img = cv2.imread("../datasets/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# --- Box blur: every output pixel is the average of its neighborhood ---
# A 5x5 kernel where every value is 1/25 -> literally "average the 25 pixels around me"
box_kernel = np.ones((51, 51), dtype=np.float32) / 2601.0
box_blur = cv2.filter2D(gray, -1, box_kernel)

# --- Gaussian blur: like box blur, but nearby pixels matter more than far ones ---
gaussian_blur = cv2.GaussianBlur(gray, (51, 51), sigmaX=0)

# --- Sharpen: exaggerate the difference between a pixel and its neighbors ---
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

# --- Edge-enhancing filter: this kernel sums to 0, not 1 ---
# So flat/uniform regions collapse to ~0 (no edge, no response)
# while regions with sharp intensity changes produce large values
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=np.float32)
edges = cv2.filter2D(gray, -1, edge_kernel)

cv2.imwrite("../outputs/box_blur.jpg", box_blur)
cv2.imwrite("../outputs/gaussian_blur.jpg", gaussian_blur)
cv2.imwrite("../outputs/sharpened.jpg", sharpened)
cv2.imwrite("../outputs/edges_filter2d.jpg", edges)

print("Saved box_blur.jpg, gaussian_blur.jpg, sharpened.jpg, edges_filter2d.jpg to outputs/")