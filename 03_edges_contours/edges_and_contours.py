import cv2
import numpy as np

img = cv2.imread("../datasets/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

low_threshold = 100
high_threshold = 200
edges = cv2.Canny(blurred, low_threshold, high_threshold)

contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"Found {len(contours)} contours")

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

analyzed_img = img.copy()
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    # Skip tiny contours first — no point analyzing noise
    if area < 100:
        continue

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    num_vertices = len(approx)

    if num_vertices == 3:
        shape_name = "Triangle"
    elif num_vertices == 4:
        shape_name = "Rectangle/Square"
    elif num_vertices > 8:
        shape_name = "Circle-ish"
    else:
        shape_name = f"Polygon ({num_vertices} sides)"

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(analyzed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(f"Contour {i}: area={area:.1f}, perimeter={perimeter:.1f}, bbox=({x},{y},{w},{h}), shape={shape_name}")

cv2.imwrite("../outputs/canny_edges.jpg", edges)
cv2.imwrite("../outputs/contours_drawn.jpg", contour_img)
cv2.imwrite("../outputs/contours_analyzed.jpg", analyzed_img)

print("Saved canny_edges.jpg, contours_drawn.jpg, contours_analyzed.jpg to outputs/")