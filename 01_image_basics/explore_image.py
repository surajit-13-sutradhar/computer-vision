import cv2
import numpy as np

img = cv2.imread("../datasets/sample.jpg")

if img is None:
    raise FileNotFoundError("Could not load image — check the path and filename.")

print("Type of img:", type(img))
print("Shape:", img.shape)        # (height, width, channels)
print("Dtype:", img.dtype)        # uint8
print("Total pixels:", img.size)  # height * width * channels

height, width, channels = img.shape
print(f"Height: {height}, Width: {width}, Channels: {channels}")


# single pixel is a 3-element array: [Blue, Green, Red]
pixel = img[0, 0]
print("Pixel at (0,0):", pixel)

# --- Modifying pixels ---
# Set a 50x50 block in the top-left corner to pure red (remember: BGR order!)
img_modified = img.copy()
img_modified[:, :, 0] = 255  # B=0, G=0, R=255

# --- Extracting a single channel ---
blue_channel = img[:, :, 0]
green_channel = img[:, :, 1]
red_channel = img[:, :, 2]
print("Blue channel shape:", blue_channel.shape)  # 2D now, no channel dim

# --- BGR vs RGB ---
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Grayscale conversion ---
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale shape:", img_gray.shape)  # 2D — no channel dimension at all

# --- Resize ---
img_resized = cv2.resize(img, (200, 200))

# --- Crop ---
# Cropping is just array slicing — no special function needed
img_cropped = img[50:200, 50:200]

# --- Normalization ---
# Convert to float and scale from [0, 255] to [0, 1]
img_normalized = img.astype(np.float32) / 255.0

print("Original dtype/range:", img.dtype, img.min(), img.max())
print("Normalized dtype/range:", img_normalized.dtype, img_normalized.min(), img_normalized.max())

# --- Saving results ---
cv2.imwrite("../outputs/modified_red_corner.jpg", img_modified)
cv2.imwrite("../outputs/grayscale.jpg", img_gray)
cv2.imwrite("../outputs/resized.jpg", img_resized)
cv2.imwrite("../outputs/cropped.jpg", img_cropped)

print("Done. Check the outputs/ folder for saved images.")