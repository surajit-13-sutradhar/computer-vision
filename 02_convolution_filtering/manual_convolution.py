import numpy as np
import cv2
# A tiny 5x5 "image" — just a synthetic array of numbers, easy to trace by hand
image = np.array([
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [0,  0,  0,  0, 0],
    [0,  0,  0,  0, 0],
], dtype=np.float32)

# A simple 3x3 kernel: this one is an "edge detector" that responds
# strongly where pixel values change sharply left-to-right
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
], dtype=np.float32)

def convolve2d_manual(img, kernel):
    kh, kw = kernel.shape
    ih, iw = img.shape

    # Output is smaller than input because the kernel can't hang off the edge
    # (this is called "valid" convolution — no padding)
    oh = ih - kh + 1
    ow = iw - kw + 1

    output = np.zeros((oh, ow), dtype=np.float32)

    for y in range(oh):
        for x in range(ow):
            # Grab the region of the image currently under the kernel
            region = img[y:y+kh, x:x+kw]
            # Element-wise multiply, then sum everything -> one number
            output[y, x] = np.sum(region * kernel)

    return output

# OpenCV's filter2D applies a kernel using "same" padding by default behavior
# differs slightly, so we tell it explicitly to use no border padding to match
# our manual "valid" convolution
cv_result = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_ISOLATED)

result = convolve2d_manual(image, kernel)
print("Input image:\n", image)
print("\nKernel:\n", kernel)
print("\nConvolution result:\n", result)
print("\nOpenCV filter2D result (full size, includes padded edges):\n", cv_result)