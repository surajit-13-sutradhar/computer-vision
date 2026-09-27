# Computer Vision from Scratch

A hands-on, step-by-step computer vision project — building up from raw pixel arrays to deep-learning-based object detection, implementing each concept before using any library's black-box version of it.

**Status: Stages 1–4 complete.** Stages 5–8 in progress. This README will be updated as the project develops.

---

## Why this project exists

The goal isn't to produce a state-of-the-art detector — it's to demonstrate a working understanding of *how* computer vision techniques function, by implementing the core ideas manually before reaching for the library shortcut. Every stage below distinguishes what was implemented by hand from what a library (OpenCV, PyTorch) provided.

---

## Environment

- **OS:** Windows (PowerShell)
- **Language:** Python 3.10
- **Core libraries:** OpenCV (`opencv-python` 5.0.0), NumPy 2.2.6, PyTorch 2.11.0 (+cu128), torchvision, matplotlib 3.10.9, scikit-learn 1.7.2
- **Hardware:** NVIDIA RTX 4050 Laptop GPU, CUDA 12.8, verified via `torch.cuda.is_available()`

See `requirements.txt` for the exact pinned environment.

---

## Project structure

```
computer-vision/
│
├── 01_image_basics/          # Images as NumPy arrays
├── 02_convolution_filtering/  # Manual + OpenCV convolution, blur, sharpen
├── 03_edges_contours/         # Canny edges, contour analysis, shape approximation
├── 04_segmentation/           # Thresholding, morphology, connected components
├── 05_feature_extraction/     # (planned) ORB keypoints/descriptors
├── 06_object_detection/       # (planned) Grid-based detection concepts + pretrained detector
├── 07_transfer_learning/      # (planned) Frozen-backbone classifier training
├── 08_evaluation/             # (planned) NMS, precision/recall/F1, honest metrics
│
├── datasets/                  # Local input images (gitignored)
├── outputs/                   # Generated results (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Stage 1 — Image Basics

Explored the foundational idea that an image is a NumPy array: shape `(height, width, channels)`, dtype `uint8`, BGR channel order (OpenCV-specific, not RGB). Covered pixel access, editing spatial regions vs. individual channels, grayscale conversion, resizing/cropping as array slicing, and normalization (`uint8 [0,255]` → `float32 [0,1]`).

**Run:** `python 01_image_basics/explore_image.py`

## Stage 2 — Convolution and Filtering

Implemented 2D convolution manually with NumPy (a double loop sliding a kernel over an array), then verified the result matched OpenCV's `cv2.filter2D` exactly on the interior (non-border) pixels. Covered the distinction between "valid" convolution (output shrinks, no invented pixels) and "same" convolution (OpenCV's default, pads borders to preserve output size). Applied real kernels to a photo: box blur, Gaussian blur, a sharpening kernel, and an edge-enhancing kernel — including observing what happens when a kernel's values don't sum to 1 (brightness shifts) or sum to 0 (pure edge response).

**Run:** `python 02_convolution_filtering/manual_convolution.py` and `python 02_convolution_filtering/real_filters.py`

## Stage 3 — Edges and Contours

Built a `image → blur → Canny edges → contours → analysis` pipeline. Used OpenCV's Canny detector (Gaussian smoothing, gradient computation, non-maximum suppression, dual-threshold hysteresis) rather than a hand-rolled edge detector. Extracted contours and computed area, perimeter, bounding box, and a rough polygon-based shape classification (`cv2.approxPolyDP`).

**Honest finding:** on a real photograph, *no* contour was cleanly classified as a rectangle or triangle — natural images rarely have the crisp geometric edges that shape-approximation heuristics assume. This technique works well on controlled inputs (scanned documents, industrial parts) but is unreliable on general photos.

**Run:** `python 03_edges_contours/edges_and_contours.py`

## Stage 4 — Segmentation

Implemented binary thresholding, inverse thresholding, adaptive (local) thresholding, and Otsu's automatic thresholding. Applied morphological opening (erosion→dilation, removes noise specks) and closing (dilation→erosion, fills small gaps) to clean the binary mask. Extracted labeled regions via `cv2.connectedComponentsWithStats`, reporting per-blob area, bounding box, and centroid.

**Honest finding:** on the test photo, connected-components analysis found **1,706 raw components**, reduced to **258** after morphological cleanup — a ~7x reduction, demonstrating that cleanup is a functional necessity rather than a cosmetic step on real-world images. The results also showed that Otsu-based thresholding doesn't know which side of the threshold is "foreground" — the largest detected components corresponded to background/large uniform surfaces, not discrete objects, illustrating a real limitation of thresholding-based segmentation on natural photos versus controlled imaging conditions.

**Run:** `python 04_segmentation/segmentation.py`

---

## Stages 5–8 (planned)

- **Feature extraction:** ORB keypoint detection and descriptor matching between an image and a transformed version of itself.
- **Object detection:** conceptual grounding in grid-based detection (bounding boxes, confidence scores, IoU), followed by inference with a pretrained detector via torchvision — clearly labeled as pretrained, not built from scratch.
- **Transfer learning:** fine-tuning a frozen pretrained CNN backbone (new classification head only) on a chosen dataset, with a proper train/val/test split and no data leakage.
- **Evaluation:** manual non-max suppression implementation compared against a library version; classification metrics (precision, recall, F1, confusion matrix) computed only on held-out data.

This section will be filled in with real results as each stage is completed — no results are reported here in advance of actually running the code.

---

## Honesty notes

- Where a pretrained model is used (Stage 6 onward), it is explicitly labeled as pretrained.
- Where a technique is a simplified/educational implementation rather than a production-grade one (e.g., manual convolution, manual NMS), that distinction is stated rather than implied otherwise.
- No accuracy, loss, or other metric is reported unless it came from an actual run of the code.

## Limitations (so far)

- Classical shape/contour analysis (Stage 3) is unreliable on natural photographs.
- Thresholding-based segmentation (Stage 4) conflates large background regions with foreground objects unless the image has strong, deliberate contrast (e.g., a controlled photography setup).

## Future improvements

- Add example input/output images to this README once a stable test image set is finalized.
- Add a results table after Stages 7–8 produce real, measured metrics.
