# Computer Vision Assignment 1 – O-Ring Inspection

## Overview
This project is a Python-based O-ring defect inspection system developed for **Computer Vision Assignment 1**.  
The aim is to automatically inspect images of O-rings and determine whether each ring is a **PASS** or **FAIL** based on visible defects.

The solution follows the assignment requirements by implementing the image processing stages manually using **raw Python and NumPy**, while only using **OpenCV** for:
- reading images
- saving images
- displaying/annotating results

---

## Features
The program performs the following steps:

1. **Grayscale conversion**
   - Converts each input image to grayscale without using OpenCV image-processing functions.

2. **Histogram-based thresholding**
   - Builds a 256-bin histogram manually.
   - Uses **Otsu’s method** to automatically choose a threshold.

3. **Binary morphology**
   - Implements:
     - erosion
     - dilation
     - closing
   - Used to clean the binary image and close small holes/gaps.

4. **Connected Component Labelling**
   - Uses **8-connected BFS labelling**.
   - Extracts the **largest foreground region**, assumed to be the O-ring.

5. **Defect analysis**
   - Uses radial/polar sampling from the ring centre.
   - Measures:
     - missing sectors
     - multiple visible segments
     - thickness variation
     - inner/outer radius variation
   - Classifies each O-ring as **PASS** or **FAIL**.

6. **Output generation**
   - Saves:
     - annotated output images
     - binary/largest-mask images
     - CSV summary file
   - Displays processing time on each annotated image.

---

## Assignment Requirements Coverage
This project satisfies the assignment rubric as follows:

- **Threshold using image histogram and perform thresholding**  
  Implemented with manual histogram generation and Otsu thresholding.

- **Perform binary morphology to close interior holes**  
  Implemented with custom erosion, dilation, and closing.

- **Implement connected component labelling**  
  Implemented using 8-connected BFS labelling.

- **Analyse regions to classify O-rings as pass or fail**  
  Implemented using radial feature extraction and heuristic classification.

- **Program structure and timing annotation**  
  Processing time is measured with `time.perf_counter()` and written onto output images.

---

## Technologies Used
- **Python 3**
- **NumPy**
- **OpenCV** (`cv2`) for image input/output and annotation only

---

## File Structure
```text
.
├── oring_inspection.py
├── README.md
├── Oring1.jpg
├── Oring2.jpg
├── ...
└── Orings_out/
    ├── Oring1_annotated.png
    ├── Oring1_mask.png
    ├── Oring2_annotated.png
    ├── Oring2_mask.png
    └── summary.csv
