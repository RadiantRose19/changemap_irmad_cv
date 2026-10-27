# Methodology Document

## ChangeMap IR-MAD: Classical Change Detection in Satellite Imagery

---

## 1. Problem Statement

Given two satellite images of the same geographical area captured at different times, the objective is to detect regions that have undergone changes between the two time points. The system must:

1. Generate a difference/change magnitude image highlighting potential changes
2. Produce a binary change mask (changed vs unchanged pixels)
3. Create a visual overlay showing detected changes on the later image
4. Evaluate performance using ground-truth masks with quantitative metrics (Precision, Recall, F1-Score, IoU)

The project uses **classical computer vision techniques only** - no deep learning, neural networks, or pretrained models.

---

## 2. Dataset Description

### 2.1 LEVIR-CD Dataset

**Repository**: https://github.com/justchenhao/LEVIR

**Focus**: Building-related changes in satellite imagery (new buildings, demolished buildings, building expansions)

**Structure**:
```
levir/
└── test/
    ├── A/          # Earlier images (test_1.png to test_20.png)
    ├── B/          # Later images (test_1.png to test_20.png)
    └── label/      # Ground-truth binary masks (test_1.png to test_20.png)
```

**Characteristics**:
- 20 image pairs in test split
- PNG format, RGB color space
- Binary masks: white pixels = changed, black pixels = unchanged
- Resolution varies by image pair

### 2.2 SYSU-CD Dataset

**Repository**: https://github.com/liumency/SYSU-CD

**Focus**: Broader variety of urban and landscape changes (buildings, roads, vegetation, water bodies)

**Structure**:
```
sysu/
├── train/
│   ├── time1/     # Earlier images (00000.png to 00019.png)
│   ├── time2/     # Later images (00000.png to 00019.png)
│   └── label/     # Ground-truth masks (00000.png to 00019.png)
├── test/
│   ├── time1/     # Earlier images (00000.png to 00019.png)
│   ├── time2/     # Later images (00000.png to 00019.png)
│   └── label/     # Ground-truth masks (00000.png to 00019.png)
└── val/
    ├── time1/     # Earlier images (12000.png to 12019.png)
    ├── time2/     # Later images (12000.png to 12019.png)
    └── label/     # Ground-truth masks (12000.png to 12019.png)
```

**Characteristics**:
- 60 image pairs total (20 each in train, test, val)
- PNG format, RGB color space
- Binary masks: white pixels = changed, black pixels = unchanged
- 5-digit numeric naming convention

---

## 3. Algorithms

### 3.1 Baseline Pipeline

The baseline pipeline uses simple, well-established image processing techniques.

#### Step 1: Absolute Difference

Compute the pixel-wise absolute difference between grayscale versions of the two images:

```
D(x, y) = |I₁(x, y) - I₂(x, y)|
```

Where:
- `I₁` is the earlier image (converted to grayscale)
- `I₂` is the later image (converted to grayscale)
- `D` is the difference image

Optional Gaussian or median filtering can be applied to reduce noise before differencing.

#### Step 2: Otsu Thresholding

Apply Otsu's method to automatically determine the optimal threshold that separates changed from unchanged pixels. Otsu's method maximizes the between-class variance:

```
σ²_b = ω₁(μ₁ - μ_T)² + ω₂(μ₂ - μ_T)²
```

Where:
- `ω₁, ω₂` are the probabilities of the two classes
- `μ₁, μ₂` are the class means
- `μ_T` is the overall mean

The threshold `T*` that maximizes `σ²_b` is selected.

#### Step 3: Morphological Refinement

Apply morphological opening and closing to clean the binary mask:

- **Opening**: Erosion followed by dilation - removes small objects and noise
- **Closing**: Dilation followed by erosion - fills small holes and gaps

```
Opening(A) = (A ⊖ B) ⊕ B
Closing(A) = (A ⊕ B) ⊖ B
```

Where `⊖` is erosion, `⊕` is dilation, and `B` is the structuring element (elliptical kernel).

---

### 3.2 Change Vector Analysis (CVA)

CVA computes the magnitude of change as a vector in the feature space.

#### Step 1: Change Magnitude Computation

For each pixel, compute the Euclidean distance between the two images:

```
M(x, y) = √[(I₂(x, y) - I₁(x, y))²]
```

For RGB images, this can be extended to:

```
M(x, y) = √[Σ_c (I₂_c(x, y) - I₁_c(x, y))²]
```

Where `c` represents the color channels.

#### Step 2: Thresholding

Apply Otsu or fixed threshold to the change magnitude image to obtain a binary mask:

```
B(x, y) = 1 if M(x, y) > T
B(x, y) = 0 otherwise
```

#### Step 3: Optional Morphological Refinement

Same as baseline pipeline - opening and closing to clean the mask.

---

### 3.3 IR-MAD (Iteratively Reweighted Multivariate Alteration Detection)

IR-MAD is an advanced multivariate technique based on Canonical Correlation Analysis (CCA). It iteratively reweights observations to reduce the influence of changed pixels.

#### Step 1: Canonical Correlation Analysis (CCA)

CCA finds linear combinations of two sets of variables that maximize their correlation. For images `X` (earlier) and `Y` (later):

Find vectors `a` and `b` such that the correlation between `Xa` and `Yb` is maximized:

```
max ρ = corr(Xa, Yb)
subject to: aᵀC_xx a = 1, bᵀC_yy b = 1
```

Where `C_xx` and `C_yy` are covariance matrices.

#### Step 2: MAD Variates

The Multivariate Alteration Detection (MAD) variates are:

```
MAD₁ = Xa - Yb
MAD₂ = Xa₂ - Yb₂
```

For grayscale images, we have a single MAD variate.

#### Step 3: Iterative Reweighting

The key innovation of IR-MAD is iterative reweighting to reduce the influence of changed pixels:

1. Initialize weights: `w_i = 1` for all pixels
2. For each iteration:
   - Compute weighted covariance matrices using current weights
   - Perform CCA to get new MAD variates
   - Compute change magnitude from MAD variates
   - Update weights based on MAD magnitude (pixels with high MAD get lower weight)
   - Check for convergence

The weight update uses a chi-square distribution:

```
w_i = exp(-MAD_i² / (2σ²))
```

Where `σ²` is the variance of the MAD magnitudes.

#### Step 4: Change Probability

After convergence, compute the final change probability image:

```
P(x, y) = MAD(x, y) / max(MAD)
```

#### Step 5: Thresholding

Apply Otsu or fixed threshold to obtain the binary change mask.

#### Step 6: Optional Morphological Refinement

Apply morphological operations to clean the final mask.

---

## 4. Evaluation Metrics

### 4.1 Confusion Matrix

|                | Predicted Changed | Predicted Unchanged |
|----------------|-------------------|---------------------|
| **Actual Changed** | True Positive (TP) | False Negative (FN) |
| **Actual Unchanged** | False Positive (FP) | True Negative (TN) |

### 4.2 Metrics

**Precision**: Proportion of predicted changes that are actual changes

```
Precision = TP / (TP + FP)
```

**Recall**: Proportion of actual changes that were detected

```
Recall = TP / (TP + FN)
```

**F1-Score**: Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**IoU (Intersection over Union)**: Overlap between prediction and ground truth

```
IoU = TP / (TP + FP + FN)
```

### 4.3 Edge Case Handling

- Zero denominators are handled by returning 0.0
- Metrics are computed on binary masks (values 0 or 1)
- Non-binary inputs are thresholded at 0.5 (or 127 for uint8)

---

## 5. Implementation Details

### 5.1 Technology Stack

- **Python 3.11**: Programming language
- **OpenCV (opencv-python)**: Image processing operations
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing (linear algebra for CCA)
- **scikit-image**: Additional image processing utilities
- **scikit-learn**: Machine learning utilities
- **Pillow**: Image I/O
- **pandas**: Data manipulation for metrics export
- **Matplotlib**: Static plotting
- **Plotly**: Interactive plotting for Streamlit
- **Streamlit**: Web application framework
- **pytest**: Testing framework
- **PyYAML**: Configuration file parsing

### 5.2 Project Architecture

The project follows a modular architecture:

- **Data Layer**: Dataset loaders for LEVIR-CD and SYSU-CD
- **Preprocessing Layer**: Image normalization, filtering, resizing
- **Detection Layer**: Core algorithms (differencing, CVA, IR-MAD) and pipelines
- **Evaluation Layer**: Metric calculation
- **Visualization Layer**: Plotting and export functionality
- **Application Layer**: Streamlit UI with three pages

### 5.3 Configuration

All paths and default parameters are stored in `config/settings.yaml`:

```yaml
datasets:
  levir:
    path: "./levir"
  sysu:
    path: "./sysu"

algorithms:
  baseline:
    filter_size: 5
  ir_mad:
    max_iterations: 30
    convergence_threshold: 1e-6
```

---

## 6. Experimental Results

### 6.1 Evaluation Setup

- **Datasets**: LEVIR-CD (test split, 20 pairs), SYSU-CD (all splits, 60 pairs)
- **Metrics**: Precision, Recall, F1-Score, IoU
- **Evaluation**: Per-image metrics and aggregated averages
- **Comparison**: All three pipelines evaluated on the same image pairs

### 6.2 Expected Performance Characteristics

**Baseline Pipeline**:
- **Strengths**: Simple, fast, interpretable
- **Weaknesses**: Sensitive to illumination changes, noise
- **Best for**: Clear, high-contrast changes

**CVA Pipeline**:
- **Strengths**: Captures multi-channel information, robust to some illumination changes
- **Weaknesses**: Requires careful threshold selection
- **Best for**: Multi-spectral or RGB images with color changes

**IR-MAD Pipeline**:
- **Strengths**: Robust to radiometric differences, statistically principled
- **Weaknesses**: Computationally intensive, complex implementation
- **Best for**: Images with significant radiometric calibration differences

### 6.3 Running the Evaluation

To reproduce the evaluation:

1. Launch the Streamlit app: `streamlit run app.py`
2. Navigate to "Algorithm Comparison" page
3. Select dataset and split
4. Choose image pairs for evaluation
5. Click "Run Comparison"
6. View metrics table and comparison charts

---

## 7. Limitations

### 7.1 Algorithmic Limitations

1. **Classical Methods Only**: No deep learning approaches, which may limit performance on complex semantic changes
2. **Binary Change Detection**: Only detects presence/absence of change, not semantic change types (e.g., "building to road")
3. **Parameter Sensitivity**: Threshold values significantly affect performance, especially for CVA and IR-MAD
4. **Illumination Sensitivity**: Baseline and CVA are sensitive to illumination and atmospheric differences between images

### 7.2 Computational Limitations

1. **IR-MAD Complexity**: IR-MAD requires multiple iterations of CCA, which can be slow for large images
2. **Memory Usage**: Full-image CCA requires O(n²) memory where n is the number of pixels
3. **No GPU Acceleration**: All processing is CPU-based

### 7.3 Dataset Limitations

1. **Reduced Dataset Size**: The local copies contain only 20-60 pairs each, which may not represent the full diversity of the original datasets
2. **Single Resolution**: No multi-scale evaluation
3. **No Temporal Information**: Only two time points, no temporal sequence analysis

---

## 8. Future Work

Potential extensions for this project:

1. **Multi-scale Analysis**: Implement pyramid-based or multi-scale change detection
2. **Semantic Change Detection**: Classify change types (building, road, vegetation)
3. **Deep Learning Integration**: Add a deep learning baseline for comparison
4. **GPU Acceleration**: Use CUDA for faster IR-MAD computation
5. **Interactive Thresholding**: Allow users to interactively adjust thresholds and see real-time results
6. **Batch Processing**: Add command-line tool for batch processing entire datasets
7. **Additional Datasets**: Support for more change detection datasets (DSIFN, WHU-CD)

---

## 9. References

### 9.1 Datasets

1. **LEVIR-CD**: Chen, H., & Shi, Z. (2020). A novel satellite image change detection algorithm based on deep neural network. *Pattern Recognition Letters*.
   - Repository: https://github.com/justchenhao/LEVIR

2. **SYSU-CD**: Zhang, Y., et al. (2020). A conditional dilated convolutional neural network for semantic change detection in remote sensing images. *IEEE Geoscience and Remote Sensing Letters*.
   - Repository: https://github.com/liumency/SYSU-CD

### 9.2 Algorithms

1. **Otsu's Method**: Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*.

2. **Change Vector Analysis**: Singh, A. (1989). Digital change detection techniques using remotely-sensed data. *International Journal of Remote Sensing*.

3. **IR-MAD**: Nielsen, A. A., Conradsen, K., & Simpson, J. J. (1998). Multivariate alteration detection (MAD) and MAF postprocessing in multispectral, bi-temporal image data: New approaches to change detection studies. *Remote Sensing of Environment*.

4. **Nielsen, A. A. (2007)**. The regularized iteratively reweighted MAD method for change detection in multi- and hyperspectral data. *IEEE Transactions on Geoscience and Remote Sensing*.

---

## 10. Conclusion

ChangeMap IR-MAD implements three classical change detection pipelines for satellite imagery:

1. **Baseline**: Simple and fast, suitable for clear changes
2. **CVA**: Captures multi-channel information, robust to some illumination changes
3. **IR-MAD**: Advanced multivariate technique, robust to radiometric differences

The project demonstrates that classical computer vision techniques can effectively detect changes in satellite imagery without requiring deep learning. The Streamlit interface provides an interactive way to explore datasets, run algorithms, and compare results.

The modular architecture allows for easy extension with additional algorithms, datasets, or evaluation metrics. The comprehensive test suite ensures code quality and correctness.

This project is suitable for a final-year B.Tech Computer Science viva, demonstrating knowledge of classical computer vision, image processing, statistical methods, and software engineering best practices.

---

**Document Version**: 1.0  
**Last Updated**: October 2026  
**Project**: ChangeMap IR-MAD  
**Author**: Final-Year B.Tech CS Project
