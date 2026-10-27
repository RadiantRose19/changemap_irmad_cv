# IRMAD-Based Change Detection Mapping

## Description

ChangeMap IR-MAD is a classical computer vision project for detecting regions of change between two satellite images of the same geographical area captured at different times. The system implements the Iteratively Reweighted Multivariate Alteration Detection (IR-MAD) algorithm, an advanced multivariate technique based on Canonical Correlation Analysis (CCA) with iterative reweighting to reduce the influence of changed pixels.

The project uses only classical computer vision techniques (no deep learning) and implements three primary detection pipelines:

1. **Baseline**: Image differencing → Otsu thresholding → Morphological refinement
2. **CVA (Change Vector Analysis)**: Computes change magnitude vectors → Thresholding → Optional morphology
3. **IR-MAD (Iteratively Reweighted MAD)**: Advanced multivariate technique using Canonical Correlation Analysis with iterative reweighting

The system generates difference images, binary change masks, visual overlays, and evaluates performance using Precision, Recall, F1-Score, and IoU metrics against ground-truth masks.

## Project Structure

```
changemap/
├── app.py                          # Streamlit application (main entry point)
├── config/
│   └── settings.yaml              # Configuration file for paths and parameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py         # Abstract base dataset class
│   │   └── dataset_loader.py      # LEVIR-CD and SYSU-CD dataset loaders
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_utils.py         # Image preprocessing utilities
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── differencing.py         # Absolute difference algorithm
│   │   ├── thresholding.py        # Otsu and fixed thresholding
│   │   ├── morphology.py           # Morphological operations
│   │   ├── cva.py                  # Change Vector Analysis
│   │   ├── ir_mad.py               # IR-MAD algorithm implementation
│   │   └── pipelines.py            # Three detection pipelines
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Precision, Recall, F1, IoU calculation
│   └── visualization/
│       ├── __init__.py
│       ├── plotting.py             # Visualization utilities
│       └── export.py               # Export functionality
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # Tests for preprocessing utilities
│   ├── test_detection.py           # Tests for detection algorithms
│   ├── test_metrics.py             # Tests for evaluation metrics
│   └── test_dataset.py             # Tests for dataset loaders
├── docs/
│   └── methodology.md              # Detailed methodology document
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Windows, Linux, or macOS
- **Git**: Optional, for version control
- **System Dependencies**: None (all dependencies are Python packages)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/RadiantRose19/changemap_irmad_cv.git
cd changemap_irmad_cv
```

### 2. Create Virtual Environment

**Using venv (recommended):**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

**Using conda (alternative):**

```bash
conda create -n changemap python=3.11
conda activate changemap
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Setup (Optional)

The project supports LEVIR-CD and SYSU-CD datasets. Download and organize them as follows:

```
levir/
└── test/
    ├── A/          # Earlier images
    ├── B/          # Later images
    └── label/      # Ground-truth masks

sysu/
├── train/
│   ├── time1/     # Earlier images
│   ├── time2/     # Later images
│   └── label/     # Ground-truth masks
├── test/
│   ├── time1/
│   ├── time2/
│   └── label/
└── val/
    ├── time1/
    ├── time2/
    └── label/
```

Datasets are not included in the repository due to size. See the methodology document for dataset sources.

## How to Run

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

The Streamlit app provides three pages:
1. **Dataset Explorer**: Browse datasets and image pairs
2. **Change Detection**: Run algorithms with adjustable parameters
3. **Algorithm Comparison**: Benchmark all pipelines

### Run Automated Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_detection.py
pytest tests/test_metrics.py
pytest tests/test_preprocessing.py
pytest tests/test_dataset.py

# Run with verbose output
pytest tests/ -v
```

### Run Specific Pipeline Programmatically

```python
from src.data.dataset_loader import LevirDataset
from src.detection.pipelines import get_pipeline
from src.evaluation.metrics import calculate_metrics

# Load dataset
dataset = LevirDataset("./levir")
pairs = dataset.get_image_pairs()

# Load image pair
earlier_img, later_img, label_img = load_image_pair(*pairs[0], root_path=dataset.root_path)

# Run pipeline
pipeline = get_pipeline("ir_mad")
predicted_mask, intermediate = pipeline.run(earlier_img, later_img)

# Calculate metrics
metrics = calculate_metrics(predicted_mask, label_img)
print(f"F1-Score: {metrics.f1:.4f}, IoU: {metrics.iou:.4f}")
```

## Additional Documentation

- **docs/methodology.md**: Detailed explanation of algorithms, mathematical background, experimental results, and limitations

## License

This project is for educational purposes. The datasets (LEVIR-CD and SYSU-CD) have their own licenses.

## Acknowledgments

- LEVIR-CD Dataset: https://github.com/justchenhao/LEVIR
- SYSU-CD Dataset: https://github.com/liumency/SYSU-CD
- IR-MAD Algorithm: Based on the work by Nielsen et al. on Iteratively Reweighted Multivariate Alteration Detection

## 📊 Streamlit Application Pages

### Page 1: Dataset Explorer
- Select LEVIR-CD or SYSU-CD dataset
- Browse available image pairs
- View earlier image, later image, and ground-truth mask
- Check image dimensions

### Page 2: Change Detection
- Select dataset and image pair (or upload custom images)
- Choose pipeline: Baseline, CVA, or IR-MAD
- Adjust algorithm parameters (threshold, morphology kernel, IR-MAD iterations)
- Run change detection
- View results:
  - Earlier and later images
  - Difference/change magnitude image
  - Predicted binary mask
  - Overlay of detected changes
  - Evaluation metrics (F1-score, IoU, Precision, Recall)

### Page 3: Algorithm Comparison
- Select dataset and split
- Choose images for benchmark evaluation
- Run all three pipelines
- Compare metrics table
- View F1-score and IoU comparison charts
- Analyze results and limitations

## 🔬 Algorithms

### Baseline Pipeline
1. **Absolute Difference**: Compute pixel-wise absolute difference between grayscale versions of the two images
2. **Otsu Thresholding**: Automatically determine optimal threshold to separate changed from unchanged pixels
3. **Morphological Refinement**: Apply opening and closing to remove noise and fill small holes

### Change Vector Analysis (CVA)
1. **Change Magnitude**: Compute Euclidean distance between corresponding pixels in the two images
2. **Thresholding**: Apply Otsu or fixed threshold to the change magnitude image
3. **Optional Morphology**: Refine the binary mask using morphological operations

### IR-MAD (Iteratively Reweighted MAD)
1. **Canonical Correlation Analysis (CCA)**: Find linear combinations of the two images that maximize correlation
2. **MAD Variates**: Compute Multivariate Alteration Detection variates
3. **Iterative Reweighting**: Reduce influence of changed pixels by iteratively updating weights based on MAD magnitude
4. **Change Probability**: Generate chi-square-based change probability from MAD variates
5. **Thresholding**: Apply threshold to produce binary change mask

## 📈 Evaluation Metrics

The project evaluates change detection performance using four metrics:

- **Precision**: TP / (TP + FP) - How many predicted changes are actual changes
- **Recall**: TP / (TP + FN) - How many actual changes were detected
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: TP / (TP + FP + FN) - Overlap between prediction and ground truth

## 🎯 Key Features

- **Classical Computer Vision Only**: No deep learning, neural networks, or cloud APIs
- **Multiple Datasets**: Support for LEVIR-CD and SYSU-CD datasets
- **Three Detection Pipelines**: Compare baseline, CVA, and IR-MAD approaches
- **Interactive UI**: Clean Streamlit interface with three pages
- **Real Metrics**: Actual evaluation against ground-truth masks
- **Export Functionality**: Save masks, overlays, metrics as CSV, and settings as JSON
- **Automated Tests**: Comprehensive test suite for all components
- **Configurable**: YAML-based configuration for paths and parameters

## 📚 Datasets

### LEVIR-CD
- **Focus**: Building-related changes in satellite images
- **Structure**: `test/{A, B, label}/`
- **Format**: PNG images, RGB
- **Size**: 20 image pairs in test split

### SYSU-CD
- **Focus**: Change detection across urban and landscape scenes
- **Structure**: `{train, test, val}/{time1, time2, label}/`
- **Format**: PNG images, RGB
- **Size**: 20 pairs each in train, test, and val splits (60 total)

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

- Dataset paths
- Algorithm parameters (filter size, thresholds, iterations)
- Morphological kernel sizes
- Output directories

## 🧪 Testing

The project includes comprehensive tests:

- **Preprocessing tests**: Normalization, filtering, resizing
- **Detection tests**: All algorithms and pipelines
- **Metrics tests**: Metric calculation with edge cases
- **Dataset tests**: Dataset loading and validation

Run tests with `pytest tests/`

## 📖 Documentation

- **README.md**: This file - project overview and setup instructions
- **docs/methodology.md**: Detailed explanation of algorithms, mathematical background, and experimental results

## 🤝 Contributing

This is a final-year academic project. For questions or suggestions, please refer to the methodology document.

## 📝 License

This project is for educational purposes. The datasets (LEVIR-CD and SYSU-CD) have their own licenses.

## 🎓 Acknowledgments

- LEVIR-CD Dataset: https://github.com/justchenhao/LEVIR
- SYSU-CD Dataset: https://github.com/liumency/SYSU-CD
- IR-MAD Algorithm: Based on the work by Nielsen et al. on Iteratively Reweighted Multivariate Alteration Detection

## 🐛 Known Limitations

1. **IR-MAD Computation**: The IR-MAD algorithm can be computationally intensive for large images
2. **Parameter Sensitivity**: Some algorithms (especially CVA and IR-MAD) are sensitive to threshold parameters
3. **Classical Methods Only**: No deep learning approaches, which may limit performance on complex scenes
4. **Binary Change Detection**: Only detects presence/absence of change, not semantic change types

## 📞 Contact

For questions about this project, please refer to the methodology document or contact the project developer.

---

Built with ❤️ using Python, OpenCV, and Streamlit
