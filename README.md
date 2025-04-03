Not yet completed -_-!

---

# Neuro Decoding Toolkit (NeuroDecKit)

**NeuroDecKit** is a comprehensive Python toolkit for decoding Motor Imagery (MI) EEG signals and performing advanced statistical analyses. By standardizing and optimizing the entire neural decoding workflow, this toolkit enhances classification accuracy and reproducibility, making it a valuable resource for researchers in neurotechnology and brain–computer interface (BCI) fields.

**Developer Version:** v0.1.0  
**Release Date:** 2024-06-01

---

## Overview

NeuroDecKit offers an end-to-end solution organized into four primary modules:
- **Dataset:** Seamless online access to a variety of MI-EEG datasets.
- **Pipeline:** A flexible framework with thousands of computational pipelines for signal processing and classification.
- **Evaluation:** Comprehensive performance assessments under various experimental scenarios.
- **Analysis:** Advanced tools for post-hoc statistical analysis and result visualization.

This modular design allows users to quickly download datasets, experiment with numerous pipelines, rigorously evaluate performance, and conduct in-depth analyses.

---

## Features

### 1. Dataset Module

NeuroDecKit provides robust dataset management with online access to a wide range of publicly available MI-EEG datasets. The key dataset attributes are summarized in the table below:

| **Dataset Attribute**                | **Details**                                      |
|--------------------------------------|--------------------------------------------------|
| **Number of Public Datasets**        | 16                                               |
| **Total Number of Subjects**         | 407                                              |
| **Subjects with Motor Disorders**    | 59                                               |
| **MI Tasks**                         | 8 (various MI psychological tasks)             |

- **Online Download:** Quick and seamless access to data for immediate integration into your research workflow.

---

### 2. Pipeline Module

#### Pre-processing

- **Frequency-Domain Pre-processing:**
  - Band-pass filtering
  - Notch filtering
  - Chebyshev IIR filter
  - Wavelet transform

- **Channel Selection:**
  - Correlation-based channel selection (CCS)
  - Riemannian Geometry-based channel selection (RCS)

- **Spatial Filtering:**
  - Common Spatial Pattern (CSP)
  - Laplace Transform (Laplacian)
  - Riemannian Geometry-based spatial filtering (RSF)

- **data augmentation:**
  - Time window slicing
  - Filter bank
  - Signal perturbation

- **normalization:**
  - Z-score normalization
  - Min-max normalization
  - Channel normalization
  - Trial normalization
  - Batch normalization

#### Pipeline Components Overview

The Pipeline Module provides a rich library of methods for processing EEG signals and generating classification pipelines. With **5296 distinct computational pipelines** available (excluding additional preprocessing and ensemble techniques), the module covers all critical steps:

| **Component**                    | **Number of Methods** | **Description**                                                       |
|----------------------------------|-----------------------|-----------------------------------------------------------------------|
| **Data Pre-Alignment**           | 5                     | Align raw EEG signals before further processing.                      |
| **Feature Extraction**           | 6+                    | Extract informative features from EEG signals.                        |
| **Feature Selection**            | 5                     | Select the most relevant features for classification.                 |
| **Feature Alignment**            | 3                     | Ensure consistent alignment of features across trials.                |
| **Traditional Classification**   | 13                    | Classical algorithms for EEG signal classification.                   |
| **Deep Learning Approaches**     | 7                     | Neural network-based methods for advanced classification.             |
| **End-to-End Classification**    | 6                     | Integrated methods for direct classification from raw data.           |
| **Ensemble Learning Methods**    | 3                     | Ensemble methods for combining multiple classifiers.                  |

#### Detailed Pipeline Method Lists

- **Data Pre-Alignment Strategies:**
  - None/TLDummy
  - CORAL
  - EA
  - RA
  - PRA

- **Feature Extraction Methods:**
  - CSP
  - TRCSP
  - CTSSP (CSSP|CTSP|TRCSP|STRCSP|SBLEST)
  - MDM
  - FGMDM
  - TSM

- **Feature Selection Methods:**
  - None
  - ANOVA
  - MIC
  - PCA
  - LASSO
  - RFE

- **Feature Alignment Methods:**
  - None
  - Z-score normalization
  - MMD
  - MEKT

- **Classification Algorithms:**
  - **Traditional Methods:**  
    - LDA
    - SVM
    - LR
    - KNN
    - DTC/ETC
    - Random Forest (RF)
    - GBC
    - XGBoost
    - LGBM
    - CatBoost
    - Gaussian Naive Bayes (GNB)
    - MLP   
  - **Deep Learning Approaches:**  
    - shallow CNN
    - deep CNN
    - EEGNet
    - FBCNet
    - Tensor-CSPNet
    - Graph-CSPNet
    - LMDA-Net
  - **End-to-End Methods:**  
    - SBLEST
    - MDWM
    - RKNN
    - RKSVM
    - TRCA
    - DCPM

- **Ensemble Learning Methods:**
  - None
  - Bagging
  - Stacking
  - Adaptive Boosting

*Note: The method names listed above are placeholders representing the current implementation. Future versions may introduce additional or refined methods.*

---

### 3. Evaluation Module

The Evaluation Module allows for robust performance assessment of the classification pipelines under multiple scenarios:

- **Within-Session Evaluation:** Tests consistency within a single session.
- **Cross-Experiment/Time Evaluation:** Assesses stability over different sessions or time periods.
- **Cross-Subject Evaluation:** Evaluates generalizability across different subjects.
- **Cross-Dataset Evaluation:** Benchmarks performance using independent datasets.

This module includes a complete evaluation framework with various metrics and statistical tests to ensure a comprehensive assessment.

---

### 4. Analysis Module

The Analysis Module offers advanced tools for post-hoc evaluation of experimental results. It supports in-depth statistical testing and result aggregation through the following components:

| **Analysis Component**             | **Number of Methods** | **Description**                                           |
|------------------------------------|-----------------------|-----------------------------------------------------------|
| **Model Evaluation Metrics**       | 8+                    | Quantitative metrics such as Accuracy, Precision, Recall, etc. |
| **Hypothesis Testing Methods**     | 4                     | Statistical tests (e.g., t-test, Wilcoxon test, ANOVA, permutation tests) for validating findings. |
| **Correction Methods**             | 5                     | Techniques (e.g., Bonferroni, Holm-Bonferroni, FDR) to control for multiple comparisons. |
| **Meta-Analysis Schemes**          | 2                     | Strategies to aggregate and synthesize results across experiments. |

#### Detailed Analysis Method Lists

- **Model Evaluation Metrics:**
  - Accuracy, Time cost, Precision, Recall, F1-Score, AUC, Kappa, Sensitivity, Specificity.
  
- **Hypothesis Testing Methods (4):**
  - t-test, Wilcoxon test, ANOVA, Permutation tests.
  
- **Correction Methods (5):**
  - Bonferroni correction, Holm-Bonferroni, False Discovery Rate (FDR), Šidák correction, tukey's HSD, among others.
  
- **Meta-Analysis Schemes (2):**
  - Stouffers' method, Standard meta-analytic approaches for combining results across studies. 
  - Fisher's method, a popular method for combining results across studies.

---

## Installation

To install NeuroDecKit locally, clone the repository and install via pip:

```bash
git clone https://github.com/your_username/NeuroDecKit.git
cd NeuroDecKit
pip install .
```

*Note: Ensure you are using Python 3.7+ and have installed all required dependencies (refer to `requirements.txt`).*

---

## Usage

Below is a basic example to get started with NeuroDecKit:

```python
from neurodeckit import .

# Reference to RSFDA and CTSSP repositories

```

For more detailed examples and API references, please see our [documentation](https://github.com/your_username/NeuroDecKit/wiki).

---

## Contributing

Contributions are welcome! Please review our [Contribution Guidelines](CONTRIBUTING.md) to learn how to get involved.

---

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for additional details.

---

## Contact

For inquiries, suggestions, or feedback, please contact:

- **Author:** LC Pan  
- **Email:** panlincong@tju.edu.cn

---

## Acknowledgments

NeuroDecKit was developed to meet the growing demand for standardized and efficient tools in MI-EEG signal decoding. We thank the research community for their support and valuable feedback throughout the development process.
