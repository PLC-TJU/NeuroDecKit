# NeuroDecKit: A Comprehensive MI-EEG Decoding Toolbox

[![Python](https://img.shields.io/badge/python-3.10.11-red.svg)](https://www.python.org/downloads/release/python-31011/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![996.icu](https://img.shields.io/badge/link-996.icu-green.svg)](https://996.icu)

NeuroDecKit is a modular and extensible MATLAB toolbox for Motor Imagery Electroencephalography (MI-EEG) decoding, featuring comprehensive algorithm integration and flexible pipeline construction capabilities.

## 🚀 Key Features

### Comprehensive Algorithm Ecosystem
NeuroDecKit systematically integrates algorithms across nine functional components:

| Module | Methods Count | Representative Algorithms |
|--------|---------------|---------------------------|
| **Spectral Filtering** | 2 | Butterworth, Chebyshev |
| **Channel Selection** | 3 | Correlation-based, CSP weights, Riemannian distance |
| **Spatial Filtering** | 3 | CSP, RSF, Laplace |
| **Data Alignment** | 4 | EA, RA, RPA, None|
| **Feature Extraction** | 8 | CSP, CSSP, RCSP, DSP, MDM, FgMDM, TSM, CTSSP |
| **Feature Selection** | 6 | ANOVA, MIC, PCA, LASSO, RFE, None |
| **Feature Alignment** | 5 | Z-score, MMD, MEKT, MEFA, None |
| **Classification** | 7 | LDA, SVM, LR, KNN, DTC, GNB, MLP |
| **End-to-End Models** | 5 | SBLEST, RKNN, RKSVM, TRCA, DCPM |
| **Deep Learning** | 11 | sCNN, dCNN, FBCNet, EEGNet, Tensor-CSPNet, Graph-CSPNet, LMDA-Net, EEGConformer, LightConvNet, IFNet, MSVTNet |
| **Ensemble Learning** | 3 | Voting, Boosting, Stacking|

### Advanced Transfer Learning Framework
NeuroDecKit features a sophisticated transfer learning system with:
- **6,000+** configurable transfer learning pipelines
- **Multi-level domain adaptation**: Data-level and feature-level alignment
- **Comprehensive transfer scenarios**: Cross-session, Cross-subject & Cross-dataset

### Flexible Pipeline Construction
- **300+** non-transfer learning pipeline configurations
- **Modular design** for easy algorithm combination
- **Extensible architecture** for custom method integration

## 📊 Experimental Results

### Experimental Protocols

NeuroDecKit supports multiple experimental paradigms to evaluate algorithm generalization:

#### 🔄 Cross-Session Validation
- **Training**: All data from first experimental session
- **Testing**: All data from second experimental session  
- **Objective**: Evaluate temporal stability and session-to-session transfer

#### 👥 Cross-Subject Validation (LOSO)
- **Training**: All subjects except one (source domain)
- **Testing**: Left-out subject (target domain)
- **Objective**: Evaluate intersubject generalization and domain adaptation

#### 🧬 Cross-Dataset Validation
- **Training**: All data from one dataset (source domain)
- **Testing**: All data from another dataset (target domain)
- **Objective**: Evaluate cross-dataset generalization and domain adaptation

### Benchmark Performance

Detailed results available in:

#### BNCI2014-001 (BCI Competition IV-2a)

[Cross-Session](tests/results/cross_session/BNCI2014_001/csv_result/result_mean.csv) |
[Cross-Subject](tests/results/cross_subject/BNCI2014_001/csv_result/result_mean.csv)

#### BNCI2015_001

[Cross-Session](tests/results/cross_session/BNCI2015_001/csv_result/result_mean.csv) |
[Cross-Subject](tests/results/cross_subject/BNCI2015_001/csv_result/result_mean.csv)

#### Pan2023 Dataset

[Cross-Session](tests/results/cross_session/Pan2023/csv_result/result_mean.csv) |
[Cross-Subject](tests/results/cross_subject/Pan2023/csv_result/result_mean.csv)

#### Shin2017A Dataset

[Cross-Session](tests/results/cross_session/Shin2017A/csv_result/result_mean.csv) |
[Cross-Subject](tests/results/cross_subject/Shin2017A/csv_result/result_mean.csv)


## 🤝 Related Research Resources

We express our gratitude to the open-source community, which facilitates the broader dissemination of research by other researchers and ourselves. The coding style in this repository is relatively rough. We welcome anyone to refactor it to make it more efficient. Our model codebase is largely based on the following repositories:

- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb) An open science project aimed at establishing a comprehensive benchmark for BCI algorithms using widely available EEG datasets.

- [<img src="https://img.shields.io/badge/GitHub-MetaBCI-b31b1b"></img>](https://github.com/TBC-TJU/MetaBCI) An open-source non-invasive brain-computer interface platform.

- [<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann) A Python library focused on Riemannian geometry methods for EEG signal classification. pyRiemann provides a suite of tools for processing and classifying EEG signals in Riemannian space.

- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode) Contains several deep learning models such as EEGNet, ShallowConvNet, and DeepConvNet, designed specifically for EEG signal classification. Braindecode aims to provide an easy-to-use deep learning toolbox.

- [<img src="https://img.shields.io/badge/GitHub-dpeeg-b31b1b"></img>](https://github.com/SheepTAO/dpeeg) provides several deep learning models such as EEGConformer, LightConvNet, and MSVTNet for EEG signal classification.

## 📄 License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.