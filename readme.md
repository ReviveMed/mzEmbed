# mzLearn, a data-driven LC/MS signal detection algorithm, enables pre-trained generative models for untargeted metabolomics

This repository contains the codebase for **mzEmbed**, a framework for developing pre-trained generative models and fine-tuning them for specific tasks for untargeted metabolomics datasets.

**Author**:
- [Leila Pirhaji](https://www.linkedin.com/in/pirhaji/)


## Overview of mzLearn

**mzLearn** is a data-driven algorithm designed to autonomously detect metabolite signals from raw LC/MS data without requiring input parameters from the user. The algorithm processes raw LC/MS data files in the open-source `mzML` format, iteratively learning signal characteristics to ensure high-quality signal detection. 

### Key Features of mzLearn:
- **Zero-parameter design:** No prior knowledge or QC samples are required.
- **Iterative learning:** mzLearn autonomously refines signal detection, correcting for retention time (rt) and intensity drifts caused by batch effects and run order.
- **Output:** A two-dimensional table of detected features defined by median rt and m/z values, with normalized intensities across samples.
- **Scalability:** Capable of handling large-scale datasets (e.g., 2,075 files in a single run).
- **Accessibility:** mzLearn’s website for accessing the tool is available at [http://mzlearn.com/](http://mzlearn.com/).

---

## Overview of mzEmbed Codebase

**mzEmbed** extends mzLearn’s capabilities by combining outputs from multiple datasets to develop pre-trained generative models and applying them to a range of metabolomics applications.

### Key Components of mzEmbed:
1. **Pre-trained Model Development:**
   - Combines metabolomics data from multiple studies to create robust pre-trained generative models.
   - Supports Variational Autoencoders (VAEs) for unsupervised learning of metabolite representations.
   - Enables parameter optimization using grid search and Optuna for hyperparameter tuning.
   - Outputs embeddings that capture biological and demographic variability, such as age, disease state.

2. **Fine-Tuning Pre-Trained Models:**
   - Allows fine-tuning of pre-trained models on independent datasets for improved task-specific performance.
   - Supports fine-tuning for binary classification, multi-class classification, and survival analysis.

3. **Task-Specific Model Refinement:**
   - Retrains the last layer of fine-tuned models for specific tasks, such as clinical classifcation  and surivival analysis.
   

4. **Advanced Architectures:**
   - Supports the development of joint learning models for treatment-independent, prognostic stratification of patient.
   - Implements adversarial learning to isolate treatment-specific predictive biomarkers, or predictive stratification of patient.

---

## Getting Started

### Requirements
- Python 3.9 or higher

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:ReviveMed/mzEmbed.git
   cd mzEmbed
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. building the package:
    ```
    cd mz_embed'
    python -m build
    pip install -e .
    ```

---


## Usage: 

The repository supports six main use cases, including pretraining, fine-tuning, and advanced learning architectures. **pretrain** and **finetune** directories includes examples of the Python commands for each use case. 


---

## License
This project is licensed under the Academic and Non-Profit Use License. See the LICENSE.txt file for details.


---

## Citation
If you use mzLearn or mzEmbed in your research, please cite:
```
[mzLearn, a data-driven LC/MS signal detection algorithm, enables pre-trained generative models for untargeted metabolomics]
[Leila Pirhaji, Jonah Eaton, Adarsh K. Jeewajee, Min Zhang, Matthew Morris, Maria Karasarides]
[Journal/Conference Name]
```

---