
# Input Dataset for mzLearn and mzEmbed

This document describes the structure and requirements for the input datasets used in **mzEmbed** workflows.

---

## Overview

The input data for **mzEmbed** workflows consists of processed LC/MS metabolomics data by **mzLearn**, saved in `.csv` format. These datasets are formatted into training, validation, and test sets for pretraining and fine-tuning applications. Proper organization of the input data folder is essential for running the provided scripts.

---

## Input Folder Format

The input data folder must follow the structure below:

### Folder Contents:
- **Pretraining Data**:
  - `X_Pretrain_Discovery_Train.csv`, `X_Pretrain_Discovery_Val.csv`, `X_Pretrain_Test.csv`: Split pretraining data into training, validation, and test sets.
  - `y_Pretrain_Discovery_Train.csv`, `y_Pretrain_Discovery_Val.csv`, `y_Pretrain_Test.csv`: Labels for corresponding pretraining splits.

- **Fine-Tuning Data**:
  - `X_Finetune_Discovery_Train.csv`, `X_Finetune_Discovery_Val.csv`, `X_Finetune_Test.csv`: Split fine-tuning data into training, validation, and test sets.
  - `y_Finetune_Discovery_Train.csv`, `y_Finetune_Discovery_Val.csv`, `y_Finetune_Test.csv`: Labels for corresponding fine-tuning splits.

### Example Directory Structure:
```
input_data/
├── X_Pretrain_Discovery_Train.csv
├── X_Pretrain_Discovery_Val.csv
├── X_Pretrain_Test.csv
├── y_Pretrain_Discovery_Train.csv
├── y_Pretrain_Discovery_Val.csv
├── y_Pretrain_Test.csv
├── X_Finetune_Discovery_Train.csv
├── X_Finetune_Discovery_Val.csv
├── X_Finetune_Test.csv
├── y_Finetune_Discovery_Train.csv
├── y_Finetune_Discovery_Val.csv
├── y_Finetune_Test.csv
```

---

## Dataset Availability

The dataset used in the paper are availbe from [Metabolomics Workbench](https://www.metabolomicsworkbench.org/) and [Metabolights](https://www.ebi.ac.uk/metabolights/) datasbased.

---

## Data File Format

### **1. Feature Data (`X_*` Files)**

- **Description**: Each `X_*` file contains the metabolite features for the samples in the corresponding dataset (e.g., pretraining or fine-tuning).
- **Structure**:
  - **Rows**: Each row corresponds to a sample.
  - **Columns**: Each column corresponds to normalized intensity of a metabolite feature.
  - **File Format**: `.csv` file where the first row contains feature names, and the first column contains sample IDs.

#### Example: `X_Finetune_Discovery_Train.csv`

| StudyID_SampleID  | Metabolite_1 | Metabolite_2 | Metabolite_3 | ... |
|-------------------|--------------|--------------|--------------|-----|
| ST001422_001      | 0.234        | 1.876        | 0.643        | ... |
| ST001422_002      | 0.189        | 2.304        | 0.561        | ... |
| ST001422_003      | 0.467        | 1.923        | 0.782        | ... |

---

### **2. Label/Metadata Data (`y_*` Files)**

- **Description**: Each `y_*` file contains the corresponding clinical or demographic metadata for the samples.
- **Structure**:
  - **Rows**: Each row corresponds to a sample.
  - **Columns**: Each column corresponds to a specific clinical or demographic variable (e.g., age, gender, treatment group).
  - **File Format**: `.csv` file where the first row contains variable names, and the first column contains sample IDs.

#### Example: `y_Finetune_Discovery_Train.csv`

| StudyID_SampleID  | Age  | Gender | Overall Survival | Treatment   |
|-------------------|------|--------|----------------|-------------|
| ST001422_001      | 45   | Male   | 32.5           | Drug_A      |
| ST001422_002      | 50   | Female | 27.1           | Placebo     |
| ST001422_003      | 38   | Male   | 22.8           | Drug_B      |

---

### Summary

- **Feature Data (`X_*` Files)**: Samples as rows, metabolite features as columns.
- **Metadata (`y_*` Files)**: Samples as rows, clinical/demographic variables as columns.

---

## Dataset Requirements

### **Pretraining Data**:
- Should contain large-scale metabolomics data (e.g., >5,000 samples).
- Can optionally include demographic or clinical metadata in for more targeted pretraining.

### **Fine-Tuning Data**:
- Designed for domain-specific tasks.
- Requires training, validation, and test splits for supervised fine-tuning.
- Labels (e.g., binary classification, multi-class labels, or survival times) should be included in the `y_Finetune_*` files.

---

## Additional Notes
- **Data Format**: Ensure all `.csv` files are formatted consistently, with feature values in columns and samples in rows.
- **Compatibility**: The provided input structure is compatible with all mzEmbed scripts, ensuring seamless execution of pretraining, fine-tuning, and advanced architectures.

---
