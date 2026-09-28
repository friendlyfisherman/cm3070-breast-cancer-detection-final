# AI-Driven Breast Cancer Detection from Mammographic Images Using Deep Learning

This repository contains the implementation for a final-year project investigating deep-learning approaches for binary breast cancer classification using mammographic images from the CBIS-DDSM dataset.

The project compares a custom convolutional neural network (CNN) with a transfer-learning model based on VGG16. Particular emphasis is placed on recall (sensitivity), because false-negative malignant cases are especially important in a screening context.

## Repository Structure

```text
.
├── notebooks/
│   ├── 00_dataset_audit_patient_split.ipynb
│   ├── 01_baseline_cnn.ipynb
│   ├── 01b_baseline_cnn_patient_grouped.ipynb
│   ├── 02_transfer_learning_vgg16.ipynb
│   └── 03_independent_test_evaluation.ipynb
│
├── Splits/
│   ├── train_patient_grouped.csv
│   ├── val_patient_grouped.csv
│   └── test_independent.csv
│
├── Models/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Notebook Workflow

The notebooks are intended to be read in the following order:

1. `00_dataset_audit_patient_split.ipynb`  
   Audits the CBIS-DDSM training metadata, removes duplicate/conflicting mammogram records, extracts patient identifiers, and creates patient-grouped training and validation splits.

2. `01_baseline_cnn.ipynb`  
   Contains the original prototype CNN experiment. This notebook is retained as development history and uses the earlier row-level train-validation split.

3. `01b_baseline_cnn_patient_grouped.ipynb`  
   Rebuilds the baseline CNN using the corrected patient-grouped split and provides the main baseline used for subsequent comparison.

4. `02_transfer_learning_vgg16.ipynb`  
   Trains and evaluates a frozen ImageNet-pretrained VGG16 transfer-learning model using the same patient-grouped split.

5. `03_independent_test_evaluation.ipynb`  
   Evaluates the corrected baseline CNN and VGG16 models on an independent CBIS-DDSM test set using fixed thresholds selected during validation.

## Dataset

This project uses the CBIS-DDSM mammography dataset.

The full image dataset is not included in this repository because of its size. The notebooks expect the dataset to contain:

```text
CBIS-DDSM/
├── csv/
└── jpeg/
```

The dataset location can be supplied using the environment variable:

```text
CBIS_DDSM_DIR
```

If this variable is not set, the notebooks use the default relative path:

```text
../data/CBIS-DDSM
```

For example, on Windows:

```python
import os
os.environ["CBIS_DDSM_DIR"] = r"C:\path\to\CBIS-DDSM"
```

This local override should be adjusted to match the user's own environment.

## Environment

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The primary dependencies are:

- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- Pillow

## Saved Data Splits

The `Splits` directory contains the final patient-independent manifests used by the corrected experiments:

- `train_patient_grouped.csv`
- `val_patient_grouped.csv`
- `test_independent.csv`

The training and validation split is performed at patient level to prevent patient overlap between partitions.

The independent test set is also checked against the training and validation patients before final evaluation.

## Trained Models

The trained `.keras` files are not included in the repository because of file-size constraints.

They can be reproduced by running:

- `notebooks/01b_baseline_cnn_patient_grouped.ipynb`
- `notebooks/02_transfer_learning_vgg16.ipynb`

The independent evaluation notebook expects the resulting model files to be placed in the `Models` directory as:

```text
baseline_cnn_patient_grouped.keras
vgg16_patient_grouped.keras
```

## Evaluation

Models are evaluated using multiple metrics including:

- accuracy
- precision
- recall / sensitivity
- F1-score
- specificity
- ROC-AUC
- confusion matrices

Recall is treated as a particularly important metric because the project focuses on reducing missed malignant cases.

Decision-threshold analysis is performed using the validation set. Candidate thresholds are then applied unchanged to the independent test set.

## Reproducibility

The patient-grouped dataset split uses a fixed `random_state=42`, so the train-validation partition can be reproduced.

Neural-network training remains stochastic because of operations such as random weight initialisation, dropout and mini-batch ordering. Consequently, retraining may produce somewhat different model metrics.

The results discussed in the project report correspond to the recorded trained-model evaluations used during the project.

## Final Evaluation Summary

The independent test set contains 359 unique mammograms from 200 patients:

- 216 benign
- 143 malignant

At the default decision threshold:

- Baseline CNN recall: 55.24%
- Baseline CNN ROC-AUC: 0.6459
- VGG16 recall: 51.05%
- VGG16 ROC-AUC: 0.6269

Using validation-selected recall-oriented thresholds:

- Baseline CNN at 0.45 threshold: 68.53% recall
- VGG16 at 0.40 threshold: 77.62% recall

The increase in sensitivity is accompanied by reduced specificity and an increased false-positive burden. The models should therefore be regarded as experimental research models rather than systems suitable for clinical deployment.

## Project Scope

This project is intended as an academic investigation of deep-learning methods for mammography classification.

It is not a clinical diagnostic system and has not undergone the external validation, regulatory assessment, or prospective clinical evaluation required for real-world medical use.
