# Anomaly Detection in Log Data

This project is designed to detect anomalies in log sequences using various techniques such as n-gram detection, count vector clustering, and edit distance detection. It works with structured log datasets and requires specific input files to function properly.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset Structure](#dataset-structure)
3. [How to Use](#how-to-use)
4. [Detection Methods](#detection-methods)
5. [Output](#output)
6. [Customizing for New Data](#customizing-for-new-data)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

To run this project, you need the following:

- Python 3.7 or higher
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Structure

The system expects the dataset to be structured as follows:

```
dataset_directory/
    train
    test_normal
    test_abnormal
    anomaly_label.csv
```

### File Descriptions

- **train:**
  - Contains training sequences in the format:
    ```
    blk_12345,1 2 3 4 5
    blk_67890,6 7 8 9 10
    ```
  - Each line represents a sequence of log events, where:
    - `blk_12345` is the sequence ID.
    - `1 2 3 4 5` is the sequence of log events.

- **test_normal:**
  - Contains normal test sequences in the same format as `train`.

- **test_abnormal:**
  - Contains abnormal test sequences in the same format as `train`.

- **anomaly_label.csv:**
  - Contains labels for the test sequences in the format:
    ```
    BlockId,Label
    blk_12345,Normal
    blk_67890,Anomaly
    ```

## How to Use

### Prepare Your Dataset:
- Organize your log data into the required folder structure (see [Dataset Structure](#dataset-structure)).

### Clone the Repository:
```bash
git clone https://github.com/your-repo/anomaly-detection.git
cd anomaly-detection
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Run the Script:
```bash
python main.py --data_dir /path/to/dataset_directory
```

#### Command-Line Arguments
- `--data_dir`: Path to the directory containing the dataset files.

## Detection Methods

The system implements the following detection methods:

- **New Event Detection:** Detects sequences containing events not seen in the training data.
- **Sequence Length Detection:** Detects sequences that are shorter or longer than the expected range.
- **Count Vector Clustering:** Detects anomalies based on the similarity of count vectors between test and training sequences.
- **N-gram Detection:** Detects anomalies based on the presence of unknown n-grams in test sequences.
- **Edit Distance Detection:** Detects anomalies based on the edit distance (Levenshtein distance) between test and training sequences.

## Output

The script prints evaluation metrics for each detection method, including:

- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)
- Precision (P)
- Recall (R)
- F1 Score (F1)
- Accuracy (ACC)
- Matthews Correlation Coefficient (MCC)

### Example Output:
```
New event detection
 Threshold=None
 Time=0.123
 TP=100
 FP=10
 TN=90
 FN=5
 TPR=R=0.952
 FPR=0.1
 TNR=0.9
 P=0.909
 F1=0.93
 ACC=0.95
 MCC=0.85
```

## Customizing for New Data

To use the system with a new dataset:

1. **Prepare the Dataset:**
   - Ensure the dataset is structured as described in the [Dataset Structure](#dataset-structure) section.
   - Place the dataset files in a directory (e.g., `new_dataset/`).

2. **Run the Script:**
```bash
python main.py --data_dir /path/to/new_dataset
```

## Troubleshooting

- **Missing Files:**
  - Ensure all required dataset files (`train`, `test_normal`, `test_abnormal`, `anomaly_label.csv`) are present in the dataset directory.

- **Unexpected Results:**
  - Check the dataset format and ensure it matches the expected structure.
  - Verify that the detection methods are appropriate for your dataset.
