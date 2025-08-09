# Fraud Detection with PyOD AutoEncoder

## Overview
This project uses a deep learning AutoEncoder model from the [PyOD](https://pyod.readthedocs.io/) library to detect fraudulent credit card transactions. The model is trained on only normal transactions and tested on a mixed dataset to evaluate detection performance. Evaluation includes both the default contamination-based threshold and an optional top-k% threshold for anomaly scores.

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions
- **Fraud cases:** 492 (~0.172%)
- **Features:** 30 anonymized features (V1–V28, Time, Amount) + target label (`Class`)

### 1. Environment Setup

It is highly recommended to use a Python virtual environment to manage dependencies.

1.  **Create and activate a virtual environment:**
    ```bash
    # Create the project directory and a data folder
    mkdir fraud-detection-python
    cd fraud-detection-python
    
    # Create and activate the virtual environment
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate      # On Windows
    ```

2.  **Install the required libraries:**
    The project relies on a few key libraries, including `pyod` for the AutoEncoder model, `pandas` for data manipulation, and `scikit-learn` for data preprocessing.
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file yet, you can install them manually:
    ```bash
    pip install pyod scikit-learn pandas numpy matplotlib seaborn tqdm
    ```

3.  **Obtain the dataset:**
    * Download the `creditcard.csv` file from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-garr/creditcardfraud).
    * Place the downloaded file in the `data/` directory.

### 2. How to Run the Project

Once the environment is set up and the data is in place, you can run the main script.

1.  Ensure your virtual environment is active.
2.  Execute the Python script from the project's root directory:
    ```bash
    python main.py
    ```

The script will perform the following actions:
* Load and preprocess the data.
* Train the AutoEncoder model on non-fraudulent data.
* Evaluate the model's performance on the entire dataset.
* Print a classification report, confusion matrix, and key metrics.
* Display visualizations of the anomaly scores.

### 3. Code Description

The `src/main.py` script follows a standard machine learning workflow:
* **Data Loading**: The `creditcard.csv` is loaded using `pandas`.
* **Preprocessing**: The `Time` and `Amount` features are scaled using `StandardScaler` from `scikit-learn` to ensure they have a similar scale to the PCA-transformed `V` features.
* **Model**: A `pyod.models.auto_encoder.AutoEncoder` is used. The model is trained exclusively on normal transactions (Class 0) to learn their underlying structure.
* **Evaluation**: The trained model's `decision_function` calculates the reconstruction error for each transaction. A higher error indicates a higher likelihood of being an anomaly (fraud). The results are evaluated using a classification report and a confusion matrix to measure the model's precision, recall, and F1-score.
* **Visualization**: The script generates plots to visualize the distribution of anomaly scores, providing a clear picture of how the model separates normal and fraudulent transactions.

