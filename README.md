# Score Predictor: Student Performance Analysis and Prediction

This project implements a machine learning pipeline to analyze various student attributes and predict their final test scores. It includes data loading from an SQLite database, extensive Exploratory Data Analysis (EDA), data cleaning, feature engineering, preprocessing (handling missing values, encoding categorical data, scaling numerical features), and training/evaluation of both Multiple Linear Regression and Polynomial Regression models.

## Project Structure

```

Score_Predictor/
├── test.py \# Main Python script containing all modular functions for analysis.
├── score.db \# SQLite database file containing the student score data.
├── .gitignore \# Specifies intentionally untracked files to ignore (e.g., virtual environments, IDE files).
├── README.md \# This file.
└── requirements.txt \# Lists all Python dependencies for the project.

```

## Features

- **Data Ingestion:** Loads student data directly from an SQLite database.
- **Exploratory Data Analysis (EDA):**
  - Basic DataFrame information (`.info()`, `.describe()`).
  - Visualizations: Box plots and histograms for numerical distributions, correlation heatmap.
- **Data Cleaning & Preprocessing:**
  - Handles erroneous 'age' values.
  - Drops irrelevant/redundant columns.
  - Standardizes categorical string values.
  - Imputes missing numerical data (median for `attendance_rate`).
  - Imputes missing categorical data (mode for `CCA`).
  - Removes rows with missing target values (`final_test`).
- **Feature Engineering:**
  - Calculates `total_sleep_hours` from `sleep_time` and `wake_time`.
  - Derives `study_efficiency` from `hours_per_week` and `attendance_rate`.
  - Calculates `total_students` from `n_male` and `n_female`.
  - Creates `healthy_routine_score` from `total_sleep_hours` and `attendance_rate`.
- **Categorical Encoding:** Applies One-Hot Encoding to categorical features.
- **Data Splitting:** Divides data into training and testing sets.
- **Feature Scaling:** Standardizes continuous numerical features using `StandardScaler`.
- **Model Training & Evaluation:**
  - **Multiple Linear Regression:** Trains and evaluates a basic linear model.
  - **Polynomial Regression:** Trains and evaluates a polynomial regression model (default degree 3).
  - Evaluates models using R² score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
  - Visualizes actual vs. predicted values.
- **Model Comparison:** Presents a comparison of model performance metrics to identify the best-performing model.

## Requirements

- Python 3.x (tested with Python 3.9-3.12)
- All libraries listed in `requirements.txt`

## Installation

It is highly recommended to use a [Python virtual environment](https://docs.python.org/3/library/venv.html) to manage dependencies and avoid conflicts with other projects.

1.  **Clone the Repository (or download the files):**

    ```bash
    git clone [https://github.com/YourGitHubUsername/Score_Predictor.git](https://github.com/YourGitHubUsername/Score_Predictor.git)
    cd Score_Predictor
    ```

    (If you downloaded the zip, just extract it and navigate into the `Score_Predictor` folder).

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv_score_predictor
    # On Windows:
    .\venv_score_predictor\Scripts\activate
    # On macOS/Linux:
    source venv_score_predictor/bin/activate
    ```

3.  **Install Dependencies:**
    With your virtual environment activated, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    **Troubleshooting Installation Errors:** If you encounter permission errors (`OSError: [WinError 2]`) or issues with `numpy` or `scikit-learn`, try running your terminal/command prompt **as an Administrator** and then repeat the `pip install` command. Temporary disabling of antivirus might also be necessary.

## Usage

1.  **Place `score.db`:**
    Ensure the `score.db` SQLite database file is located in the same directory as the `test.py` script.

2.  **Run the Analysis Script:**
    Activate your virtual environment (if not already active) and execute the `test.py` script:

    ```bash
    python test.py
    ```

    The script will perform the entire analysis pipeline, printing progress and displaying plots. Close each plot window to proceed with the script execution.

## Customization

- **Database Path/Table Name:** Modify the `db_path` and `table_name` arguments in the `main_analysis` function call within the `if __name__ == "__main__":` block in `test.py`.
- **Polynomial Degree:** Change the `poly_degree` argument in the `main_analysis` function call to experiment with different polynomial complexities for Polynomial Regression.
- **EDA Plots:** The `explore_data` function generates many plots. If you want to skip this step for faster execution, you can comment out the `explore_data(df.copy())` line in the `main_analysis` function.
