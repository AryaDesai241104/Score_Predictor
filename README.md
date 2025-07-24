# Score Predictor

A data science project for predicting student test scores based on various academic, demographic, and lifestyle features. This project includes data exploration, preprocessing, feature engineering, and prepares the data for machine learning modeling.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Usage](#usage)
- [Requirements](#requirements)
- [References](#references)

---

## Project Overview

This repository demonstrates the workflow of a predictive modeling project where the goal is to predict students' `final_test` scores using a variety of features such as academic habits, demographics, and lifestyle factors. The workflow covers data extraction from a SQLite database, exploratory data analysis (EDA), data cleaning, feature engineering, encoding, scaling, and preparation for machine learning.

## Features

- **Data Extraction:** Reads student data from a SQLite database (`score.db`).
- **EDA:** Summary statistics, boxplots, histograms, and correlation heatmaps.
- **Data Cleaning:** Handles missing values, corrects inconsistent entries, and drops redundant columns.
- **Feature Engineering:** 
    - Calculates total sleep hours.
    - Computes study efficiency.
    - Generates a healthy routine score.
    - Encodes categorical features.
- **Preprocessing:** Splits data into training and testing sets, and standardizes numerical features.
- **Ready for Modeling:** The cleaned and engineered dataset can be used for regression or classification tasks.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryaDesai241104/Score_Predictor.git
   cd Score_Predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- The main dataset is stored in a SQLite database file: `score.db`.
- The data table (`score`) includes fields such as number of siblings, direct admission status, CCA (co-curricular activity), learning style, gender, tuition, final test score, number of male/female students, age, hours per week, attendance rate, sleep/wake times, mode of transport, and bag color.

## Exploratory Data Analysis

The notebook `test.ipynb` includes:

- Loading data from the SQLite database.
- Displaying the structure (`df.info()`), sample rows, and summary statistics (`df.describe()`).
- Visualizing distributions and outliers using boxplots and histograms.
- Plotting a correlation heatmap for all numeric features.

## Data Preprocessing

- **Handling Missing Values:** Uses `SimpleImputer` to fill missing attendance rates with the median and fills missing CCA values with the mode. Drops rows where `final_test` is missing.
- **Cleaning & Standardizing:** Unifies category names, replaces inconsistent entries, and removes invalid ages.
- **Encoding Categorical Features:** Uses one-hot encoding for CCA, gender, direct admission, tuition, and learning style.
- **Feature Engineering:** 
    - Calculates total sleep hours from sleep and wake times.
    - Computes study efficiency as `hours_per_week * attendance_rate`.
    - Sums male and female counts for total students.
    - Creates a healthy routine score as `total_sleep_hours * attendance_rate`.
- **Scaling:** Standardizes selected numerical columns using `StandardScaler`.

## Usage

- Open and run the Jupyter Notebook `test.ipynb` step by step to perform EDA, cleaning, feature engineering, and data preparation.
- The notebook prepares the data for regression modeling (e.g., multiple linear regression, XGBoost, LightGBM).

## Requirements

See `requirements.txt` for all dependencies.

```
pandas>=1.0.0
numpy>=1.20.0
seaborn>=0.11.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
xgboost>=1.7.0  
lightgbm>=3.3.0 
```

Install with:
```bash
pip install -r requirements.txt
```

## References

- [AIAP® Preparatory Bootcamp Intake 1 - Assignment (PDF)](AIAPⓇPreparatoryBootcampIntake1-Assignment.pdf)
- Data and notebook contributors: AryaDesai241104

---

**Note:** This repository is for educational and demonstration purposes. The data and code can be adapted for similar machine learning workflows.
