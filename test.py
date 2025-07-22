import sqlite3 as sql
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1. Data Loading and Connection ---

def connect_to_database(db_path='score.db'):
    """
    Connects to an SQLite database and returns the connection and cursor objects.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        tuple: (sqlite3.Connection, sqlite3.Cursor)
    """
    conn = sql.connect(db_path)
    cursor = conn.cursor()
    print(f"Connected to database: {db_path}")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    return conn, cursor

def load_data_from_table(conn, table_name='score'):
    """
    Loads data from a specified SQLite table into a Pandas DataFrame.

    Args:
        conn (sqlite3.Connection): Active database connection.
        table_name (str): Name of the table to load.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    print(f"\nDataFrame loaded from table '{table_name}'. Sample (5 rows):")
    print(df.sample(5))
    return df

# --- 2. Data Exploration ---

def explore_data(df):
    """
    Performs basic data exploration (info, describe) and generates plots.

    Args:
        df (pd.DataFrame): DataFrame to explore.
    """
    print("\n--- Data Exploration ---")
    print("\nDataFrame Info:")
    df.info()

    print("\nDataFrame Description (Numeric Columns):")
    print(df.describe())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nGenerating Boxplots for Numeric Columns:")
    for col in numeric_cols:
        plt.figure(figsize=(6, 2.5))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    print("\nGenerating Histograms (KDE) for Numeric Columns:")
    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], bins=30, kde=True, color='orange')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    print("\nGenerating Correlation Heatmap for Numeric Features:")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

# --- 3. Data Cleaning and Preprocessing ---

def clean_and_transform_data(df):
    """
    Corrects data types, handles missing values, and engineers new features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    print("\n--- Data Cleaning and Transformation ---")

    # Correcting 'age' values (assuming negative ages are errors)
    df = df[df['age'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Drop original redundant columns
    df.drop(columns=['index', 'bag_color', 'student_id', 'age', 'mode_of_transport'], inplace=True)

    # Replace 'None' string with actual NaN
    df.replace('None', np.nan, inplace=True)

    # Standardize categorical string values
    df.replace('ARTS', 'Arts', inplace=True)
    df.replace('CLUBS', 'Clubs', inplace=True)
    df.replace('SPORTS', 'Sports', inplace=True)
    df.replace('N', 'No', inplace=True)
    df.replace('Y', 'Yes', inplace=True)
    
    print("\nDataFrame after initial corrections (age filter, column drops, value replacements):")
    print(df.sample(5))

    # Handle Null Values
    print("\n--- Handling Null Values ---")
    median_imputer = SimpleImputer(strategy='median')
    df["attendance_rate"] = median_imputer.fit_transform(df[["attendance_rate"]])
    df['CCA'] = df['CCA'].fillna(df['CCA'].mode()[0])
    df.dropna(subset=['final_test'], axis=0, inplace=True) # Drop rows where 'final_test' is NaN
    
    print("\nDataFrame after handling null values:")
    df.info()

    print("\nUnique values for 'tuition' after cleaning:", df['tuition'].unique())

    # Feature Engineering
    print("\n--- Feature Engineering ---")
    df['sleep_time'] = pd.to_datetime(df['sleep_time'], format='%H:%M')
    df['wake_time'] = pd.to_datetime(df['wake_time'], format='%H:%M')

    df['total_sleep_hours'] = (df['wake_time'] - df['sleep_time']).dt.total_seconds() / 3600
    df['total_sleep_hours'] = df['total_sleep_hours'].apply(lambda x: x + 24 if x < 0 else x)
    df.drop(['sleep_time', 'wake_time'], axis=1, inplace=True)
    
    df['study_efficiency'] = df['hours_per_week'] * df['attendance_rate']
    df.drop(['hours_per_week'], axis=1, inplace=True)

    df['total_students'] = df['n_male'] + df['n_female']
    df.drop(['n_male', 'n_female'], axis=1, inplace=True)

    df['healthy_routine_score'] = df['total_sleep_hours'] * df['attendance_rate']
    df.drop(columns=['total_sleep_hours', 'attendance_rate'], inplace=True)
    
    print("\nDataFrame after feature engineering. Sample (5 rows):")
    print(df.sample(5))

    return df

def encode_categorical_data(df):
    """
    Encodes categorical features using One-Hot Encoding.

    Args:
        df (pd.DataFrame): DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded numerical features.
        sklearn.compose.ColumnTransformer: The fitted ColumnTransformer for later use.
    """
    print("\n--- Encoding Categorical Data ---")
    onehot_cols = ['CCA', 'gender', 'direct_admission', 'tuition', 'learning_style']

    ct = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)
        ],
        remainder='passthrough'
    )

    df_encoded = ct.fit_transform(df)
    feature_names = ct.get_feature_names_out()
    df_processed = pd.DataFrame(df_encoded, columns=feature_names)
    df_processed.columns = df_processed.columns.str.replace('onehot__', '').str.replace('remainder__', '')
    
    print("\nDataFrame after One-Hot Encoding. Columns:")
    print(df_processed.columns)
    print("\nSample (5 rows):")
    print(df_processed.sample(5))
    
    return df_processed, ct

# --- 4. Data Splitting and Scaling ---

def split_and_scale_data(df, target_column='final_test'):
    """
    Separates features and target, splits data into train/test sets, and scales continuous features.

    Args:
        df (pd.DataFrame): Processed DataFrame.
        target_column (str): Name of the target column.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("\n--- Separating Features (X) and Target (y) ---")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("\nFeatures (X) sample (5 rows):")
    print(X.sample(5))
    print("\nTarget (y) sample (5 rows):")
    print(y.sample(5))

    print("\n--- Splitting Data into Training and Test Set ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    print("\n--- Standardizing the Data ---")
    # Identify continuous columns for scaling
    # Ensure these columns exist after encoding and feature engineering
    # This list should be updated if feature engineering changes column names or types
    continuous_cols = [
        'number_of_siblings', 'study_efficiency', 'total_students', 'healthy_routine_score'
    ]
    
    # Filter continuous_cols to ensure they are present in X_train
    actual_continuous_cols = [col for col in continuous_cols if col in X_train.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if actual_continuous_cols:
        X_train_scaled[actual_continuous_cols] = scaler.fit_transform(X_train[actual_continuous_cols])
        X_test_scaled[actual_continuous_cols] = scaler.transform(X_test[actual_continuous_cols])
    else:
        print("Warning: No continuous columns found for scaling. X_train_scaled and X_test_scaled are copies of original X_train/X_test.")

    print("\nX_train_scaled sample (5 rows):")
    print(X_train_scaled.sample(5))
    print("\nX_test_scaled sample (5 rows):")
    print(X_test_scaled.sample(5))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# --- 5. Model Training and Evaluation ---

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a regression model using R², MSE, RMSE, and MAE.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        model_name (str): Name of the model for printing.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\nEvaluation Metrics for {model_name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE     : {mse:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"MAE     : {mae:.4f}")

    return {"R² Score": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

def plot_predictions(y_true, y_pred, title):
    """
    Plots actual vs. predicted values for a regression model.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, label="Predicted")
    sns.lineplot(x=y_true, y=y_true, color='red', label="Ideal")
    plt.xlabel("Actual Final Test Score")
    plt.ylabel("Predicted Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_evaluate_linear_regression(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Multiple Linear Regression model.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.

    Returns:
        tuple: (LinearRegression model, np.array of predictions, dict of metrics)
    """
    print("\n--- Multiple Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    print("\nLinear Regression Predictions (sample):")
    print(y_pred_lr[:5]) # Print first 5 predictions
    
    metrics_lr = evaluate_model(y_test, y_pred_lr, "Multiple Linear Regression")
    plot_predictions(y_test, y_pred_lr, "Multiple Linear Regression: Predictions vs Actual")
    return lr_model, y_pred_lr, metrics_lr

def train_and_evaluate_polynomial_regression(X_train_scaled, y_train, X_test_scaled, y_test, degree=3):
    """
    Trains and evaluates a Polynomial Regression model.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
        degree (int): Degree of polynomial features.

    Returns:
        tuple: (LinearRegression model, np.array of predictions, dict of metrics)
    """
    print(f"\n--- Polynomial Regression (Degree {degree}) ---")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)
    
    print("\nPolynomial Regression Predictions (sample):")
    print(y_pred_poly[:5]) # Print first 5 predictions
    
    metrics_poly = evaluate_model(y_test, y_pred_poly, f"Polynomial Regression (Degree {degree})")
    plot_predictions(y_test, y_pred_poly, f"Polynomial Regression (Degree {degree}): Predictions vs Actual")
    return poly_reg, y_pred_poly, metrics_poly

# --- Main Execution Block ---

def main_analysis(db_path='score.db', table_name='score', target_column='final_test', poly_degree=3):
    """
    Orchestrates the complete EDA, preprocessing, and model training/evaluation pipeline.
    """
    print("Starting EDA and Regression Analysis...")

    # 1. Connect to Database and Load Data
    conn, cursor = connect_to_database(db_path)
    df = load_data_from_table(conn, table_name)

    # 2. Explore Data (optional to run all plots in a modular script, can comment out)
    # explore_data(df.copy()) # Pass a copy if you don't want explore_data to modify the original df

    # 3. Clean and Transform Data
    df_cleaned = clean_and_transform_data(df.copy()) # Pass a copy

    # 4. Encode Categorical Data
    df_encoded, ct_transformer = encode_categorical_data(df_cleaned.copy())

    # 5. Split and Scale Data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df_encoded.copy(), target_column)

    # 6. Train and Evaluate Multiple Linear Regression
    lr_model, y_pred_lr, metrics_lr = train_and_evaluate_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)

    # 7. Train and Evaluate Polynomial Regression
    poly_reg_model, y_pred_poly, metrics_poly = train_and_evaluate_polynomial_regression(X_train_scaled, y_train, X_test_scaled, y_test, degree=poly_degree)

    # --- Identify the best model ---
    results_df = pd.DataFrame([
        {"Model": "Multiple Linear Regression", **metrics_lr},
        {"Model": f"Polynomial Regression (Degree {poly_degree})", **metrics_poly}
    ]).sort_values(by='R² Score', ascending=False).reset_index(drop=True)

    print("\n--- Model Comparison ---")
    print(results_df)

    # Visualizing R² scores
    plt.figure(figsize=(6, 2.5))
    sns.barplot(x='R² Score', y='Model', data=results_df, palette='viridis')
    plt.xlabel('R² Score')
    plt.ylabel('Regression Model')
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    best_model_name = results_df.iloc[0]["Model"]
    best_r2_score = results_df.iloc[0]["R² Score"]
    best_mae = results_df.iloc[0]["MAE"]
    best_mse = results_df.iloc[0]["MSE"]
    best_rmse = results_df.iloc[0]["RMSE"]

    print(f"\nBest Model: {best_model_name}")
    print(f"R² Score : {best_r2_score:.4f}")
    print(f"MAE      : {best_mae:.4f}")
    print(f"MSE      : {best_mse:.4f}")
    print(f"RMSE     : {best_rmse:.4f}")

    # Close database connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("\nDatabase connection closed.")
    print("Analysis complete.")

if __name__ == "__main__":
    # Example usage:
    # Ensure 'score.db' is in the same directory as this script.
    # You can change 'poly_degree' to experiment with different polynomial complexities.
    main_analysis(db_path='score.db', table_name='score', poly_degree=3)