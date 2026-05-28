import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path)
    return df

# =========================
# HANDLE OUTLIERS
# =========================
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower_bound) &
            (df[col] <= upper_bound)
        ]

    return df

# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):

    # =========================
    # HANDLE MISSING VALUES
    # =========================
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    # =========================
    # HANDLE OUTLIERS
    # =========================
    continuous_features = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak'
    ]

    df = remove_outliers(df, continuous_features)

    # =========================
    # SPLIT FEATURE & TARGET
    # =========================
    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    train_data['target'] = y_train.values

    test_data = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    test_data['target'] = y_test.values

    return train_data, test_data

# =========================
# SAVE DATA
# =========================
def save_data(train_data, test_data):

    os.makedirs(
        "preprocessing/heart_disease_preprocessing",
        exist_ok=True
    )

    train_data.to_csv(
        "preprocessing/heart_disease_preprocessing/train_data.csv",
        index=False
    )

    test_data.to_csv(
        "preprocessing/heart_disease_preprocessing/test_data.csv",
        index=False
    )

    print("Preprocessing berhasil disimpan.")

# =========================
# MAIN PROGRAM
# =========================
if __name__ == "__main__":

    dataset_path = "heart_disease_raw/heart_disease.csv"

    df = load_data(dataset_path)

    train_data, test_data = preprocess_data(df)

    save_data(train_data, test_data)