import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# LOAD DATA
# =========================
def load_data(path):

    df = pd.read_csv(path)

    return df


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):

    # Pisahkan fitur dan target
    X = df.drop(columns='target')
    y = df['target']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ubah menjadi DataFrame
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

    # Membuat folder output
    os.makedirs(
        "heart_disease_preprocessing",
        exist_ok=True
    )

    # Save training data
    train_data.to_csv(
        "heart_disease_preprocessing/train_data.csv",
        index=False
    )

    # Save testing data
    test_data.to_csv(
        "heart_disease_preprocessing/test_data.csv",
        index=False
    )

    print("Preprocessing berhasil disimpan.")


# =========================
# MAIN PROGRAM
# =========================
if __name__ == "__main__":

    # Path dataset
    dataset_path = "heart_disease.csv"

    # Load dataset
    df = load_data(dataset_path)

    # Preprocessing
    train_data, test_data = preprocess_data(df)

    # Save hasil preprocessing
    save_data(train_data, test_data)
