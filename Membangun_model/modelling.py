import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# SETUP DAGSHUB + MLFLOW
# =========================
dagshub.init(
    repo_owner='zulfiana92',
    repo_name='Eksperimen_SML_Zulfiana-Majid',
    mlflow=True
)

mlflow.set_experiment("heart_disease_random_forest")

# =========================
# LOAD DATA
# =========================
def load_data():
    train_data = pd.read_csv(
        "preprocessing/heart_disease_preprocessing/train_data.csv"
    )
    test_data = pd.read_csv(
        "preprocessing/heart_disease_preprocessing/test_data.csv"
    )
    X_train = train_data.drop(columns='target')
    y_train = train_data['target']
    X_test = test_data.drop(columns='target')
    y_test = test_data['target']
    return X_train, X_test, y_train, y_test

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Autolog MLflow (Basic)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf_autolog"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall   :", recall_score(y_test, y_pred))
        print("F1 Score :", f1_score(y_test, y_pred))

    print("✅ modelling.py selesai. Cek MLflow di DagsHub!")
