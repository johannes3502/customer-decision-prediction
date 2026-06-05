import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import warnings
# Ignoriert alle zukünftigen Warnungen (sehr nützlich für saubere Ausgaben)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*backward compatibility.*")

# ============================================================
# Globale Einstellungen
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
final_results = []

def train_for_summary(df, target_column, dataset_name):
    print(f"\nVerarbeite: {dataset_name}...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == "object" or str(y.dtype) == "category":
        y = pd.factorize(y)[0]

    # Anzahl der Features vor dem Split ermitteln
    num_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist() + X.select_dtypes(include=['bool']).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
    ])

    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Hyperparameter-Raum
    param_distributions = {
        "model__max_iter": [100, 300],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [3, 10, 15]
    }

    # Suche (Wir nutzen Accuracy als Scoring, um mit der Tabelle des Kollegen identisch zu sein)
    search = RandomizedSearchCV(clf, param_distributions, n_iter=10, cv=5, scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_

    # Metriken berechnen
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    valid_acc = accuracy_score(y_test, best_model.predict(X_test))
    mean_cv_score = search.best_score_

    # Ergebnis im gewünschten Format speichern
    return {
        "Dataset": dataset_name,
        "Features": num_features,
        "Selected Features": num_features, # Bei Boosting meist alle, sofern kein Vor-Filter genutzt wird
        "Train Accuracy": round(train_acc, 4),
        "Valid Accuracy": round(valid_acc, 4),
        "Mean CV Score": round(mean_cv_score, 4)
    }

# Datensätze laden und ausführen
datasets = [
    ("telco_clean.csv", "Churn", "Telco Churn"),
    ("ecom_clean.csv", "Revenue", "Online Shoppers"),
    ("bank_clean.csv", "y", "Bank Marketing")
]

for file, target, name in datasets:
    try:
        df = pd.read_csv(file)
        final_results.append(train_for_summary(df, target, name))
    except Exception as e:
        print(f"Fehler bei {name}: {e}")

# Erstellung der finalen Tabelle
summary_df = pd.DataFrame(final_results)

# Spaltenreihenfolge
columns_order = ["Dataset", "Features", "Selected Features", "Train Accuracy", "Valid Accuracy", "Mean CV Score"]
summary_df = summary_df[columns_order]

print("\nModel Performance Summary (Gradient Boosting - Quirin):")
print("=" * 90)
print(summary_df.to_string(index=False))
print("=" * 90)

# Export für das Team
summary_df.to_csv("summary_gradient_boosting.csv", index=False)