# -----------------------------------------------------------
# Fault classifier for home appliances using extracted features
# Model: Random Forest with 5-fold stratified cross-validation
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "2-washing_machines")

METADATA_COLS = [
    "cycle_id", "begin_ts", "end_ts", "brand", "model",
    "program", "temperature", "spin_speed", "load",
    "fault_condition", "notes",
]

FAULT_LABELS = {"Working": 0, "Heating": 1, "Bearings": 2, "Motor": 3}


def load_dataset():
    features_df = pd.read_csv(os.path.join(DATA_DIR, "WM_ExtractedFeatures.csv"))
    metadata_df = pd.read_csv(
        os.path.join(DATA_DIR, "washing_machine_metadata.csv"),
        header=None,
        names=METADATA_COLS,
    )
    metadata_df = metadata_df.drop_duplicates(subset="cycle_id")
    merged = features_df.merge(
        metadata_df[["cycle_id", "fault_condition", "brand", "model"]],
        left_on="Id",
        right_on="cycle_id",
        how="inner",
    )
    return merged


def train_and_evaluate(merged_df, plot=True):
    feature_cols = [
        c for c in merged_df.columns
        if c not in ("Id", "cycle_id", "fault_condition", "brand", "model")
    ]

    X = merged_df[feature_cols].values
    y_raw = merged_df["fault_condition"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1_macro")
    print(f"\nCross-validated F1 (macro, 5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    print("\nClassification Report (training set):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    if plot:
        cm = confusion_matrix(y, y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        axes[1].barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#7e57c2")
        axes[1].set_title("Top 20 Feature Importances")
        axes[1].set_xlabel("Importance")

        plt.tight_layout()
        plt.show()

    return clf, scaler, le, feature_cols, cv_scores


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset: {len(df)} cycles, {df['fault_condition'].value_counts().to_dict()}")
    clf, scaler, le, feature_cols, scores = train_and_evaluate(df, plot=True)
