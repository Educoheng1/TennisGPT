import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "momentum_samples.csv")
MODEL_PATH = os.path.join(BASE_DIR, "momentum_model.joblib")


def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # Features (X) and target (y)
    feature_cols = ["set_number", "home_games", "away_games", "event_tag", "set_score_summary"]
    target_col = "momentum_label"

    X = df[feature_cols]
    y = df[target_col]

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, 
    )

    # 3) Preprocess:
    # - Numeric: set_number, home_games, away_games
    # - Categorical: event_tag, set_score_summary
    numeric_features = ["set_number", "home_games", "away_games"]
    categorical_features = ["event_tag", "set_score_summary"]

    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 4) Model
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # 5) Pipeline: preprocessing + model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # 6) Train
    model.fit(X_train, y_train)

    # 7) Evaluate
    y_pred = model.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 8) Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved momentum model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
