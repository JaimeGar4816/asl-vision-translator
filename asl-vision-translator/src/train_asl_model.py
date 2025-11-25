import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "asl_data.npz"
MODEL_FILE = "asl_model.joblib"


def main():
    data = np.load(DATA_FILE)
    X = data["X"]
    y = data["y"]

    print("Loaded data:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)
    print("Labels found:", np.unique(y))

    # Encode labels (letters -> 0,1,2,...)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    clf.fit(X_train, y_train)
    print("Done training.")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump({"model": clf, "label_encoder": le}, MODEL_FILE)
    print(f"\nSaved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
