
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load dataset
file_path = r"C:\Users\DELL\OneDrive\Sunil files\Smart Transformer Digital Twin\data\df_entrenamiento_final.csv"

df = pd.read_csv(file_path)

target = "burned_transformers"
threshold = 0.10

print("--- DATASET INFORMATION ---")
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Target distribution:")
print(df[target].value_counts())

# 2. Split chronologically: train on 2019, test on 2020
train_df = df[df["year"] == 2019].copy()
test_df = df[df["year"] == 2020].copy()

if train_df.empty or test_df.empty:
    raise ValueError("The dataset must contain records from 2019 and 2020.")

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# Keep the exact feature order for the website
feature_columns = X_train.columns.tolist()
X_test = X_test[feature_columns]

print("\n--- TRAINING INFORMATION ---")
print("Training year: 2019")
print("Training records:", len(X_train))
print("Testing year: 2020")
print("Testing records:", len(X_test))
print("Features:", len(feature_columns))

# 3. Train the model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 4. Evaluate on 2020 data
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\n--- MODEL EVALUATION ---")
print("Threshold:", threshold)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 5. Show probability details
print("\n--- PROBABILITY CHECK ---")
print("Minimum probability:", probabilities.min())
print("Maximum probability:", probabilities.max())
print("Average probability:", probabilities.mean())
print("First 10 probabilities:")
print(probabilities[:10])
print("\n--- SAMPLE INPUT RECORDS ---")
print(X_test.head(5).to_string())

# 6. Show feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print("\n--- TOP 10 IMPORTANT FEATURES ---")
print(importance.head(10).to_string())

# 7. Save the trained model
joblib.dump(model, "transformer_model.pkl")

print("\nModel saved successfully as transformer_model.pkl")
print("Training completed!")
