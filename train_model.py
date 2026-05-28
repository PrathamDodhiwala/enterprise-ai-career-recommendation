import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import joblib

np.random.seed(42)

size = 5000

career_data = pd.DataFrame(
    {
        "Python": np.random.randint(0, 10, size),
        "MachineLearning": np.random.randint(0, 10, size),
        "SQL": np.random.randint(0, 10, size),
        "PowerBI": np.random.randint(0, 10, size),
        "Communication": np.random.randint(0, 10, size),
        "Projects": np.random.randint(1, 15, size),
    }
)

career_labels = []

for i in range(size):

    python_skill = career_data.loc[i, "Python"]
    ml_skill = career_data.loc[i, "MachineLearning"]
    sql_skill = career_data.loc[i, "SQL"]
    powerbi_skill = career_data.loc[i, "PowerBI"]
    communication_skill = career_data.loc[i, "Communication"]
    projects = career_data.loc[i, "Projects"]

    if (
        python_skill >= np.random.randint(6, 9)
        and ml_skill >= np.random.randint(6, 9)
        and projects >= np.random.randint(5, 8)
    ):
        career_labels.append("Data Scientist")

    elif python_skill >= np.random.randint(7, 10) and ml_skill >= np.random.randint(
        7, 10
    ):
        career_labels.append("ML Engineer")

    elif sql_skill >= np.random.randint(6, 9) and powerbi_skill >= np.random.randint(
        6, 9
    ):
        career_labels.append("Data Analyst")

    elif communication_skill >= np.random.randint(
        6, 9
    ) and powerbi_skill >= np.random.randint(4, 8):
        career_labels.append("Business Analyst")

    else:
        career_labels.append("Software Engineer")

career_data["Career"] = career_labels

label_encoder = LabelEncoder()

career_data["Career"] = label_encoder.fit_transform(career_data["Career"])

X = career_data.drop("Career", axis=1)
y = career_data["Career"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300, max_depth=15, class_weight="balanced", random_state=42
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "career_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

career_data.to_csv("career_dataset.csv", index=False)

feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
)

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

print("\nTraining Completed Successfully")

print("\nFiles Saved:")
print("- career_model.pkl")
print("- scaler.pkl")
print("- label_encoder.pkl")
print("- career_dataset.csv")

print("\nAI Career Recommendation Model Ready")
