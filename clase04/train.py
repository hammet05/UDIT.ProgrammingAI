import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

COLUMNAS = ["age", "resting_bp", "cholesterol", "max_heart_rate",
            "exercise_angina", "st_depression"]

df = pd.read_csv("heart_disease_prediction_2026.csv")
X, y = df[COLUMNAS], df["heart_disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

print(f"Precisión en test: {modelo.score(X_test, y_test):.3f}")
joblib.dump(modelo, "model.joblib")