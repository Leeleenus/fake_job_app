# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ========== CONFIG ==========
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/fake_job_detector.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

# Ensure your dataset has these columns
# df['job_description'], df['label']
text_column = "job_description"
label_column = "label"

# Drop missing values
df = df.dropna(subset=[text_column, label_column])

# ========== PREPARE DATA ==========
X = df[text_column].astype(str)
y = df[label_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== VECTORIZE TEXT ==========
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ========== TRAIN MODEL ==========
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# ========== EVALUATE ==========
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ========== SAVE MODEL & VECTORIZER ==========
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Vectorizer saved to {VECTORIZER_PATH}")
