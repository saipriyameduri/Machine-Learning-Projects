"""
Fake News Detection
--------------------
Classifies news articles as REAL or FAKE using TF-IDF text features
and a Logistic Regression classifier.

Dataset: fake_and_real_news_dataset.csv
(~6,300 labeled news articles — title, text, label)
Source: https://github.com/GeorgeMcIntire/fake_real_news_dataset
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("news.csv")

# Normalize label column to 0/1 (dataset uses "FAKE"/"REAL" strings)
df["label"] = df["label"].astype(str).str.strip().str.upper()
df = df[df["label"].isin(["FAKE", "REAL"])]
df["label_num"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Combine title + body text for richer features
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

print(f"Total articles: {len(df)}")
print(df["label"].value_counts(), "\n")

# ---------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

# ---------------------------------------------------------
# 3. TF-IDF vectorization
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,      # ignore overly common terms
    max_features=5000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Train Logistic Regression model
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Evaluation Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}\n")

print("Confusion Matrix (rows=actual, cols=predicted) [FAKE, REAL]:")
print(confusion_matrix(y_test, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# ---------------------------------------------------------
# 6. Save model + vectorizer for reuse (no retraining needed)
# ---------------------------------------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nSaved model -> fake_news_model.pkl")
print("Saved vectorizer -> tfidf_vectorizer.pkl")

# ---------------------------------------------------------
# 7. Quick manual test
# ---------------------------------------------------------
def predict_news(text: str) -> str:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "REAL" if pred == 1 else "FAKE"

sample = "Scientists confirm water discovered on newly explored exoplanet"
print(f"\nSample prediction for: \"{sample}\"\n-> {predict_news(sample)}")
