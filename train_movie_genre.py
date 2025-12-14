# train_movie_genre.py
import pandas as pd
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (runs only first time)
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Load dataset (KEEP THIS FILE NAME SAME)
df = pd.read_csv("Movie Dataset.csv")

# Rename columns to simple names
df = df.rename(columns={
    "Overview": "plot",
    "Genre": "genre"
})

# Remove missing values
df = df.dropna(subset=["plot", "genre"])

# Clean plot text
df["plot"] = df["plot"].apply(clean_text)

# If multiple genres exist, keep only the first
df["genre"] = df["genre"].apply(lambda x: str(x).split(",")[0].strip().lower())

# Keep only top genres to reduce bias
top_genres = df["genre"].value_counts().head(6).index
df = df[df["genre"].isin(top_genres)]

# Features and target
X = df["plot"]
y = df["genre"]

# Train-test split (NO stratify to avoid errors)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "model_movie.pkl")
print("\nModel saved as model_movie.pkl")