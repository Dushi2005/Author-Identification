import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load your dataset
df = pd.read_csv("authors_dataset.csv")

# Rename columns if needed
df.columns = ['work', 'author']  # Make sure these match your dataset

# Drop rows with missing values
df.dropna(subset=['work', 'author'], inplace=True)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['work'])
y = df['author']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "author_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved.")
