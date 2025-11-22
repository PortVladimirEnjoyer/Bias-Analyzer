# train_bias_model.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("bias_clean.csv")

# Clean and balance dataset
df = df[df["page_text"].str.len() > 300]
df = df.groupby('bias', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 500), random_state=42))
print(df['bias'].value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(df["page_text"], df["bias"], test_size=0.2, random_state=42)

# Light TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Use lightweight model
model = MultinomialNB()  #naive bayes classifier 
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "model/bias_model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")
print("✅ Model and vectorizer saved in /model/")

