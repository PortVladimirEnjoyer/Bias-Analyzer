from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import joblib
from textblob import TextBlob
import textstat

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load("model/bias_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

def extract_article(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.find("title").text if soup.find("title") else "No title"
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    text = re.sub(r"\\s+", " ", text)
    return title, text

def analyze_bias(text):
    # Compute extra features
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    readability = textstat.flesch_reading_ease(text)
    
    # Predict bias
    X_vec = vectorizer.transform([text])
    bias = model.predict(X_vec)[0]
    
    return {
        "bias": bias,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "readability": round(readability, 2),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        try:
            title, text = extract_article(url)
            result = analyze_bias(text)
            return render_template("result.html", title=title, url=url, result=result)
        except Exception as e:
            return f"Error: {e}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
