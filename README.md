# Bias-Analyzer






A lightweight Flask web app that extracts article text from a URL, computes simple linguistic/readability features, and predicts bias using a TF-IDF + MultinomialNB classifier.

Features
- Web UI to input article URL and view bias prediction plus polarity, subjectivity, and readability.
- Training script to rebuild or improve the model.
- Tests, CI, and Docker support.
- Guidance for safe storing and distributing models (use Git LFS or external object storage).

Quickstart (local)
1. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Place your dataset `bias_clean.csv` in repo root (see Data section).

4. Train a model (optional):
   python train.py --input bias_clean.csv --outdir model

5. Run the app:
   python app.py

6. Visit http://127.0.0.1:5000

Data
- Place a CSV named `bias_clean.csv` with columns at least: `page_text` and `bias`.
- For model artifacts, avoid committing large binary files. Use Git LFS or upload models to a release or S3 and add download scripts.

Model management recommendations
- Commit only small test/dummy models. For real models, enable Git LFS or store them in releases/S3/Google Cloud Storage.
- Add a `model/README.md` describing how to recreate or download model artifacts.

Development
- Run tests: `make test`
- Lint: `make lint`
- Format: `make format`



License
- MIT. See LICENSE.
