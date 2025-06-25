import joblib
from src.preprocessing import clean_text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_email(text: str) -> str:
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return "Spam" if pred == 1 else "Ham"