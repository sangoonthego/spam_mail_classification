from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text
from src.feature_extraction import get_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
import joblib

df = pd.read_csv("data/spam.csv")
df["cleaned"] = df["text"].apply(clean_text)

X, vectorizer = get_vectorizer(df["cleaned"])
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")