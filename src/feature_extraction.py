from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
