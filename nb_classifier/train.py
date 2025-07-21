import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text

dataset_path = "data/spam.csv"

def load_data():
    df = pd.read_csv(dataset_path)
    messages = df["Message"].tolist()
    labels = df["Label"].tolist()
    return messages, labels

def create_dictionary(messages):
    dictionary = list(set(word for message in messages for word in message))
    return dictionary

def create_feature(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

if __name__ == "__main__":
    messages, labels = load_data()
    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)
    X = np.array([create_feature(tokens, dictionary) for tokens in messages])

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Training Completed.")