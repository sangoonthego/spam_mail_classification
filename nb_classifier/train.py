import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nb_classifier.preprocess import preprocess_text

dataset_path = "data/spam.csv"

def load_data():
    df = pd.read_csv(dataset_path)
    messages = df["Message"].tolist()
    labels = df["Label"].tolist()
    return messages, labels

def create_dictionary(messages):
    dictionary = list(set(word for message in messages for word in message))
    return dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

if __name__ == "__main__":
    messages, labels = load_data()
    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)
    X = np.array([create_features(tokens, dictionary) for tokens in messages])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Training Completed.")

    with open("pickle/model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open("pickle/dictionary.pkl", "wb") as file:
        pickle.dump(dictionary, file)
    with open("pickle/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
