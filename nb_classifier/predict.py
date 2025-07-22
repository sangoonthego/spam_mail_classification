import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__), "..")))

import numpy as np
import pickle

from nb_classifier.preprocess import preprocess_text
from nb_classifier.train import create_features

with open("pickle/model.pkl", "rb") as file:
    model = pickle.load(file)
with open("pickle/dictionary.pkl", "rb") as file:
    dictionary = pickle.load(file)
with open("pickle/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

def predict(text_list):
    all_features = []
    for text in text_list:
        tokens = preprocess_text(text)
        features = create_features(tokens, dictionary)
        all_features.append(features)
    
    X = np.array(all_features)
    predictions = model.predict(X)
    pred_classes = label_encoder.inverse_transform(predictions)
    return pred_classes

text_list = [
    "You have won a $1000 gift card! Claim now!",
    "I'll see you tomorrow morning.",
    "Exclusive offer just for you!",
    "Can you send me the report before noon?",
]

pred_classes = predict(text_list)

for text, label in zip(text_list, pred_classes):
    print(f"Message: {text} -> Prediction: {label}")