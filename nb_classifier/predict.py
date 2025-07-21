import numpy as np

from preprocess import preprocess_text
from train import create_feature

def predict(text, model, dictionary, label_encoder):
    tokens = preprocess_text(text)
    features = create_feature(tokens, dictionary).reshape(1, -1)
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]