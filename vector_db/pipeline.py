import pandas as pd
from embedding_model import get_embedding
from vector_index import create_index, add_to_index, save_index, save_labels, load_index, load_labels
from knn_classifier import predict_label

dataset_path = "data/spam.csv"

index_file = "vector_db/index.faiss"
label_file = "vector_db/labels.pkl"

def train_pipeline(dataset_path):
    df = pd.read_csv(dataset_path, on_bad_lines='skip', engine='python')
    messages = df["Message"].tolist()
    labels = df["Label"].tolist()
    vectors = [get_embedding(message) for message in messages]
    index = create_index(dim=len(vectors[0]))
    add_to_index(index, vectors)
    save_index(index, index_file)
    save_labels(labels, label_file)
    print("Saved Successfully!!!")

def predict_pipeline(text_list, k=3):
    index = load_index(index_file)
    labels = load_labels(label_file)
    all_predictions = []

    for text in text_list:
        vector = get_embedding(text)
        prediction = predict_label(index, vector, labels, k)
        all_predictions.append(prediction)

    return all_predictions

text_list = [
    "Congratulations, you've won a prize!",
    "Hello, are you free this evening?",
    "Free entry in a weekly competition!"
]

pred_classes = predict_pipeline(text_list, k=3)
for text, label in zip(text_list, pred_classes):
    print(f"Message: {text} -> Prediction: {label}")

if __name__ == "__main__":
    train_pipeline(dataset_path)
