import faiss
import numpy as np
import pickle
import os

# faiss index
# create index to save vector
def create_index(dim):
    index = faiss.IndexFlatIP(dim)
    return index

def add_to_index(index, vectors):
    vectors = np.array(vectors).astype(np.float32)
    index.add(vectors)

def save_index(index, index_path):
    faiss.write_index(index, index_path)

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(index_path)

def save_labels(labels, label_path):
    with open(label_path, "wb") as file:
        pickle.dump(labels, file)

def load_labels(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    with open(label_path, "rb") as file:
        labels = pickle.load(file)
    return labels

# find knn based on inner product
def search_index(index, vector, k):
    if index.ntotal < k:
        raise ValueError(f"Index only contains {index.ntotal} vectors, but k={k}")
    vector = vector.reshape(1, -1).astype(np.float32)
    distance, idx = index.search(vector, k)
    return idx[0], distance[0]

