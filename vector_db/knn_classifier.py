from collections import Counter
from vector_index import search_index

def predict_label(index, vector, labels, k=3):
    indices, distances = search_index(index, vector, k)
    nearest_labels = [labels[i] for i in indices]
    vote = Counter(nearest_labels).most_common(1)[0][0]
    return vote