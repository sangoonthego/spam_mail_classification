import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from vector_db.vector_index import search_index

def predict_label(index, vector, labels, k=3):
    try:
        indices, distances = search_index(index, vector, k)
    except Exception as e:
        print(f"Error in search_index: {e}")
        return None
    if not indices.any():
        print("No nearest neighbors found.")
        return None
    nearest_labels = [labels[i] for i in indices if i < len(labels)]
    if not nearest_labels:
        print("No valid labels found for nearest neighbors.")
        return None
    vote = Counter(nearest_labels).most_common(1)[0][0]
    return vote
