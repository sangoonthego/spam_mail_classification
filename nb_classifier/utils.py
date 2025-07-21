from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le