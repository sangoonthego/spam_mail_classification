import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extraction import get_vectorizer

texts = ["buy cheap meds", "buy now", "cheap offer limited", "hi friend"]
X, vectorizer = get_vectorizer(texts)

print(X.shape) 
print(vectorizer.get_feature_names_out())