import torch
import numpy as np

import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    prompt = f"query: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().numpy()[0].astype(np.float32)


    