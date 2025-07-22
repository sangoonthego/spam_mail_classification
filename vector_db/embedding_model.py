import torch
import numpy as np
import os

# Cho phép cấu hình động đường dẫn cache HuggingFace
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_HOME

from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None

def get_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_HOME)
        model = AutoModel.from_pretrained(model_name, cache_dir=HF_HOME)
    prompt = f"query: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().numpy()[0].astype(np.float32)


    