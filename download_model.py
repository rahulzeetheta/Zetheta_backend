# download_model.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('./chatbot_model/all-MiniLM-L6-v2')
