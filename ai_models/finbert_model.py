from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load FinBERT model and tokenizer once
MODEL_NAME = "ProsusAI/finbert"

print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("FinBERT model loaded.")

def get_finbert_sentiment(text: str):
    if not text or not isinstance(text, str) or not text.strip():
        return {"label": None, "positive": None, "negative": None, "neutral": None}

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].tolist()

    negative, neutral, positive = probs
    scores = {"Positive": positive, "Neutral": neutral, "Negative": negative}
    sentiment_label = max(scores, key=scores.get)

    return {
        "label": sentiment_label,
        "positive": positive,
        "neutral": neutral,
        "negative": negative
    }
