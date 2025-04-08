import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Sentiment Mapping
LABELS = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}

def classify_text(text):
    """
    Classifies the given text as 'very negative', 'negative', 'neutral', 'positive', or 'very positive'.
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Model Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)

    # Get the label with the highest score
    sentiment_score = torch.argmax(probabilities, dim=1).item()
    return LABELS[sentiment_score]

# Testing the model
if __name__ == "__main__":
    sample_reviews = [
        "This movie was amazing! Highly recommend it.",
        "Terrible acting and boring plot. Waste of time.",
        "It was okay, nothing special.",
        "Absolutely fantastic! The best movie I've seen all year.",
        "I wouldn't watch it again. Disappointing."
    ]

    for review in sample_reviews:
        print(f"Review: {review} -> Sentiment: {classify_text(review)}")

