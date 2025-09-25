from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Initialize FastAPI
app = FastAPI(title="Sentiment Analysis API with Confidence")

# Load Model & Vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Request Schema
class Review(BaseModel):
    text: str

# Prediction Endpoint
@app.post("/predict")
def predict_sentiment(review: Review):
    cleaned_text = preprocess_text(review.text)
    X = vectorizer.transform([cleaned_text])

    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        confidence = float(max(prob))  # highest probability
        prediction = model.predict(X)[0]
    else:  # For SVM without probability, use decision function
        decision = model.decision_function(X)[0]
        confidence = float(abs(decision) / (1 + abs(decision)))  # normalized
        prediction = model.predict(X)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment, "confidence": round(confidence, 3)}
