## Product Review Sentiment Analysis
### Project Overview
This project predicts the sentiment of Amazon product reviews as Positive or Negative using Natural Language Processing (NLP) and machine learning.

Dataset: Amazon product reviews (original source: www.kaggle.com/datasets/mahmudulhaqueshawon/amazon-product-reviews)

Task: Binary sentiment classification

Features: Preprocessed review text (cleaned, lowercased, stopwords removed, and stemmed)

Target: Sentiment (0 = Negative, 1 = Positive)

### Data
product_reviews.csv — Cleaned dataset with updated column names
- Review — Text of the review
- Sentiment — Label (0 or 1)

### Modeling
Text features are transformed using TF-IDF vectorization (unigrams + bigrams, max 5000 features).
Models trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM) - best performing
- Random Forest

Imbalance handled using class weighting for better minority class prediction.
Final model trained on train + validation data and evaluated on test set.

### Performance
Accuracy: 90.2%
