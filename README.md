Sentiment Analysis of Restaurant Reviews
This project classifies restaurant reviews as positive (1) or negative (0) using machine learning models.

The project involves the following steps:

1- Dataset: RestaurantReviews_train.tsv (training) RestaurantReviews_test.tsv (prediction)

2- Steps:

Preprocessing: Clean text by removing unnecessary characters, converting to lowercase, stemming, and removing stop words.
Feature Extraction: Use Bag-of-Words with CountVectorizer.
Model Training: Train Naive Bayes, Logistic Regression, and Random Forest models.
Evaluation: Select the best model (Logistic Regression in our case).
Prediction: Use the trained model to classify new reviews using prediction dataset.
