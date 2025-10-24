*SENTIMENT ANALYSIS OF SOCIAL MEDIA POSTS ON A BRAND AND ITS PRODUCTS USING NLP TECHNIQUES*


iPhone 17 Reddit Sentiment Analysis

Overview

This project performs sentiment analysis on Reddit posts about the iPhone 17 using Natural Language Processing (NLP) techniques. It includes lexicon-based methods (VADER, TextBlob), machine learning models (Naive Bayes, Logistic Regression, Linear SVM), and a deep learning LSTM model. The goal is to analyze public opinion and classify posts as positive or negative.


Dataset

Source: Reddit iPhone 17 Posts CSV

Columns include: id, title, author, created_utc, score, upvote_ratio, subreddit, permalink, url

The title column is used for sentiment analysis, it has the reviews left by the users.


Features

Data Cleaning & Preprocessing: Lowercasing, URL/mention removal, punctuation removal, stopword removal, lemmatization.

Lexicon-Based Sentiment: VADER and TextBlob scoring

Machine Learning Models: Naive Bayes, Logistic Regression, Linear SVM with TF-IDF features

Deep Learning Model: LSTM with Embedding layer for contextual sentiment analysis

Visualization: Model accuracy comparison plots

Results: Saved as iphone17_sentiment_results.csv


Output

Sentiment labels for each post (positive / negative)

Machine learning model performance metrics

LSTM model accuracy

Accuracy comparison plot

CSV file with sentiment results: iphone17_sentiment_results.csv



Insights

Lexicon methods give a quick overview of overall sentiment.

ML models perform well on smaller datasets.

LSTM captures context and performs best with larger datasets.

Helps understand public opinion on the iPhone 17 from Reddit posts.


SCRAPING FOR DATA

•	Fetch Reddit posts about iPhone 17 using PRAW (Reddit API)

•	Clean and preprocess text (lowercasing, removing URLs/mentions/punctuation, lemmatization, stopwords removal)

•	Perform lexicon-based sentiment analysis (VADER + TextBlob)

•	Train machine learning models (Naive Bayes, Logistic Regression, SVM) for sentiment classification

•	Build deep learning LSTM/BiLSTM models for context-aware sentiment detection

•	Visualize model performance

•	Save processed data and sentiment results to CSV


