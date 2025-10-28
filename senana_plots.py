# SENTIMENT ANALYSIS OF IPHONE 17 REDDIT POSTS USING NLP

Importing Libraries
"""

!pip install vaderSentiment

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob       #Text and Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

nltk.download('stopwords')   #Downloading NLTK Resources
nltk.download('wordnet')

"""Load Uploaded Dataset"""

url = "https://raw.githubusercontent.com/nc875-cpu/Sentiment-Analysis/main/iphone17_reddit__country.csv"
df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')

print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print("✅ Dataset loaded successfully!")
df.head()

print("Column names in your dataset:")
print(list(df.columns))

"""Automatic column detection"""

text_col = None
country_col = None
for col in df.columns:
    low = col.lower()
    if any(k in low for k in ['text','body','post','comment','review','title']):
        if text_col is None:
            text_col = col
    if any(k in low for k in ['country','location','place','region','loc','area']):
        if country_col is None:
            country_col = col
if text_col is None:
    for col in df.columns:
        if df[col].dtype == object:
            text_col = col
            break
if country_col is None:
    for col in df.columns[::-1]:
        if df[col].dtype == object and col != text_col:
            country_col = col

print(f"Detected text column: {text_col}")
print(f"Detected country column: {country_col}")

"""Keeping and renaming the required columns"""

df = df.rename(columns={text_col: 'post', country_col: 'country'})
df = df[['post','country']].dropna().reset_index(drop=True)

"""Text pre-processing"""

simple_stopwords = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at","be","because","been",
    "before","being","below","between","both","but","by","could","did","do","does","doing","down","during","each",
    "few","for","from","further","had","has","have","having","he","her","here","hers","herself","him","himself",
    "his","how","i","if","in","into","is","it","its","itself","just","me","more","most","my","myself","no","nor",
    "not","now","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","same",
    "she","should","so","some","such","than","that","the","their","theirs","them","themselves","then","there",
    "these","they","this","those","through","to","too","under","until","up","very","was","we","were","what",
    "when","where","which","while","who","whom","why","with","would","you","your","yours","yourself","yourselves"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    words = [w for w in text.split() if w not in simple_stopwords and len(w) > 1]
    return " ".join(words)

df['clean_post'] = df['post'].apply(preprocess)

"""Sentiment Analysis using VADER+TEXTBLOB"""

analyzer = SentimentIntensityAnalyzer()

def hybrid_sentiment(text):
    vader_score = analyzer.polarity_scores(text)['compound']
    blob_score = TextBlob(text).sentiment.polarity
    final_score = (vader_score + blob_score) / 2
    if final_score >= 0.05:
        return 'positive'
    elif final_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['clean_post'].apply(hybrid_sentiment)
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

"""Lexicon-Based Sentiment Analysis VADER + TextBlob"""

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['clean_post'].apply(vader_sentiment)

def textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

df['textblob_sentiment'] = df['clean_post'].apply(textblob_sentiment)

print("\nLexicon-Based Sentiment Counts (VADER):")
print(df['vader_sentiment'].value_counts())
print("\nLexicon-Based Sentiment Counts (TextBlob):")
print(df['textblob_sentiment'].value_counts())

"""data prep for ML models"""

# Using the results from TextBlob as pseudo labels
df['label'] = df['textblob_sentiment']
df = df[df['label'] != 'neutral']  # keeping only positive/negative values for binary classification

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_post'])
y = df['label']

# splitting data for Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Training ML models"""

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label='positive')
    rec = recall_score(y_test, preds, pos_label='positive')
    f1 = f1_score(y_test, preds, pos_label='positive')
    results.append([name, acc, prec, rec, f1])

ml_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\nMachine Learning Results:")
print(ml_results)

"""Text encoding for LSTM"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

X = df["clean_post"].values
y = df["sentiment"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=3)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding="post", truncating="post")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_categorical, test_size=0.2, random_state=42)

"""Build LSTM model"""

vocab_size = 10000
embedding_dim = 128
maxlen = 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    LSTM(128, dropout=0.3, recurrent_dropout=0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

# Build the model explicitly (so summary shows params)
model.build(input_shape=(None, maxlen))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

"""training the model"""

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

"""Evaluation of the model"""

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 LSTM Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix (LSTM Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy plot
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training & Validation Accuracy (LSTM)")
plt.legend()
plt.show()

"""comparision of LSTM vs RULE BASED sentiment"""

df["lstm_pred"] = model.predict(X_pad).argmax(axis=1)
df["lstm_sentiment"] = le.inverse_transform(df["lstm_pred"])

compare = pd.crosstab(df["sentiment"], df["lstm_sentiment"], normalize="index")
print("\nComparison (Rule-based vs LSTM):")
print(compare)

sns.heatmap(compare, annot=True, cmap="Blues", fmt=".2f")
plt.title("Hybrid (VADER+TextBlob) vs LSTM Sentiment")
plt.show()

country_sent = df.groupby(["country","lstm_sentiment"]).size().unstack(fill_value=0).reset_index()
country_sent["total"] = country_sent.sum(axis=1, numeric_only=True)
for s in ["positive","negative","neutral"]:
    if s not in country_sent.columns:
        country_sent[s] = 0
country_sent["positive_percent"] = country_sent["positive"] / country_sent["total"] * 100
country_sent["negative_percent"] = country_sent["negative"] / country_sent["total"] * 100
country_sent["neutral_percent"]  = country_sent["neutral"]  / country_sent["total"] * 100

fig = px.choropleth(
    country_sent, locations="country", locationmode="country names",
    color="positive_percent", hover_name="country",
    hover_data=["positive","negative","neutral"],
    color_continuous_scale="RdYlGn", title="🌍 LSTM Model: Positive Sentiment % by Country"
)
fig.show()

"""Visualization of all the distributed sentiments"""

plt.figure(figsize=(7,4))
sns.countplot(x='sentiment', data=df, order=['positive','neutral','negative'], palette=['#2ca02c','#1f77b4','#d62728'])
plt.title('Sentiment Distribution (Count)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

plt.figure(figsize=(6,6))
df['sentiment'].value_counts().reindex(['positive','neutral','negative']).plot.pie(
    autopct='%1.1f%%', colors=['#2ca02c','#1f77b4','#d62728'], startangle=90)
plt.title('Sentiment Percentage Distribution')
plt.ylabel('')
plt.show()

"""frequently used words"""

def get_word_frequencies(sentiment_label):
    words = " ".join(df[df['sentiment'] == sentiment_label]['clean_post']).split()
    freq = pd.Series(words).value_counts().head(15)
    return freq

for s in ['positive','negative','neutral']:
    print(f"\nTop words in {s} reviews:")
    print(get_word_frequencies(s))

"""Reviews by country"""

df['country'] = df['country'].fillna('Unknown')
sentiment_counts = df.groupby(['country','sentiment']).size().unstack(fill_value=0).reset_index()
for col in ['positive','negative','neutral']:
    if col not in sentiment_counts.columns:
        sentiment_counts[col] = 0
sentiment_counts['total'] = sentiment_counts[['positive','negative','neutral']].sum(axis=1)
sentiment_counts['positive_percent'] = (sentiment_counts['positive'] / sentiment_counts['total'])*100
sentiment_counts['negative_percent'] = (sentiment_counts['negative'] / sentiment_counts['total'])*100
sentiment_counts['neutral_percent'] = (sentiment_counts['neutral'] / sentiment_counts['total'])*100

print("\nSentiment by Country:")
print(sentiment_counts.head())

"""Chloropleth maps"""

fig_pos = px.choropleth(
    sentiment_counts, locations='country', locationmode='country names',
    color='positive_percent', hover_name='country',
    hover_data=['positive','negative','neutral'],
    color_continuous_scale='RdYlGn', title='🌍 Positive Sentiment % by Country'
)
fig_pos.show()

fig_neg = px.choropleth(
    sentiment_counts, locations='country', locationmode='country names',
    color='negative_percent', hover_name='country',
    hover_data=['positive','negative','neutral'],
    color_continuous_scale='Reds', title='🌍 Negative Sentiment % by Country'
)
fig_neg.show()

fig_neu = px.choropleth(
    sentiment_counts, locations='country', locationmode='country names',
    color='neutral_percent', hover_name='country',
    hover_data=['positive','negative','neutral'],
    color_continuous_scale='Blues', title='🌍 Neutral Sentiment % by Country'
)
fig_neu.show()

top_positive = sentiment_counts.sort_values('positive_percent', ascending=False).head(10)
top_negative = sentiment_counts.sort_values('negative_percent', ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(y='country', x='positive_percent', data=top_positive, palette='Greens_r')
plt.title('Top 10 Countries with Most Positive Sentiment')
plt.xlabel('Positive Sentiment (%)')
plt.ylabel('Country')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(y='country', x='negative_percent', data=top_negative, palette='Reds_r')
plt.title('Top 10 Countries with Most Negative Sentiment')
plt.xlabel('Negative Sentiment (%)')
plt.ylabel('Country')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

df['vader_sentiment'] = df['clean_post'].apply(lambda x: 'positive' if analyzer.polarity_scores(x)['compound']>0.05
                                               else 'negative' if analyzer.polarity_scores(x)['compound']<-0.05
                                               else 'neutral')
print("\nEvaluation Metrics (Hybrid vs VADER):")
print(classification_report(df['vader_sentiment'], df['sentiment']))

cm = confusion_matrix(df['vader_sentiment'], df['sentiment'], labels=['positive','negative','neutral'])
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Pred_Pos','Pred_Neg','Pred_Neu'],
            yticklabels=['True_Pos','True_Neg','True_Neu'])
plt.title('Confusion Matrix (Hybrid vs VADER)')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='Accuracy', data=ml_results)
plt.title("Machine Learning Model Accuracy Comparison")
plt.show()

"""saving the results"""

output_path = "iphone17_sentiment_results_full.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

"""Summary"""

print("\n📊 SUMMARY:")
print("- Cleaned and preprocessed Reddit text data.")
print("- Performed hybrid (VADER + TextBlob) sentiment classification.")
print("- Evaluated performance against VADER baseline.")
print("- Created bar, pie, and map visualizations by country.")
print("- Generated word frequency summaries for each sentiment type.")
print("- Saved final labeled dataset for reporting.")