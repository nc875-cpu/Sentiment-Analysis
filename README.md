# 📱 Sentiment Analysis on iPhone 17 Reddit Reviews

Analyze and visualize global sentiment around the iPhone 17 launch using NLP and Deep Learning (LSTM).
This project combines **TextBlob + VADER** for rule-based sentiment and a custom **LSTM model** for advanced prediction.

---

## 🚀 Project Overview
- **Goal:** Understand public sentiment about iPhone 17 from Reddit discussions.
- **Techniques Used:** NLP preprocessing, sentiment analysis, visualization, and LSTM modeling.
- **Key Features:**
  - Hybrid (VADER + TextBlob) sentiment scoring
  - Deep learning LSTM model
  - Interactive sentiment maps by country
  - Data visualization (bar charts, pie charts, confusion matrices)

---

## 🧩 Technologies Used
| Category | Libraries |
|-----------|------------|
| Data Processing | `pandas`, `numpy`, `re` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| NLP | `TextBlob`, `vaderSentiment` |
| ML/DL | `tensorflow`, `scikit-learn` |
| Evaluation | `classification_report`, `confusion_matrix` |

---

## Model Architecture
Embedding (10000 x 128)

        ↓
LSTM (128 units, dropout=0.3)

        ↓
Dense (64, ReLU)

        ↓
Dropout (0.3)

        ↓
Dense (3, Softmax Output)

----
## SCRAPING FOR DATA

•	Fetch Reddit posts about iPhone 17 using PRAW (Reddit API)

•	Clean and preprocess text (lowercasing, removing URLs/mentions/punctuation, lemmatization, stopwords removal)

•	Perform lexicon-based sentiment analysis (VADER + TextBlob)

•	Train machine learning models (Naive Bayes, Logistic Regression, SVM) for sentiment classification

•	Build deep learning LSTM/BiLSTM models for context-aware sentiment detection

•	Visualize model performance

•	Save processed data and sentiment results to CSV

-----
## Libraries used in this project

[1]	Pandas: Handles structured data like reading, cleaning, and manipulating the data frames, CSV.

[2]	Numpy: Used for array handling, and provides fast mathematical operation for numerical computations.

[3]	Re: Regular expression are used for cleaning the text and match the pattern in preprocessing.

[4]	Matplotlib.pyplot: Used for data visualization and plotting the data.

[5]	Seaborn: High-level statistic data visualization.

[6]	Plotly.express: Interactive charting library i.e…; the choropleth maps and better dynamic plots.

### Sentiment and text analysis libraries

[1]	TextBlob: Performs NLP such as sentiment, sujbjective analysis, polarity.

[2]	VaderSentiment: Lexicon-based sentiment analyzer for social media and short text.

### Machine Learning aand evaluation libraries

[1]	Sklearn.preprocessing.LabelEncoder: converts the sentiment labels which is our text into numeric codes for model training. 

[2]	Sklearn.metrics.classification-report: Gives a summary of model performance with precision, recall and F1 score.

[3]	Sklearn.metrics.confusion_matrix: matching of actual classes and predicted classes.

[4]	Sklearn.model_selection.train_test_split: Splits data into training and testing for evaluation.

### Deep Learning(TensorFlow/Keras)

[1]	Tensorflow.keras.preprocessing.text.Tokenizer: Raw text gets converted into sequences of integer tokens for dl models.

[2]	Tensorflows.keras.preprocessing.sequence.pad_sequences: Pads all sequences to the same length for uniform input shape. 

[3]	Tensorflow.keras.models.sequential: builds a linear stack of neural network layers.

[4]	Tensorflow.keras.layers.Embedding: Converts words into dense numerical vectors.

[5]	Tensorflow.keras.layers.LSTM: A type of recurrent neural network layer used for sequence learning.

[6]	Tensorflow.keras.layers.Dense: Fully connected neural network layer used for classification output.

[7]	Tensorflow.keras.layers.dropout: Randomly drops neurons during training to prevent overfitting.

[8]	Tensorflow.keras.optimizers.Adam: Adaptive optimizer that adjusts learning rates for efficient model training.

[9]	Tensorflow.keras.utils.to_categorical: converts integer labels into one-hot encoded vectors for multi-class classification.
