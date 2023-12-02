import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
print(df.shape)
print(df.columns)

df.head()

df.tail()

print(df.info())

df.isna().sum()

print(df['keyword'].nunique()); print(df['location'].nunique()); print(df['text'].nunique()); print(df['id'].nunique());print(df['target'].nunique())

# Data Exploration

df = df.drop(['id','keyword', 'location'], axis=1)
print(df.info())
print(df.head())

df['target'].value_counts()

df['target'].value_counts(normalize=True)

sns.countplot('target', data = df)
plt.title('Tweet Count by Category')
plt.show()

# Text Preprocessing
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lower Character all the Texts
df['text'] = df['text'].str.lower()
df['text'].head()

# Removing Punctuations and Numbers from the Text
def punctuations(inputs):
    return re.sub(r'[^a-zA-Z]', ' ', inputs)


df['text'] = df['text'].apply(punctuations)
df['text'].head()

def tokenization(inputs):
    return word_tokenize(inputs)


df['text_tokenized'] = df['text'].apply(tokenization)
df['text_tokenized'].head()

stop_words = set(stopwords.words('english'))
stop_words.remove('not')


def stopwords_remove(inputs):
    return [k for k in inputs if k not in stop_words]


df['text_stop'] = df['text_tokenized'].apply(stopwords_remove)
df['text_stop'].head()

lemmatizer = WordNetLemmatizer()


def lemmatization(inputs):
    return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]


df['text_lemmatized'] = df['text_stop'].apply(lemmatization)
df['text_lemmatized'].head()

# Joining Tokens into Sentences
df['final'] = df['text_lemmatized'].str.join(' ')
df['final'].head()

df.head()

# WordCloud
data_disaster = df[df['target'] == 1]
data_not_disaster = df[df['target'] == 0]

from wordcloud import WordCloud

WordCloud_disaster = WordCloud(max_words=500,
                                  random_state=100,background_color='white',
                                  collocations=True).generate(str((data_disaster['final'])))

plt.figure(figsize=(15, 10))
plt.imshow(WordCloud_disaster, interpolation='bilinear')
plt.title('WordCloud of the Disaster Tweets', fontsize=10)
plt.axis("off")
plt.show()

WordCloud_not_disaster = WordCloud(max_words=500,
                                      random_state=100, background_color='white',
                                      collocations=True).generate(str((data_not_disaster['final'])))

plt.figure(figsize=(15, 10))
plt.imshow(WordCloud_not_disaster, interpolation='bilinear')
plt.title('WordCloud of the Non Disaster Tweets', fontsize=10)
plt.axis("off")
plt.show()

X = df['final']
y = df['target']

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Print the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import TextVectorization

max_tokens = 7500
input_length = 128
output_dim = 128

vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode='int', standardize='lower_and_strip_punctuation', output_sequence_length=input_length)
vectorizer_layer.adapt(X_train)

#Creating and Embedding Layer:
from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(input_dim=max_tokens,
                            output_dim=output_dim,
                            input_length=input_length)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorizer_layer)
model.add(embedding_layer),
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

opt = tf.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, verbose=2)
model.evaluate(X_test, y_test)

# Build a multi-layer deep text classification model
from tensorflow.keras.regularizers import L1, L2, L1L2
model_regularized = tf.keras.models.Sequential()
model_regularized.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model_regularized.add(vectorizer_layer)
model_regularized.add(embedding_layer)
model_regularized.add(tf.keras.layers.GlobalAveragePooling1D())
model_regularized.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=L1(0.0005)))
model_regularized.add(tf.keras.layers.Dropout(0.6))
model_regularized.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L1L2(0.0005)))
model_regularized.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2(0.0005)))
model_regularized.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2(0.0005)))
model_regularized.add(tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=L2(0.0005)))
model_regularized.add(tf.keras.layers.Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model_regularized.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model_regularized.fit(X_train, y_train, epochs=10, verbose=2)
model_regularized.evaluate(X_test, y_test)

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import Adam, RMSprop

ml_bi_lstm = Sequential()
ml_bi_lstm.add(Input(shape=(1,), dtype=tf.string))
ml_bi_lstm.add(vectorizer_layer)
ml_bi_lstm.add(embedding_layer)
ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
ml_bi_lstm.add(Bidirectional(LSTM(64)))
ml_bi_lstm.add(Dense(64, activation='elu', kernel_regularizer=L1L2(0.0001)))
ml_bi_lstm.add(Dense(32, activation='elu', kernel_regularizer=L2(0.0001)))
ml_bi_lstm.add(Dense(8, activation='elu', kernel_regularizer=L2(0.0005)))
ml_bi_lstm.add(Dense(8, activation='elu'))
ml_bi_lstm.add(Dense(4, activation='elu'))
ml_bi_lstm.add(Dense(1, activation='sigmoid'))
opt = RMSprop(learning_rate=0.0001, rho=0.8, momentum=0.9)
ml_bi_lstm.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
ml_bi_lstm.fit(X_train, y_train, epochs=10)
test_loss, test_acc = ml_bi_lstm.evaluate(X_test, y_test)
print(f"Test set accuracy: {test_acc}")

# Building a Transformer Model


import datasets
import transformers
import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Tokenize the text data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    tf.constant(y_train.values, dtype=tf.int32)
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    tf.constant(y_test.values, dtype=tf.int32)
))

train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(train_dataset, epochs=10, validation_data=train_dataset)
