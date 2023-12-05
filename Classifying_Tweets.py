import os
import subprocess
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import wandb
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import TextVectorization

# ------ FETCH DATA --------------------------------------------------------------------------------------
# Download datasets
subprocess.run(['wget', 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'])

# Login to Weights & Biases
subprocess.run(['wandb', 'login', '--relogin'])

# Fetch Data
# Send the raw_data to the wandb and store it as an artifact
subprocess.run([
    'wandb', 'artifact', 'put',
    '--name', 'tweets_classifying/train',
    '--type', 'RawData',
    '--description', 'Real and Fake Disaster-Related Tweets Dataset',
    'train.csv'
])

# Clean up downloaded file (optional)
os.remove('train.csv')

# ------ EDA --------------------------------------------------------------------------------------

# Initialize wandb run
wandb.init(project='tweets_classifying', save_code=True)

# Get the artifact
artifact = wandb.use_artifact('mlops-ufrn/tweets_classifying/train:v0', type='RawData')

# Download the content of the artifact to the local directory
artifact_dir = artifact.download()

df = pd.read_csv(artifact.file())

target_value_counts = df['target'].value_counts()
wandb.log({"Target Value Counts": target_value_counts.to_dict()})

normalized_target_value_counts = df['target'].value_counts(normalize=True)
wandb.log({"Normalized Target Value Counts": normalized_target_value_counts.to_dict()})

sns.countplot(x='target', data = df)
plt.title('Tweet Count by Category')
plt.savefig('file_category_tweet.png')
plt.show()
plt.close()

# Log the table image to wandb
wandb.log({"File Category Tweet": wandb.Image('file_category_tweet.png')})

# ------ PREPROCESSING --------------------------------------------------------------------------------------

df = df.drop(['id','keyword', 'location'], axis=1)

# Lower Character all the Texts
df['text'] = df['text'].str.lower()

# Removing Punctuations and Numbers from the Text
def punctuations(inputs):
    return re.sub(r'[^a-zA-Z]', ' ', inputs)

df['text'] = df['text'].apply(punctuations)

def tokenization(inputs):
    return word_tokenize(inputs)

df['text_tokenized'] = df['text'].apply(tokenization)

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

def stopwords_remove(inputs):
    return [k for k in inputs if k not in stop_words]

df['text_stop'] = df['text_tokenized'].apply(stopwords_remove)

lemmatizer = WordNetLemmatizer()

def lemmatization(inputs):
    return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]

df['text_lemmatized'] = df['text_stop'].apply(lemmatization)

# Joining Tokens into Sentences
df['final'] = df['text_lemmatized'].str.join(' ')

# Salve o DataFrame como um arquivo CSV
df.to_csv('cleanData.csv', index=False)

subprocess.run([
    'wandb', 'artifact', 'put',
    '--name', 'tweets_classifying/clean_data',
    '--type', 'CleanData',
    '--description', 'Preprocessed data',
    'cleanData.csv'
])

# Clean up downloaded file (optional)
os.remove('cleanData.csv')

# ------ DATA SEGREGATION --------------------------------------------------------------------------------------

# Initialize wandb run
run = wandb.init(project='tweet_classifying', job_type='data_segregation')

# Get the clean_data artifact
artifact = run.use_artifact('cleanData.csv:latest')
df = pd.read_csv(artifact)

X = df['final']
y = df['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Convert split data to DataFrames
train_data = pd.DataFrame({'text': x_train, 'label': y_train})
test_data = pd.DataFrame({'text': x_test, 'label': y_test})

# Log the shapes of the training and testing datasets
wandb.log({'train_data_shape': train_data.shape,
           'test_data_shape': test_data.shape})

# Save split data to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Create new artifacts for train and test data
train_artifact = wandb.Artifact(
    name='train_data',
    type='TrainData',
    description='Training data split from cleanData'
)
test_artifact = wandb.Artifact(
    name='test_data',
    type='TestData',
    description='Testing data split from cleanData'
)

# Add CSV files to the artifacts
train_artifact.add_file('trainData.csv')
test_artifact.add_file('testData.csv')

# Log the new artifacts to wandb
run.log_artifact(train_artifact)
run.log_artifact(test_artifact)

wandb.finish()

