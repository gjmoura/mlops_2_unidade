import os
import re
import wandb
import json
import nltk
import pytest
import logging
import subprocess
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import datasets
import transformers
import tensorflow as tf

from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python_operator import PythonOperator

from datasets import load_dataset
from nltk.corpus import stopwords
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from transformers import TFAutoModelForSequenceClassification

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Lê o conteúdo do arquivo JSON contendo as variáveis de ambiente
with open('config.json', 'r') as file:
    config_data = json.load(file)




# ------ DEFINE FUNCTIONS --------------------------------------------------------------------------------------

# def delete_data(file_name):
#     logging.info(f"Deleting {file_name}")
#     os.remove(file_name)

def fetch_data(api_key):
    # Download dataset
    logging.info("Dowloading dataset")
    subprocess.run(['wget', 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'])

    # Login to Weights & Biases
    logging.info("Logging in to Weights & Biases")
    wandb.login(key=api_key)

    # Send the raw_data to the wandb and store it as an artifact
    logging.info("Send the raw_data to the wandb and store it as an artifact")
    subprocess.run([
        'wandb', 'artifact', 'put',
        '--name', 'tweets_classifying/train',
        '--type', 'RawData',
        '--description', 'Real and Fake Disaster-Related Tweets Dataset',
        'train.csv'
    ])

def data_exploration():
    # Get the artifact
    artifact = wandb.use_artifact('mlops-ufrn/tweets_classifying/train:v0', type='RawData')

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



def preprocessing_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')


    # Get the artifact
    artifact = wandb.use_artifact('mlops-ufrn/tweets_classifying/train:v0', type='RawData')

    df = pd.read_csv(artifact.file())
    
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
    df.to_csv('clean_data.csv', index=False)

    subprocess.run([
        'wandb', 'artifact', 'put',
        '--name', 'tweets_classifying/clean_data',
        '--type', 'RawData',
        '--description', 'Preprocessed data',
        'clean_data.csv'
    ])


def data_check():
    result = pytest.main(["-vv", "."])

    if result != 0:
        raise ValueError("Data checks failed")


def data_segregation(run):

    # Get the clean_data artifact
    artifact = run.use_artifact('clean_data:latest')
    path = artifact.get_path('clean_data.csv')
    cleanData = path.download()
    df = pd.read_csv(cleanData)

    X = df['final']
    y = df['target']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Convert split data to DataFrames
    train_data = pd.DataFrame({'final': x_train, 'target': y_train})
    test_data = pd.DataFrame({'final': x_test, 'target': y_test})

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
    train_artifact.add_file('train_data.csv')
    test_artifact.add_file('test_data.csv')

    # Log the new artifacts to wandb
    run.log_artifact(train_artifact)
    run.log_artifact(test_artifact)

  
def data_train(run):

    # Get the train artifact
    train_artifact = run.use_artifact('train_data:latest')
    train_path = train_artifact.get_path('train_data.csv')
    train_data = train_path.download()
    train_df = pd.read_csv(train_data)

    # Get the test artifact
    test_artifact = run.use_artifact('test_data:latest')
    test_path = test_artifact.get_path('test_data.csv')
    test_data = test_path.download()
    test_df = pd.read_csv(test_data)

    X_train = train_df['final']
    y_train = train_df['target']

    X_test = test_df['final']
    y_test = test_df['target']

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

    # Define model, optimizer, loss function and metrics
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(train_dataset, epochs=10, validation_data=train_dataset)

    # ------ NOVO ------
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(np.arange(0, 1), history.history["loss"], label="train_loss",linestyle='--')
    ax.plot(np.arange(0, 1), history.history["val_loss"], label="val_loss",linestyle='--')
    ax.plot(np.arange(0, 1), history.history["accuracy"], label="train_acc")
    ax.plot(np.arange(0, 1), history.history["val_accuracy"], label="val_acc")

    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")

    ax.legend()
    plt.tight_layout()

    plt.savefig(f"training_loss_acc.png")

    wandb.log({f"training_loss_acc": wandb.Image(f"training_loss_acc.png")})


def wandb_finish(run):
    run.finish()





run = wandb.init(project='tweets_classifying', save_code=True)

WANDB_API_KEY = config_data.get('WANDB_API_KEY')

DEFAULT_ARGS = {
    "owner": "airflow",
    "start_date": datetime(2023, 11, 30),
    "catchup": False,
}

with DAG("tweets_classifying", default_args=DEFAULT_ARGS, schedule_interval="@daily") as dag:
# ------ FETCH DATA --------------------------------------------------------------------------------------
    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
        op_kwargs={
            "api_key": WANDB_API_KEY
        }
    )

    # delete_raw_data = PythonOperator(
    #     task_id="delete_raw_data",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'train.csv'
    #     }
    # )

# ------ EDA --------------------------------------------------------------------------------------
    data_exploration = PythonOperator(
        task_id="data_exploration",
        python_callable=data_exploration
    )

    # delete_raw_data_eda = PythonOperator(
    #     task_id="delete_raw_data_eda",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'train.csv'
    #     }
    # )

# ------ PREPROCESSING --------------------------------------------------------------------------------------
    preprocessing_data = PythonOperator(
        task_id="preprocessing_data",
        python_callable=preprocessing_data
    )

    # delete_preprocessing_data = PythonOperator(
    #     task_id="delete_preprocessing_data",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'clean_data.csv'
    #     }
    # )

# ------ DATA CHECK --------------------------------------------------------------------------------------
    data_check = PythonOperator(
        task_id="data_check",
        python_callable=data_check
    )

# ------ DATA SEGREGATION --------------------------------------------------------------------------------------
    data_segregation = PythonOperator(
        task_id="data_segregation",
        python_callable=data_segregation,
        op_kwargs={
            "run": run
        }
    )

    # delete_clean_data = PythonOperator(
    #     task_id="delete_clean_data",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'clean_data.csv'
    #     }
    # )

    # delete_train_data = PythonOperator(
    #     task_id="delete_train_data",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'train_data.csv'
    #     }
    # )

    # delete_test_data = PythonOperator(
    #     task_id="delete_test_data",
    #     python_callable=delete_data,
    #     op_kwargs={
    #         "file_name": 'test_data.csv'
    #     }
    # )

# ------ DATA TRAIN --------------------------------------------------------------------------------------
    data_train = PythonOperator(
        task_id="data_train",
        python_callable=data_train,
        op_kwargs={
            "run": run
        }
    )

    wandb_finish = PythonOperator(
        task_id="wandb_finish",
        python_callable=wandb_finish,
        op_kwargs={
            "run": run
        }
    )

    
#--------------- SEM DELETE_DATA() --------------------------------------------------
fetch_data.set_downstream(data_exploration)
data_exploration.set_downstream(preprocessing_data)
preprocessing_data.set_downstream(data_check)
data_check.set_downstream(data_segregation)
data_segregation.set_downstream(data_train)
data_train.set_downstream(wandb_finish)


#--------------- COM DELETE_DATA() --------------------------------------------------
# fetch_data.set_downstream([data_exploration, delete_raw_data])
# data_exploration.set_downstream([preprocessing_data, delete_raw_data_eda])
# preprocessing_data.set_downstream([data_check, delete_preprocessing_data])
# data_check.set_downstream(data_segregation)
# data_segregation.set_downstream([delete_clean_data, delete_train_data, delete_test_data, data_train])
# data_train.set_downstream(wandb_finish)
