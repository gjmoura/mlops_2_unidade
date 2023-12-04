import os
import subprocess


# Download datasets
subprocess.run(['wget', 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'])

# Login to Weights & Biases
subprocess.run(['wandb', 'login', '--relogin'])

# Fetch Data
# Send the raw_data to the wandb and store it as an artifact
subprocess.run([
    'wandb', 'artifact', 'put',
    '--name', 'classifying_tweets/train',
    '--type', 'RawData',
    '--description', 'Real and Fake Disaster-Related Tweets Dataset',
    'train.csv'
])

# Clean up downloaded file (optional)
os.remove('train.csv')
