import os
import subprocess
import matplotlib.pyplot as plt
import wandb
import pandas as pd

# Initialize wandb run
wandb.init(project='tweet_classifying', save_code=True)

# Get the artifact
artifact = wandb.use_artifact('raw_data:latest')

# Download the content of the artifact to the local directory
artifact_dir = artifact.download()

# read data
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

# finish run
wandb.finish()