#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: sentiment analysis of SUTD Reddit
# **Assignment due 21 March 11:59pm**
# 
# Welcome to the second assignment for 50.055 Machine Learning Operations. These assignments give you a chance to practice the methods and tools you have learned. 
# 
# **This assignment is an individual assignment.**
# 
# - Read the instructions in this notebook carefully
# - Add your solution code and answers in the appropriate places. The questions are marked as **QUESTION:**, the places where you need to add your code and text answers are marked as **ADD YOUR SOLUTION HERE**
# - The completed notebook, including your added code and generated output and a labeled dataset which you create in the assignment will be your submission for the assignment.
# - The notebook should execute without errors from start to finish when you select "Restart Kernel and Run All Cells..". Please test this before submission.
# - Use the SUTD Education Cluster to solve and test the assignment.
# 
# **Rubric for assessment** 
# 
# Your submission will be graded using the following criteria. 
# 1. Code executes: your code should execute without errors. The SUTD Education cluster should be used to ensure the same execution environment.
# 2. Correctness: the code should produce the correct result or the text answer should state the factual correct answer.
# 3. Style: your code should be written in a way that is clean and efficient. Your text answers should be relevant, concise and easy to understand.
# 4. Partial marks will be awarded for partially correct solutions.
# 5. There is a maximum of 150 points for this assignment.
# 
# **ChatGPT policy** 
# 
# If you use AI tools, such as ChatGPT, to solve the assignment questions, you need to be transparent about its use and mark AI-generated content as such. In particular, you should include the following in addition to your final answer:
# - A copy or screenshot of the prompt you used
# - The name of the AI model
# - The AI generated output
# - An explanation why the answer is correct or what you had to change to arrive at the correct answer
# 
# **Assignment Notes:** Please make sure to save the notebook as you go along. Submission Instructions are located at the bottom of the notebook.
# 

# In[ ]:


# Installing all required packages
# Note: Do not add to this list.
# ----------------
get_ipython().system(' pip install transformers[torch]==4.37.2')
get_ipython().system(' pip install datasets==2.17.1')
get_ipython().system(' pip install seaborn==0.13.2')
get_ipython().system(' pip install pyarrow==15.0.0')
get_ipython().system(' pip install scikit-learn==1.4.0')
get_ipython().system(' pip install emoji==0.6.0')
get_ipython().system(' pip install accelerate==0.27.2')
# ----------------


# In[ ]:


# Importing all required packages
# ----------------
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
# ----------------


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Sentiment analysis
# 
# Sentiment analysis is a natural language processing technique that identifies the polarity of a given text. There are different flavors of sentiment analysis, but one of the most widely used techniques labels data into positive, negative and neutral. We have already encountererd sentiment analysis in the hands-on sessions.
# 
# In this assignment, you will conduct sentiment analysis on posts and comments from the SUTD subreddit. You will run experiments with pre-trained sentiment models, evaluate their performance and simulate improving the model by re-training it with newly annotated data. 
# 

# In[ ]:


# Load SUTD subreddit data set as dataframe
# posts and comments have been downloaded from https://www.reddit.com/r/sutd/

df_submissions = pd.read_parquet('reddit_submissions.parquet.gzip').set_index("Id")
df_comments = pd.read_parquet('reddit_comments.parquet.gzip').set_index("CommentId")


# In[ ]:


#Let's have a look at the data. The data schema is as follows.

# Submissions
# Id - unique id for submission
# Title - text of the submission title
# Upvotes - upvotes on this submission
# Created - date time of submission creation date and time

# Comments
# CommentId - unique id for comment
# Comment - text content of the comment
# CommentCreated - date time of comment creation date and time
# Id - unique id for submission on which the comment was posted

# See the Reddit API documentation for details https://www.reddit.com/dev/api/
df_submissions


# In[ ]:


df_comments


# You can read the SUTD reddit submissions in your web browser by navigating to 
# https://www.reddit.com/r/sutd/comments/{Id}
# 
# 
# ### QUESTION: 
# How easy is it to make sense of the submissions and comments? Is it easier to understand the posts when you read them in the browser? 
# Explain why or why not (max 100 words)
# 
# **--- ADD YOUR SOLUTION HERE (5 points)---**
# 
# 
# ------------------------------
# 

# In[ ]:


# QUESTION: Join the data frames into a joined data_frame 'df_reddit' which  contains both submissions and comments. 
# Each row should contain a submission paired with one associated comment. Comments that do not have a matching submission shall be dropped. The joined data frame should have the following schema.

# Submissions
# Id - unique id for submission
# Title - text of the submission title
# Upvotes - upvotes on this submission
# Created - date time of submission creation date and time
# CommentId - unique id for comment, comment is posted for this submission
# Comment - text content of the comment
# CommentCreated - date time of comment creation date and time


#--- ADD YOUR SOLUTION HERE (5 points)---

#------------------------------


# In[ ]:


# Print the first 10 rows of the joined data frame
df_reddit.head(10)

# Hint: submission will be duplicated as many times as there are comments


# In[ ]:


# Now let's run a pre-trained sentiment analysis model on the submissions and comments
# A convenient way to execute pre-trained models for standard tasks are Huggingface pipelines
# Here we run a standard sentiment analysis pipeline on the first ten submission titles 
sentiment_pipeline = pipeline("sentiment-analysis", device=0)
print(sentiment_pipeline(list(df_submissions['Title'][:10])))


# In[ ]:


# QUESTION: Complete the function 'analyse_sentiment' which takes a data frame, a Huggingface sentiment pipeline object 
# and a target column name and adds two columns 'Label' and 'Score' to the data frame in place.
# pass the provided tokenizer arguments to the pipeline
# The new columns should contain the sentiment labels and scores, respectively.


def analyse_sentiment(df, sentiment_pipeline, column):
    tokenizer_kwargs = {'padding':True, 'truncation':True, 'max_length':128,}
#--- ADD YOUR SOLUTION HERE (10 points)---

#------------------------------


# In[ ]:


# add sentiment labels and scores to the submissions and comments dataframes
analyse_sentiment(df_submissions, sentiment_pipeline, 'Title')
analyse_sentiment(df_comments, sentiment_pipeline, 'Comment')


# In[ ]:


# display dataframe 
df_submissions


# In[ ]:


# display dataframe 
df_comments


# ### QUESTION: 
# From a first inspection of the results, what problems can you see with our current sentiment analysis?
# What model is used for the sentiment analysis and how was is trained?
# 
# **--- ADD YOUR SOLUTION HERE (5 points) ---**
# 
# 
# ------------------------------
# 

# In[ ]:


# QUESTION: Update the sentiment pipeline to use the model "finiteautomata/bertweet-base-sentiment-analysis" from Huggingface
# The model should output three classes: 'POS', 'NEG', 'NEU'
# Store the model name in separate variable "model_name"

#--- ADD YOUR SOLUTION HERE (5 points) ---
#------------------------------


# ### QUESTION: 
# 
# Explain why this model is better suited for the task (max 100 words).
# 
# **--- ADD YOUR SOLUTION HERE (5 points) ---**
# 
# ------------------------------
# 

# In[ ]:


# re-run the sentiment analysis of submissions and comments
analyse_sentiment(df_submissions, sentiment_pipeline, 'Title')
analyse_sentiment(df_comments, sentiment_pipeline, 'Comment')


# In[ ]:


# display dataframe 
df_submissions


# In[ ]:


# display dataframe 
df_comments


# In[ ]:


# QUESTION: What is the time frame covered by the data set, i.e. what is the earliest time of a submission or comment and what is the most recent time?
# Find the earliest and latest timestamp and print them
#--- ADD YOUR SOLUTION HERE (8 points)---


#------------------------------


# In[ ]:


# QUESTION: How did the volume of posts on the SUTD subreddit change over the years?
# Create a bar chart diagram that plots the number of submissions per year on the y-axis and the year on the x-axis.

#--- ADD YOUR SOLUTION HERE (8 points) ---

#------------------------------


# In[ ]:


# QUESTION: What is the distribution of positive, neutral and negative sentiment?
# Create a bar chart diagram that plots the number of submissions on the y-axis and the sentiment label on the x-axis.

#--- ADD YOUR SOLUTION HERE (5 points)---

#------------------------------


# In[ ]:


# QUESTION: What is the distribution of positive, neutral and negative sentiment for comments?
# Create a bar chart diagram that plots the number of comments on the y-axis and the sentiment label on the x-axis.

#--- ADD YOUR SOLUTION HERE (5 points)---

#------------------------------


# In[ ]:


# QUESTION: combine submission titles and comments for the time period from 2021 until today into one data frame.
# The resulting data frame 'df_text' should have the following schema

# Id - unique id of the comment or the submissions, this column is the index of the data frame 
# Text - text content of the comment or the submission title
# Created - date time when submission or comment was created
# Label - sentiment label as predicted by ML

#--- ADD YOUR SOLUTION HERE (10 points)---

#------------------------------


# In[ ]:


# inspect the resulting data frame
df_text


# In[ ]:


# QUESTION: sort the data frame by date time descending and save it in the same variable

#--- ADD YOUR SOLUTION HERE (3 points)---

#------------------------------


# In[ ]:


# inspect the resulting data frame
df_text


# In[ ]:


# save data frame to csv
df_text.to_csv("reddit.csv")


# Download the csv file and open it in a spreadsheet application or text editor. 
# 
# Inspec the first 10-20 entries in the list to get a feeling for the data domain.
# 
# ### QUESTION: 
# Write a short labeling guide for annotating the SUTD reddit data with sentiment labels. 
# You can write the labeling guide in a bullet point format and should have 5-10 points.
# 
# **--- ADD YOUR SOLUTION HERE (10 points)---**
# 
# 
# ------------------------------
# 

# ## Label the data
# Add a new column 'HumanLabel' to the csv file and label the 500 most recent entries, including the first 10-20 you inspected to create the label guide, using a spreadsheet application (Excel, Google Docs, Numbers) or just a text editor. 
# 
# ### QUESTION: 
# What were some of the ambiguous cases or corner cases you encountered?
# List 3-5 issues
# 
# **--- ADD YOUR SOLUTION HERE (30 points)---**
# 
# 
# 
# ------------------------------
# 
# 
# Upload your 500 labeled instances as **reddit_labeled.csv** to JupyterLab.

# ## Evaluate
# Compare your human-corrected labels with the original predicted labels.

# In[ ]:


# 
# QUESTION: Read the 500 labeled rows from the CSV file into a dataframe "df_labeled". 
# The data frame should have this schema.

# Id - unique id of the comment or the submissions, Id is the index of the data frame 
# Text - text content of the comment or the submission title
# Created - date time when submission or comment was created
# Label - sentiment label as predicted by ML
# HumanLabel - manually reviewed 'gold sentiment label'

#--- ADD YOUR SOLUTION HERE (5 points)---



#------------------------------


# In[ ]:


# check the data was loaded correctly
df_labeled


# In[ ]:


# split the labeled data into two chunks, ordered by time
df_labeled.sort_values('Created', ascending=True, inplace=True)

df_labeled1 = df_labeled[:250]
df_labeled2 = df_labeled[250:]


# In[ ]:


# check that the each split is 250 instances and that they don't overlap
df_labeled1


# In[ ]:


df_labeled2


# In[ ]:


# Compute the agreement between the predicted labels and your manually created "gold labels" in split 1. 
# Compute scores for overall accuracy as well as precision/recall/f1 score for each label class
# Print all scores 

print(sklearn.metrics.classification_report(df_labeled1["Label"], df_labeled1["HumanLabel"]))


# In[ ]:


# Compute the agreement between the predicted labels and your manually created "gold labels" in split 2. 
# Compute scores for overall accuracy as well as precision/recall/f1 score for each label class
# Print all scores 

print(sklearn.metrics.classification_report(df_labeled2["Label"], df_labeled2["HumanLabel"]))


# ## Retrain sentiment model
# 
# Now let us use the data in df_labeled1 to try improve the sentiment classifier.
# Train the Huggingface model you have chosen with the 250 examples and your human gold labels.
# 
# Start by converting the data from data frames into a 2 Huggingface datasets. 
# - dataset1 : a Huggingface dataset object which includes the data from dataframe df_labeled1
# - dataset2 : a Huggingface dataset object which includes the data from dataframe df_labeled2
# 
# 
# In each dataset, there should be the following fields
# - text : the text of the reddit submission or comment
# - label: the human gold label, encoded as integer
# 
# With these dataset we will simulate the process of improving a model in production. Dataset1 is simulating a batch of data which we observed in production, annotated and then use to improve the model. We evaluate the change on the new training data and on the next batch of production data, simulated by dataset2.
# 

# In[ ]:


def convert_label(df, pipeline):
    # drop predicted label column
    df = df.drop("Label", axis=1)
    # convert string labels to integers as column 'label' using the sentiment pipeline config
    label_id_mapping = lambda label: pipeline.model.config.label2id[label]
    df['label'] = df['HumanLabel'].apply(label_id_mapping)
    return df

df_labeled1 = convert_label(df_labeled1, sentiment_pipeline)
df_labeled2 = convert_label(df_labeled2, sentiment_pipeline)


# In[ ]:


# QUESTION: Convert the text and human labels from the data frame to a huggingface dataset format
# create a huggingface 'dataset1' from data frame 'df_labeled1' and 'dataset2' from data frame 'df_labeled2' 
#
# each dataset has the following fields
# text : the text of the reddit submission or comment
# label: the human gold label, encoded as integer

#--- ADD YOUR SOLUTION HERE (5 points)---

#------------------------------



# In[ ]:


# inspect the first example
dataset1[0]


# In[ ]:


# load tokenizer and tokenize data set
# 
# QUESTION: Load the required tokenizer from Huggingface into a variable 'tokenizer'
# Then tokenize 'dataset1' into 'tokenized_dataset1' and 'dataset2' into 'tokenized_dataset2'
# Use the Huggingface libraries. Remember that we stored the model name in a variable "model_name"

# helper function for tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

#--- ADD YOUR SOLUTION HERE (5 points)---

#------------------------------



# In[ ]:


# load Hugging model for classification initialized with the sentiment model you have chosen

#--- ADD YOUR SOLUTION HERE (3 points)---

#------------------------------
# Hint: make sure your model corresponds to your tokenizer


# In[ ]:


# add custom metrics that computes precision, recall, f1, accuracy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# In[ ]:


#
# QUESTION: configure the training parameters using the Hugginface TrainingArguments class
# - set the output directory to "finetuning-reddit"
# - do not report training metrics to an external experiment tracking service
# - learning rate to 2e-5, 
# - set weight decay to 0.01
# - set logging_steps to 10,
# - set evaluation_strategy to "steps",
# - set epochs to 3


#--- ADD YOUR SOLUTION HERE (3 points)---

#------------------------------



# In[ ]:


# initialize trainer
# train on the split dataset1
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset1,
    eval_dataset=tokenized_dataset2,
    compute_metrics=compute_metrics,
)


# In[ ]:


# Evaluate on dataset1 set before training 
predictions = trainer.predict(tokenized_dataset1)
print(sklearn.metrics.classification_report(predictions.predictions.argmax(-1), dataset1['label']))


# In[ ]:


# Evaluate on dataset2 set before training 
predictions = trainer.predict(tokenized_dataset2)
print(sklearn.metrics.classification_report(predictions.predictions.argmax(-1), dataset2['label']))


# In[ ]:


# train the model
train_output = trainer.train()


# In[ ]:


# Evaluate on dataset1, i.e the training set again
preditions = trainer.predict(tokenized_dataset1)
print(sklearn.metrics.classification_report(preditions.predictions.argmax(-1), dataset1['label']))


# In[ ]:


# Evaluate on dataset2 set i.e. the test set again
predictions = trainer.predict(tokenized_dataset2)
print(sklearn.metrics.classification_report(predictions.predictions.argmax(-1), dataset2['label']))


# ### QUESTION: 
# Has the model improved performance on the first batch of data? Does the model generalize well to the next batch of data?
# Do you see any signs of overfitting or underfitting based on the evaluation scores
# Explain why or why not
# 
# **--- ADD YOUR SOLUTION HERE (5 points)---**
# 
# 
# ------------------------------
# 

# ### QUESTION: 
# Is the model good enough to be used for practical applications?
# Given the results you have so far, what additional measures would you recommend to continuously improve the SUTD reddit sentiment classifier? What other functionalities beyond sentiment could be useful? Write a paragraph (max 200 words) to explain your choice
# 
# **--- ADD YOUR SOLUTION HERE (10 points)---**
# 
# 
# ------------------------------
# 

# # End
# 
# This concludes assignment 2.
# 
# Please submit this notebook with your answers and the generated output cells as a **Jupyter notebook file** and the **text file reddit_labeled_STUDENT_NAME.csv** via github.
# 
# 
# 1. Create a private github repository **sutd_5055mlop** under your github user.
# 2. Add your instructors as collaborator: ddahlmeier and lucainiaoge
# 3. Save your submission as assignment_02_STUDENT_NAME.ipynb and reddit_labeled_STUDENT_NAME.csv where STUDENT_NAME is your name in your SUTD email address.  
# 4. Push the submission files to your repo 
# 5. Submit the link to the repo via eDimensions
# 
# Example:<br/>
# Email: michael_tan@mymail.sutd.edu.sg<br/>
# STUDENT_NAME: michael_tan<br/>
# Submission file name: assignment_02_michael_tan.ipynb
# 
# 
# 
# **Assignment due 21 March 2025 11:59pm**
# 
# 

# 
