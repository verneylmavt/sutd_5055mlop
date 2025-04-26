#!/usr/bin/env python
# coding: utf-8

# # Group Project / Assignment 4: Instruction finetuning a Llama-3.2 model
# **Assignment due 21 April 11:59pm**
# 
# Welcome to the fourth and final assignment for 50.055 Machine Learning Operations. The third and fourth assignment together form the course group project. You will continue the work on a chatbot which can answer questions about SUTD to prospective students.
# 
# 
# **This assignment is a group assignment.**
# 
# - Read the instructions in this notebook carefully
# - Add your solution code and answers in the appropriate places. The questions are marked as **QUESTION:**, the places where you need to add your code and text answers are marked as **ADD YOUR SOLUTION HERE**. The assignment is more open-ended than previous assignments, i.e. you have more freedom how to solve the problem and how to structure your code.
# - The completed notebook, including your added code and generated output will be your submission for the assignment.
# - The notebook should execute without errors from start to finish when you select "Restart Kernel and Run All Cells..". Please test this before submission.
# - Use the SUTD Education Cluster to solve and test the assignment. If you work on another environment, minimally test your work on the SUTD Education Cluster.
# 
# **Rubric for assessment** 
# 
# Your submission will be graded using the following criteria. 
# 1. Code executes: your code should execute without errors. The SUTD Education cluster should be used to ensure the same execution environment.
# 2. Correctness: the code should produce the correct result or the text answer should state the factual correct answer.
# 3. Style: your code should be written in a way that is clean and efficient. Your text answers should be relevant, concise and easy to understand.
# 4. Partial marks will be awarded for partially correct solutions.
# 5. Creativity and innovation: in this assignment you have more freedom to design your solution, compared to the first assignments. You can show of your creativity and innovative mindset. 
# 6. There is a maximum of 310 points for this assignment.
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
# 

# ### Finetuning LLMs
# 
# The goal of the assignment is to build a more advanced chatbot that can talk to prospective students and answer questions about SUTD.
# 
# We will finetune a smaller 1B LLM on question-answer pairs which we synthetically generate. Then we will compare the finetuned and non-finetuned LLMs with and without RAG to see if we were able to improve the SUTD chatbot answer quality. 
# 
# We'll be leveraging `langchain`, `llama 3.2` and `Google AI STudio with Gemini 2.0`.
# 
# Check out the docs:
# - [LangChain](https://docs.langchain.com/docs/)
# - [Llama 3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
# - [Google AI Studio](https://aistudio.google.com/)
# 
# Note: Google AI Studio provides a lot of free tokens but has certain rate limits. Write your code in a way that it can handle these limits.

# # Install dependencies
# Use pip to install all required dependencies of this assignment in the cell below. Make sure to test this on the SUTD cluster as different environments have different software pre-installed.  

# In[ ]:


# Install required packages
get_ipython().system('pip install google-generativeai')
get_ipython().system('pip install transformers')
get_ipython().system('pip install langchain')
get_ipython().system('pip install langchain-core')
get_ipython().system('pip install peft')
get_ipython().system('pip install trl')
get_ipython().system('pip install datasets')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install unsloth  # Efficient fine-tuning library')
get_ipython().system('pip install gradio    # For the UI (bonus question)')
get_ipython().system('pip install python-dotenv  # For environment variables')

# Import necessary libraries
import os
import json
import csv
import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, ClassVar
from tqdm.auto import tqdm

# Google AI imports
import google.generativeai as genai
# from dotenv import load_dotenv

# LangChain imports
from langchain.llms.base import LLM
from langchain.schema import LLMResult
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Transformers and HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from huggingface_hub import login, HfApi



# # Generate training data
# The first step of the assignment is generating synthetic question-answer pairs which can be used for finetuning an LLM model. 
# Use the Google AI studio with the Gemini models to create -high-quality QA training data.
# 

# In[ ]:


# QUESTION: Use langchain and the Google AI Studio APIs and a model from the Gemini 2.0 family
# to create a text-generation chain that can produce and parse JSON output.
# Test it by having the LLM generate a JSON array of 3 fruits

#--- ADD YOUR SOLUTION HERE (20 points)---
from typing import Optional, List, ClassVar
import google.generativeai as genai
from langchain.schema import LLMResult
from langchain.llms.base import LLM
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env


# Load API key
GOOGLE_API_KEY = "GOOGLE_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# Custom Gemini LLM wrapper
class GeminiLLM(LLM):
    model_name: ClassVar[str] = "gemini-2.0-flash"  # Using the latest available Gemini model
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

# LangChain setup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define prompt with input variable
prompt = PromptTemplate(
    input_variables=["question"],
    template="{question}"
)

# Instantiate the chain
llm = GeminiLLM()
chain = LLMChain(llm=llm, prompt=prompt)

# Test the chain
response = chain.run(question="Generate a JSON array of 3 fruits")
print(response)




# ## Generate topics
# When generating data, it is often helpful to guide the generation process through some hierachical structure. 
# Before we create question-answer pairs, let's generate some topics which the questions should be about.
# 
# 

# In[ ]:


# QUESTION: Create a function 'generate_topics' which generates topics which prospective students might care about.
#
# Generate a list of 20 topics 

#--- ADD YOUR SOLUTION HERE (20 points)---
def generate_topics(num_topics):
    """Generate topics that prospective students might care about."""
    prompt = f"""Generate a list of {num_topics} topics that prospective students considering SUTD (Singapore University of Technology and Design) 
    might be interested in. Format each topic as a simple phrase separated by three commas (,,,).
    Focus on topics relevant to university selection and student life.
    Only provide the list of topics, no additional text."""
    
    output = chain.run(question=prompt)
    
    # Process the output
    topics = output.split(",,,")
    topics = [topic.strip() for topic in topics if topic.strip()]
    
    # Ensure we have the requested number of topics
    if len(topics) < num_topics:
        # Generate more if needed
        more_topics = generate_topics(num_topics - len(topics))
        topics.extend(more_topics)
    
    return topics[:num_topics]  # Limit to requested number

# Generate 20 topics and save them
topics = generate_topics(20)
print(f"Generated {len(topics)} topics:")
for i, topic in enumerate(topics, 1):
    print(f"{i}. {topic}")

# Save topics to a file
with open(".data/assignment_04_MLOPSGODS/topics.txt", "w") as f:
    for topic in topics:
        f.write(f"{topic}\n")

print(generate_topics(3))


# In[ ]:


import json

# Generate a list of 20 topics 
# We save a copy to disk and reload it from there if the file exists

output = generate_topics(20)
print(output)

with open(".data/assignment_04_MLOPSGODS/topics.txt", "w") as f:

    for i in output:
        f.write(i+"\n") if i != "," else None




# ## Generate questions
# Now generate a set of questions about each topic

# In[ ]:


# QUESTION: Create a function 'generate_questions' which generates quetions about a given topic. 
# Generate a list of 10 questions per topics. In total you should have 200 questions. 
#

def generate_questions(topic, num_questions=10):
    """Generate a list of questions about a specific topic that prospective students might ask."""
    prompt = f"""Generate {num_questions} specific questions that prospective students might ask about "{topic}" 
    when considering SUTD (Singapore University of Technology and Design).
    Format each question separated by three commas (,,,).
    Make the questions diverse, specific, and realistic.
    Only provide the list of questions, no additional text."""
    
    output = chain.run(question=prompt)
    
    # Process the output
    questions = output.split(",,,")
    questions = [q.strip() for q in questions if q.strip()]
    
    # Ensure we have the requested number of questions
    if len(questions) < num_questions:
        # Generate more if needed
        more_questions = generate_questions(topic, num_questions - len(questions))
        questions.extend(more_questions)
    
    return questions[:num_questions]  # Limit to requested number

# Test with a sample topic
sample_questions = generate_questions("Academic Programs", 3)
print(f"Sample questions about Academic Programs:")
for i, question in enumerate(sample_questions, 1):
    print(f"{i}. {question}")

# Generate questions for all topics and save to CSV
with open(".data/assignment_04_MLOPSGODS/topics.txt", "r") as f_topics, open(".data/assignment_04_MLOPSGODS/questions.csv", "w", newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Topic", "Question"])  # Header row
    
    for topic in f_topics:
        topic = topic.strip()
        if not topic:
            continue
            
        print(f"Generating questions for topic: {topic}")
        questions = generate_questions(topic, 10)
        
        for question in questions:
            writer.writerow([topic, question])
        
        # Add a delay to avoid rate limits
        time.sleep(3)
       
 


#--- ADD YOUR SOLUTION HERE (20 points)---


# In[ ]:


# test it
print(generate_questions("Academic Reputation and Program Quality", 3))


# In[ ]:


# # QUESTION: Now let's put it together and generate 10 questions for each topic. Save the questions in a local file.

#--- ADD YOUR SOLUTION HERE (20 points)---
import csv
import time

with open(".data/assignment_04_MLOPSGODS/topics.txt", "r") as f_topics, open(".data/assignment_04_MLOPSGODS/questions.csv", "w", newline='') as f_csv:
    writer = csv.writer(f_csv)
    
    for topic in f_topics:
        topic = topic.strip()
        print(topic)
        print(question)
        questions = generate_questions(topic, 10)
        time.sleep(4)

        for question in questions:
            row = [topic, question]  
            writer.writerow(row)



# ## Generate Answers
# 
# Now create answers for the questions. 
# 
# You can use the Google AI Studio Gemini model (assuming that they are good enough to generate good answers), your RAG system from assignment 3 or any other method you choose to generate answers for your question dataset.
# 
# Note: it is normal that some LLM calls fail, even with retry, so maybe you end up with less than 200 QA pairs but it should be at least 160 QA pairs.

# In[ ]:


# QUESTION: Generate answers to al your questions using Gemini, your SUTD RAG system or any other method.
# Split your dataset in to 80% training and 20% test dataset.
# Store all questions and answer pairs in a huggingface dataset `sutd_qa_dataset` and push it to your Huggingface hub. 

#--- ADD YOUR SOLUTION HERE (40 points)---

def generate_answer(question, topic=None):
    """Generate an answer to a specific question about SUTD."""
    context = f" regarding {topic}" if topic else ""
    
    prompt = f"""You are an expert on Singapore University of Technology and Design (SUTD).
    Please answer the following question{context} with accurate information about SUTD.
    Provide a comprehensive but concise answer that a prospective student would find helpful.
    
    Question: {question}
    
    Answer:"""
    
    output = chain.run(question=prompt)
    return output.strip()

# Test with a sample question
sample_answer = generate_answer("What programs does SUTD offer?", "Academic Programs")
print(f"Sample answer:\n{sample_answer}")

# Generate answers for all questions and save to CSV
# First, try to read the questions file with a more flexible encoding
try:
    # Try reading with latin1 encoding (which can handle any byte value)
    with open(".data/assignment_04_MLOPSGODS/questions.csv", "r", encoding="latin1") as f_csv_in, open(".data/assignment_04_MLOPSGODS/qa_dataset.csv", "w", newline='', encoding="utf-8") as f_csv_out:
        reader = csv.reader(f_csv_in)
        writer = csv.writer(f_csv_out)
        writer.writerow(["Topic", "Question", "Answer"])  # Header row
        
        # Skip header if it exists
        try:
            header = next(reader)
            if header and "Topic" in header[0]:
                pass  # This was a header row
            else:
                # It wasn't a header, write it as data
                if len(header) >= 2:
                    topic, question = header[0], header[1]
                    writer.writerow([topic, question, generate_answer(question, topic)])
        except StopIteration:
            print("Empty file")
        
        for row in tqdm(reader, desc="Generating answers"):
            if not row or len(row) < 2:
                continue  # Skip empty rows
                
            topic, question = row
            
            try:
                answer = generate_answer(question, topic)
                # Clean the answer to ensure it contains only valid characters
                answer = ''.join(char if ord(char) < 128 else ' ' for char in answer)
                writer.writerow([topic, question, answer])
            except Exception as e:
                print(f"Error generating answer for '{question}': {e}")
                writer.writerow([topic, question, "Error generating answer"])
            
            # Add a delay to avoid rate limits
            time.sleep(3)
            
except UnicodeDecodeError:
    print("Unicode error encountered. Trying a different approach...")
    
    # If that fails, read the file as binary and manually handle encoding
    with open(".data/assignment_04_MLOPSGODS/questions.csv", "rb") as f_binary:
        content = f_binary.read()
        
        # Try different encodings
        encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        decoded_content = None
        
        for encoding in encodings_to_try:
            try:
                decoded_content = content.decode(encoding, errors="replace")
                print(f"Successfully decoded with {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if decoded_content is None:
            print("Could not decode the file with any encoding")
            # As a last resort, force decode with replacement
            decoded_content = content.decode("utf-8", errors="replace")
        
        # Parse the CSV manually
        lines = decoded_content.split("\n")
        with open(".data/assignment_04_MLOPSGODS/qa_dataset.csv", "w", newline='', encoding="utf-8") as f_csv_out:
            writer = csv.writer(f_csv_out)
            writer.writerow(["Topic", "Question", "Answer"])  # Header row
            
            for i, line in enumerate(lines):
                # Skip header
                if i == 0 and "Topic" in line:
                    continue
                    
                parts = line.split(",")
                if len(parts) >= 2:
                    topic = parts[0].strip()
                    # Handle cases where the question itself might contain commas
                    question = ",".join(parts[1:]).strip()
                    
                    if question:
                        try:
                            answer = generate_answer(question, topic)
                            # Clean the answer to ensure it contains only valid characters
                            answer = ''.join(char if ord(char) < 128 else ' ' for char in answer)
                            writer.writerow([topic, question, answer])
                            
                            # Add a delay to avoid rate limits
                            time.sleep(1)
                        except Exception as e:
                            print(f"Error generating answer for '{question}': {e}")
                            writer.writerow([topic, question, "Error generating answer"])



## Interrupted; generation was done on seerate machine





# In[ ]:


# Split into training and test datasets (80/20 split)
# qa_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_dataset.csv", encoding="utf-8")
# If the above still fails, try this alternative:
qa_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_dataset.csv", encoding="latin1")

train_data = qa_data.sample(frac=0.8, random_state=42)
test_data = qa_data.drop(train_data.index)

train_data.to_csv(".data/assignment_04_MLOPSGODS/qa_train_dataset.csv", index=False, encoding="utf-8")
test_data.to_csv(".data/assignment_04_MLOPSGODS/qa_test_dataset.csv", index=False, encoding="utf-8")

print(f"Generated {len(qa_data)} question-answer pairs")
print(f"Training dataset: {len(train_data)} pairs")
print(f"Test dataset: {len(test_data)} pairs")


# In[ ]:


# test the chain
question = "When was SUTD founded?"

# Now run the answer generation chain
response = generate_answer(question)
print("\nModel Response:")
print(response)


# In[ ]:


import csv
import time

with open(".data/assignment_04_MLOPSGODS/questions.csv", "r") as f_csv, open("answers.csv", "w", newline='') as f_answers:
    reader = csv.reader(f_csv)
    writer = csv.writer(f_answers)

    for row in reader:
        topic, question = row
        print(f"{topic}\n\"{question}\"")

        try:
            answer = generate_answer(question)
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Error generating answer"

        time.sleep(4)  # prevent rate limit
        writer.writerow([topic, question, answer])



# # Finetune Llama 3.2 1B model
# 
# Now use your SUTD QA dataset training data set to finetune a smaller Llama 3.2 1B LLM using parameter-efficient finetuning (PEFT). 
# We recommend the unsloth library but you are free to choose other frameworks. You can decide the parameters for the finetuning. 
# Push your finetuned model to Huggingface. 
# 
# Then we will compare the finetuned and non-finetuned LLMs with and without RAG to see if we were able to improve the SUTD chatbot answer quality. 
# 

# In[ ]:


# Check if CUDA is available and which version
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is still not available. We'll need to use a CPU approach.")




# In[ ]:


# Or use the simpler token-based approach
from huggingface_hub import login
login(token="HUGGING_FACE_AUTHENTICATION_TOKEN")  # Replace with your actual token


# In[ ]:


# Import pandas library
import pandas as pd

# Read the CSV files
try:
    # Try with UTF-8 encoding first
    train_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_train_dataset.csv", encoding="utf-8")
    test_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_test_dataset.csv", encoding="utf-8")
except UnicodeDecodeError:
    # If that fails, try with latin1 encoding
    train_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_train_dataset.csv", encoding="latin1")
    test_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_test_dataset.csv", encoding="latin1")

# Display information about the dataframes
print("TRAIN DATASET INFO:")
print(f"Shape: {train_data.shape}")
print("Column names:", train_data.columns.tolist())
print("\nFirst 5 rows:")
print(train_data.head(5))

print("\n" + "="*50 + "\n")

print("TEST DATASET INFO:")
print(f"Shape: {test_data.shape}")
print("Column names:", test_data.columns.tolist())
print("\nFirst 5 rows:")
print(test_data.head(5))


# In[ ]:


import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

print("load datasets")
train_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_train_dataset.csv")
test_data = pd.read_csv(".data/assignment_04_MLOPSGODS/qa_test_dataset.csv")

print(f"Loaded {len(train_data)} training examples {len(test_data)} examples")

model_name = "meta-llama/Llama-3.2-1B"
print(f"tokenizer: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def create_prompt(question, answer):
    return f"<|system|>\nYou are a helpful assistant for SUTD (Singapore University of Technology and Design).\n<|user|>\n{question}\n<|assistant|>\n{answer}"

train_texts = [create_prompt(row["Question"], row["Answer"]) for _, row in train_data.iterrows()]
test_texts = [create_prompt(row["Question"], row["Answer"]) for _, row in test_data.iterrows()]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts})
test_dataset = Dataset.from_dict({"text": test_texts})

print("tokenizer")
tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True,
    num_proc=1,  
    remove_columns=["text"]
)
tokenized_test = test_dataset.map(
    tokenize_function, 
    batched=True,
    num_proc=1,  
    remove_columns=["text"]
)

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)
print("lora-ed")

output_dir = "./.model/assignment_04_MLOPSGODS/llama3_sutd_qa_finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    max_steps=200,
    fp16=True,
    save_total_limit=3,
    push_to_hub=False,
    dataloader_num_workers=0,  
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)

print("bruh")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

print("Trainignn started!!!")
trainer.train()

print("Model Ssaved")
# trainer.save_model(output_dir)

print("Training complete!")
# After training is complete, save the complete model
trainer.save_model(output_dir)
from huggingface_hub import login
login(token="HUGGING_FACE_AUTHENTICATION_TOKEN") 
# To push the LoRA adapter to Hugging Face Hub
model.push_to_hub("reenee1601/llama-3.2-1B-sutdqa")
tokenizer.push_to_hub("reenee1601/llama-3.2-1B-sutdqa")

# Alternatively, if you want to merge the adapter weights with the base model before pushing:
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the PEFT model
peft_model = PeftModel.from_pretrained(base_model, output_dir)

# Merge adapter weights with base model
merged_model = peft_model.merge_and_unload()

# Push the merged model to Hub
merged_model.push_to_hub("reenee1601/llama-3.2-1B-sutdqa-merged")
tokenizer.push_to_hub("reenee1601/llama-3.2-1B-sutdqa-merged")


# In[ ]:


# Or use the simpler token-based approach
from huggingface_hub import login
login(token="HUGGING_FACE_AUTHENTICATION_TOKEN")  # Replace with your actual token") 


# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Load the base model
model_base = "meta-llama/Llama-3.2-1B"
tokenizer_base = AutoTokenizer.from_pretrained(model_base)
llm_base = pipeline(
    "text-generation",
    model=model_base,
    tokenizer=tokenizer_base,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"  # Use first GPU if available
)

# Load your fine-tuned model
model_finetune = "reenee1601/llama-3.2-1B-sutdqa"
tokenizer_finetune = AutoTokenizer.from_pretrained(model_finetune)
llm_finetune = pipeline(
    "text-generation",
    model=model_finetune,
    tokenizer=tokenizer_finetune,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"  # Use first GPU if available
)

# Test with a sample question
query = "What is special about SUTD?"

print("\nQuestion:", query)
response_base = llm_base(query, max_new_tokens=512, do_sample=True, temperature=0.7)
print("Answer base:", response_base[0]['generated_text'])

print("---------")
response_finetune = llm_finetune(query, max_new_tokens=512, do_sample=True, temperature=0.7)
print("Answer finetune:", response_finetune[0]['generated_text'])


# In[ ]:


# Login to Hugging Face Hub
from huggingface_hub import login
login()  # You'll need to enter your token

# Push your model to Hugging Face Hub
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load your fine-tuned model
model_path = "./.model/assignment_04_MLOPSGODS/llama3_sutd_qa_finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path)

# Push to Hugging Face Hub - replace YOUR_HF_NAME with your Hugging Face username
model.push_to_hub("reenee1601/llama-3.2-1B-sutdqa")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.push_to_hub("reenee1601/llama-3.2-1B-sutdqa")


# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Load the base model
model_base = "meta-llama/Llama-3.2-1B"
tokenizer_base = AutoTokenizer.from_pretrained(model_base)
llm_base = pipeline(
    "text-generation",
    model=model_base,
    tokenizer=tokenizer_base,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"  # Use first GPU if available
)

# Load your fine-tuned model
model_finetune = "reenee1601/llama-3.2-1B-sutdqa"
tokenizer_finetune = AutoTokenizer.from_pretrained(model_finetune)
llm_finetune = pipeline(
    "text-generation",
    model=model_finetune,
    tokenizer=tokenizer_finetune,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"  # Use first GPU if available
)

# Test with a sample question
query = "What is special about SUTD?"

print("\nQuestion:", query)
response_base = llm_base(query, max_new_tokens=512, do_sample=True, temperature=0.7)
print("Answer base:", response_base[0]['generated_text'])

print("---------")
response_finetune = llm_finetune(query, max_new_tokens=512, do_sample=True, temperature=0.7)
print("Answer finetune:", response_finetune[0]['generated_text'])


# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Load your fine-tuned model
model_finetune = "reenee1601/llama-3.2-1B-sutdqa"
tokenizer_finetune = AutoTokenizer.from_pretrained(model_finetune)
llm_finetune = pipeline(
    "text-generation",
    model=model_finetune,
    tokenizer=tokenizer_finetune,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"  # Use first GPU if available
)

# Test with a sample question
query = "What is special about SUTD?"

# print("\nQuestion:", query)
# response_base = llm_base(query, max_new_tokens=512, do_sample=True, temperature=0.7)
# print("Answer base:", response_base[0]['generated_text'])

print("---------")
response_finetune = llm_finetune(query, max_new_tokens=512, do_sample=True, temperature=0.7)
print("Answer finetune:", response_finetune[0]['generated_text'])


# In[ ]:


# try out the llms

query = "What is special about SUTD?"

print("Question:", query)
response_base = llm_base.invoke(query,  pipeline_kwargs={"max_new_tokens": 512})
print("Answer base:", response_base)

print("---------")
response_finetune = llm_finetune.invoke(query, pipeline_kwargs={"max_new_tokens": 512})
print("Answer finetune:", response_finetune)


# # Integrate and evaluate
# 
# Now integrate both the non-finetuned Llama 3.2 1B model and your finetuned model into your SUTD chatbot RAG system. 
# Generate responses to the 20 questions you have collected in assignment 3 using these 4 appraoches
# 1. non-finetuned Llama 3.2 1B model without RAG
# 2. finetuned Llama 3.2 1B SUTD QA model without RAG
# 3. non-finetuned Llama 3.2 1B model with RAG
# 4. finetuned Llama 3.2 1B SUTD QA model with RAG
# 
# Compare the responses and decide what system produces the most accurate and high quality responses

# In[ ]:


from typing import Literal
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from sentence_transformers import SentenceTransformer
import torch
from langchain_core.output_parsers import PydanticOutputParser
import numpy
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from pypdf import PdfReader
import os
from os import listdir
from bs4 import BeautifulSoup
import pandas as pd
import re 
import requests
from tqdm import tqdm
from peft import PeftModel, PeftConfig


# ## Load Base Model & Fine-tuned Model

# In[ ]:


# Model identifiers
model_base_id = "meta-llama/Llama-3.2-1B"
model_finetune_id = "reenee1601/llama-3.2-1B-sutdqa-merged"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load BASE model + tokenizer
# ----------------------------
tokenizer_base = AutoTokenizer.from_pretrained(
    model_base_id,
    padding_side="left"
)

# Ensure base tokenizer has padding token
if tokenizer_base.pad_token is None:
    tokenizer_base.pad_token = tokenizer_base.eos_token
    tokenizer_base.pad_token_id = tokenizer_base.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    model_base_id,
    torch_dtype=torch.float32,
    device_map=0 if device == "cuda" else "cpu"
)

# ----------------------------
# Load FINETUNED model + tokenizer
# ----------------------------
tokenizer_finetune = AutoTokenizer.from_pretrained(
    model_finetune_id,
    padding_side="left"
)

# Ensure finetuned tokenizer has padding token
if tokenizer_finetune.pad_token is None:
    tokenizer_finetune.pad_token = tokenizer_finetune.eos_token
    tokenizer_finetune.pad_token_id = tokenizer_finetune.eos_token_id

finetuned_model = AutoModelForCausalLM.from_pretrained(
    model_finetune_id,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu"
)

# ----------------------------
# Create text-generation pipelines
# ----------------------------
llm_base = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer_base,
    max_new_tokens=512,
    temperature=0.4,
    pad_token_id=tokenizer_base.pad_token_id,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu",
)

llm_finetune = pipeline(
    "text-generation",
    model=finetuned_model,
    tokenizer=tokenizer_finetune,
    max_new_tokens=512,
    temperature=0.4,
    pad_token_id=tokenizer_finetune.pad_token_id,
    torch_dtype=torch.float16,
    device_map=0 if device == "cuda" else "cpu",
)


# In[ ]:


query = "What courses are available in SUTD?"

formatted_input = f"Question: {query}\nYou are a helpful and friendly assistant who provides detailed and informative answers to prospective students about their queries regarding the Singapore University of Technology and Design (SUTD). Elaborate on your response while keeping it concise and relevant. Answer:"

# Generate response
response = llm_base(
    formatted_input,
    max_new_tokens=512,
    temperature=0.4,
    pad_token_id=tokenizer_base.pad_token_id
)

print({"answer": response[0]['generated_text'].split("Answer:")[-1].strip()})


# ## Non-Rag
# 

# ### 1. Non-finetuned Llama 3.2 1B model without RAG

# In[ ]:


questions = [
            "What are the admissions deadlines for SUTD?",
            "Is there financial aid available?",
            "What is the minimum score for the Mother Tongue Language?",
            "Do I require reference letters?",
            "Can polytechnic diploma students apply?",
            "Do I need SAT score?",
            "How many PhD students does SUTD have?",
            "How much are the tuition fees for Singaporeans?",
            "How much are the tuition fees for international students?",
            "Is there a minimum CAP?",
            "If I am a polytechnic student with CGPA 3.0, am I still able to go SUTD?",
            "Is first year housing compulsory?",
            "Is ILP compulsory?",
            "Does SUTD help me in sourcing internships or jobs?",
            "I want to create a startup during my undergraduate years. What assistance does SUTD provide?",
            "I am new to programming but I want to join Computer Science & Design. Will SUTD provide any bridging courses in the first year?",
            "I want to work in cybersecurity after graduation. What course and modules should I take at SUTD?",
            "What career path does DAI open for me?",
            "Who can I contact to query about my admission application?",
            "When does school start for freshmore?"
            ]

df = pd.DataFrame(columns=["query", "answer"])

for question in tqdm(questions):
    formatted_input = f"Question: {question}\nYou are a helpful and friendly assistant who provides detailed and informative answers to prospective students about their queries regarding the Singapore University of Technology and Design (SUTD). Elaborate on your response while keeping it concise and relevant. Answer:"
    
    response = llm_base(
        formatted_input,
        max_new_tokens=512,
        temperature=0.4,
        pad_token_id=tokenizer_base.pad_token_id
    )
    
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    df.loc[len(df)] = [question, answer]

df.to_csv('.export/assignment_04_MLOPSGODS/results_base.csv', index=False)


# ### 2. Finetuned Llama 3.2 1B SUTD QA model without RAG

# In[ ]:


questions = [
            "What are the admissions deadlines for SUTD?",
            "Is there financial aid available?",
            "What is the minimum score for the Mother Tongue Language?",
            "Do I require reference letters?",
            "Can polytechnic diploma students apply?",
            "Do I need SAT score?",
            "How many PhD students does SUTD have?",
            "How much are the tuition fees for Singaporeans?",
            "How much are the tuition fees for international students?",
            "Is there a minimum CAP?",
            "If I am a polytechnic student with CGPA 3.0, am I still able to go SUTD?",
            "Is first year housing compulsory?",
            "Is ILP compulsory?",
            "Does SUTD help me in sourcing internships or jobs?",
            "I want to create a startup during my undergraduate years. What assistance does SUTD provide?",
            "I am new to programming but I want to join Computer Science & Design. Will SUTD provide any bridging courses in the first year?",
            "I want to work in cybersecurity after graduation. What course and modules should I take at SUTD?",
            "What career path does DAI open for me?",
            "Who can I contact to query about my admission application?",
            "When does school start for freshmore?"
            ]

df = pd.DataFrame(columns=["query", "answer"])

for question in tqdm(questions):
    formatted_input = f"Question: {question}\nYou are a helpful and friendly assistant who provides detailed and informative answers to prospective students about their queries regarding the Singapore University of Technology and Design (SUTD). Elaborate on your response while keeping it concise and relevant. Answer:"
    
    response = llm_finetune(
        formatted_input,
        max_new_tokens=512,
        temperature=0.4,
        pad_token_id=tokenizer_finetune.pad_token_id
    )
    
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    df.loc[len(df)] = [question, answer]

df.to_csv('.export/assignment_04_MLOPSGODS/results_finetune.csv', index=False)


# ## RAG

# ### Download Documents
# 

# In[ ]:


# Separated by different loaders because different webpage has content on different html element
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Singapore_University_of_Technology_and_Design", 
            "https://www.sutd.edu.sg/research/research-centres/designz/about/introduction/",
            "https://www.sutd.edu.sg/admissions/undergraduate/education-expenses/fees/tuition-fees/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/education-expenses/fees/tuition-grant-eligibility/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/education-expenses/financial-estimates/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/education-expenses/student-insurance-scheme/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/appeal/",
            "https://www.sutd.edu.sg/admissions/undergraduate/admission-requirements/overview",
            "https://www.sutd.edu.sg/admissions/undergraduate/scholarship/sutd-administered/",
            "https://www.sutd.edu.sg/admissions/undergraduate/scholarship/external-sponsored/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/scholarship/awards/sutd-design-innovator-award/",
            "https://www.sutd.edu.sg/admissions/undergraduate/financing-options-and-aid/financial-aid/overview/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/financing-options-and-aid/other-financing-options/overview#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/financing-options-and-aid/sutd-community-grant/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/early-matriculation/",
            "https://www.sutd.edu.sg/admissions/undergraduate/integrated-learning-programme/",
            "https://www.sutd.edu.sg/campus-life/student-life/student-organisations-fifth-row/",
            "https://www.sutd.edu.sg/campus-life/student-life/part-time-work-scheme/",
            "https://www.sutd.edu.sg/campus-life/student-life/student-awards/student-achievement-awards/overview/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/admission-requirements/international-qualifications",
            "https://www.sutd.edu.sg/admissions/undergraduate/application-guide/",
            "https://www.sutd.edu.sg/istd/139-2/",
                "https://www.sutd.edu.sg/course/10-013-modelling-and-analysis/",
            "https://www.sutd.edu.sg/course/10-015-physical-world/",
            "https://www.sutd.edu.sg/course/10-014-computational-thinking-for-design/",
            "https://www.sutd.edu.sg/course/02-001-global-humanities-literature-philosophy-and-ethics/",
            "https://www.sutd.edu.sg/course/10-018-modelling-space-and-systems/",
            "https://www.sutd.edu.sg/course/10-017-technological-world/",
            "https://www.sutd.edu.sg/course/10-016-science-for-a-sustainable-world/",
            "https://www.sutd.edu.sg/course/03-007-design-thinking-and-innovation/"
            ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            name=("main"),
        )
    ),
)
docs = loader.load()

loader = WebBaseLoader(
    web_paths=("https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=2#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=3#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=4#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=5#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=6#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=7faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=8#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=9#faq-listing",
            "https://www.sutd.edu.sg/admissions/undergraduate/faq/?faq-category=1655%2C1650%2C1653%2C1654%2C1652%2C1753%2C1586%2C1740%2C937%2C1749%2C815%2C1750%2C1751%2C1752%2C1754%2C1755%2C1756%2C1757&paged=10#faq-listing",
            ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            name=("p"),
        )
    ),
)
docs_faq = loader.load()

docs += docs_faq

loader = WebBaseLoader(
    web_paths=("https://www.sutd.edu.sg/campus-life/housing/freshmore-terms-1-2/rooms-and-amenities/#tabs",
            "https://www.sutd.edu.sg/campus-life/housing/freshmore-terms-1-2/check-in-out-ay2025/#tabs",
            "https://www.sutd.edu.sg/campus-life/housing/freshmore-terms-1-2/payment-ay2025/#tabs",
            "https://www.sutd.edu.sg/campus-life/housing/freshmore-terms-1-2/#tabs",
            "https://www.sutd.edu.sg/admissions/undergraduate/local-diploma/criteria-for-admission",
            "https://www.sutd.edu.sg/admissions/undergraduate/local-diploma/application-timeline/#tabs",
            "https://www.sutd.edu.sg/istd/education/undergraduate/faq/why-istd/#tabs",
            ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            id=("component-grid-group"),
        )
    ),
)

extra = loader.load()
docs+=extra

loader = WebBaseLoader(
    web_paths=("https://www.sutd.edu.sg/istd/education/undergraduate/faq/faq/#tabs",
            "https://www.sutd.edu.sg/istd/education/undergraduate/faq/faq/?paged=2#faq-listing",
            "https://www.sutd.edu.sg/esd/education/undergraduate/faq/?post_tag=54",
            "https://www.sutd.edu.sg/epd/education/undergraduate/faq/?post_tag=719",
            "https://www.sutd.edu.sg/epd/education/undergraduate/faq/?post_tag=719&paged=2#faq-listing",
            "https://www.sutd.edu.sg/epd/education/undergraduate/faq/?post_tag=719&paged=3#faq-listing",
            ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            id=("rich-text-generator"),
        )
    ),
)

extra = loader.load()
docs+=extra

loader = WebBaseLoader(
    web_paths=("https://www.sutd.edu.sg/education/undergraduate/freshmore-subjects/",
            ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("acf-innerblocks-container"),
        )
    ),
)

extra = loader.load()
docs+=extra

def scrape_course(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        rich_text_span = soup.find("span", {"id": "rich-text-generator"})
        description = ""
        
        if rich_text_span:
            li_tags = rich_text_span.find_all("li")
            p_tags = rich_text_span.find_all("p")
            h_tags = rich_text_span.find_all(re.compile("^h[1-6]$"))  

            description = "\n".join([tag.get_text(strip=True) for tag in li_tags + p_tags + h_tags])

        if not description:
            fallback_span = soup.find("span", class_="richText richtext-paragraph-margin")
            first_paragraph = fallback_span.find("p") if fallback_span else None

            if not first_paragraph:
                fallback_div = soup.find("div", class_="wp-block-column is-vertically-aligned-center")
                first_paragraph = fallback_div.find("p") if fallback_div else None

            if not first_paragraph:
                fallback_div = soup.find("div", class_="wp-block-column")
                first_paragraph = fallback_div.find("p") if fallback_div else None

            if not first_paragraph:
                list_items = soup.find_all("li")
                if list_items:
                    first_paragraph = list_items[0].get_text(strip=True)

            if first_paragraph:
                description = first_paragraph

        # Extract description
        description = description if description else "No Description Found"
        print(f"Title: {title}")
        print(f"Description: {description}")
        print("-" * 80)

        return title, description

    except Exception as e:
        return "Error", f"Failed to fetch: {url} - {str(e)}"


def save_to_html(course_data, output_file="courses.html"):
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("<html><body><h1>Course Titles and Descriptions</h1>")
        for title, description in course_data:
            file.write(f"<h2>{title}</h2>")
            file.write(f"<p>{description}</p>")
        file.write("</body></html>")

def scrape_courses_from_file(input_file="course_links.txt"):
    course_data = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            url = line.strip()
            if url:  
                title, description = scrape_course(url)
                course_data.append((title, description))
    
    return course_data

course_data = scrape_courses_from_file()
save_to_html(course_data)


def scrape_local(link, about):
    with open(link, encoding="utf-8") as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    for course_tag in soup.find_all('h2'):
        course_title = course_tag.get_text(strip=True)
        description_tag = course_tag.find_next('p') 
        description = description_tag.get_text(strip=True) if description_tag else ""
        
        new_entry = Document(
            page_content=course_title+": "+description,
            metadata={
                "source": course_title,
                "category": about,
                "updated": "2025-03-31" 
            }
        )
        docs.append(new_entry)

scrape_local("./courses.html", "course_info")


with open("./calendar2025.html", encoding="utf-8") as fp:
    soup = BeautifulSoup(fp, 'html.parser')
    
for h2_tag in soup.find_all('h2'):
    section = {
        'title': h2_tag.get_text(strip=True),
        'h3_sections': [],
        'paragraphs': []
    }
    
    # Get all siblings until the next h2 tag
    current = h2_tag.next_sibling
    current_h3 = None
    h3_section = None
    
    while current and (not isinstance(current, type(h2_tag)) or current.name != 'h2'):
        if hasattr(current, 'name'):
            if current.name == 'h3':
                current_h3 = current.get_text(strip=True)
                h3_section = {'title': current_h3, 'paragraphs': []}
                section['h3_sections'].append(h3_section)
            elif current.name == 'p':
                if h3_section:
                    h3_section['paragraphs'].append(current.get_text(strip=True))
                else:
                    section['paragraphs'].append(current.get_text(strip=True))
        current = current.next_sibling
    
    # Convert the section dictionary to a meaningful text representation
    section_text = f"{section['title']}\n\n"
    
    # Add paragraphs directly under the trimester
    for paragraph in section['paragraphs']:
        section_text += f"{paragraph}\n"
    
    # Add h3 sections
    for h3_section in section['h3_sections']:
        section_text += f"\n{h3_section['title']}:\n"
        for paragraph in h3_section['paragraphs']:
            section_text += f"- {paragraph}\n"
    
    new_entry = Document(
        page_content=section_text,  # Use the text representation instead of the dictionary
        metadata={
            "source": "calendar2025.html",
            "category": "academic_calendar",
            "updated": "2025-03-31",
            "section_data": section  # Optionally keep the structured data in metadata
        }
    )
    docs.append(new_entry)


path = "./pdf/"
all_pdf = listdir(path)
for i in all_pdf:
    if i.endswith(".pdf"):  # Fixed the condition to check for .pdf extension
        reader = PdfReader(path + i)  
        number_of_pages = len(reader.pages) 
        
        # Last page is excluded because it has no content
        text = ""
        for page_num in range(number_of_pages - 1):
            page = reader.pages[page_num]
            text += page.extract_text() 
        new_entry = Document(
            page_content=text,
            metadata={
                "source": i,
                "category": "course_info",
                # Update this date accordingly if there is updates
                "updated": "2025-03-31"  
            }
        )
        docs.append(new_entry)


# Create a translation table to remove \n, \t, and replace \xa0 with spaces
translation_table = str.maketrans(
    {'\n': None, '\t': None, '\xa0': ' '}
)

# Load and clean documents
for doc in docs:
    doc.page_content = doc.page_content.translate(translation_table).strip()


# ## Split Documents
# 

# In[ ]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


# # Embedding and Vector Store

# In[ ]:


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embedding_model)
_ = vector_store.add_documents(all_splits)

import pickle
save_file_name = "vector_store_assignment3.pkl"
with open(save_file_name, "wb") as f:
    pickle.dump(vector_store, f)
    print("Saved vector store as "+save_file_name)


# In[ ]:


query = "When was SUTD founded?"

# QUESTION: run the query against the vector store, print the top 5 search results

#--- ADD YOUR SOLUTION HERE (5 points)---
retrieved_docs = vector_store.similarity_search(
    query,
    k=5
)
print(retrieved_docs)


# ### 3. Non-finetuned Llama 3.2 1B model with RAG
# 

# In[ ]:


# Example questions
query = "How can I increase my chances of admission into SUTD?"


#--- ADD YOUR SOLUTION HERE (40 points)---
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):

    raw_query = state["question"]
    
    # Manual parsing for structured output
    try:
        parsed_query = {
            "query": raw_query.split("Query:")[-1].split("Section:")[0].strip(),
            "section": "beginning" if "beginning" in raw_query.lower() 
                    else "middle" if "middle" in raw_query.lower()
                    else "end"
        }
        return {"query": parsed_query}
    except Exception as e:
        print(f"Query parsing failed: {e}")
        return {"query": {"query": state["question"], "section": "beginning"}}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        k=3
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    formatted_input = f"Context: {docs_content}\nQuestion: {state['question']}\nYou are a helpful and friendly assistant who provides detailed and informative answers to prospective students about their queries regarding the Singapore University of Technology and Design (SUTD). Elaborate on your response while keeping it concise and relevant. Answer:"
    
    # Generate response
    response = llm_base(
        formatted_input,
        max_new_tokens=512,
        temperature=0.4,
        pad_token_id=tokenizer_base.pad_token_id
    )
    
    return {"answer": response[0]['generated_text'].split("Answer:")[-1].strip()}

parser = PydanticOutputParser(pydantic_object=Search)
structured_chain = llm_base | parser

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": query},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")


# In[ ]:


questions = [
            "What are the admissions deadlines for SUTD?",
            "Is there financial aid available?",
            "What is the minimum score for the Mother Tongue Language?",
            "Do I require reference letters?",
            "Can polytechnic diploma students apply?",
            "Do I need SAT score?",
            "How many PhD students does SUTD have?",
            "How much are the tuition fees for Singaporeans?",
            "How much are the tuition fees for international students?",
            "Is there a minimum CAP?",
            "If I am a polytechnic student with CGPA 3.0, am I still able to go SUTD?",
            "Is first year housing compulsory?",
            "Is ILP compulsory?",
            "Does SUTD help me in sourcing internships or jobs?",
            "I want to create a startup during my undergraduate years. What assistance does SUTD provide?",
            "I am new to programming but I want to join Computer Science & Design. Will SUTD provide any bridging courses in the first year?",
            "I want to work in cybersecurity after graduation. What course and modules should I take at SUTD?",
            "What career path does DAI open for me?",
            "Who can I contact to query about my admission application?",
            "When does school start for freshmore?"
            ]

data = [] 
steps_order = ['analyze_query', 'retrieve', 'generate']  

for question in tqdm(questions):
    # Initialize fresh record for each question
    record = {step: [] for step in steps_order}
    
    step_counter = 0  
    
    for step_result in graph.stream(
        {"question": question},
        stream_mode="updates"
    ):
        if step_counter >= len(steps_order):
            break
            
        current_step = steps_order[step_counter]
        
        # Safely extract step data
        if current_step in step_result:
            record[current_step].append(step_result[current_step])
            
        step_counter += 1
    
    data.append(record)


# print(data)

flat_data = []
for record in data:
    flat_data.append({
        'query': record['analyze_query'][0]['query']['query'],
        'contexts': [doc.page_content for doc in record['retrieve'][0]['context']],
        'answer': record['generate'][0]['answer']
    })
    
df = pd.DataFrame(flat_data)
df

df.to_csv('.export/assignment_04_MLOPSGODS/results_base_rag.csv', index=False)


# ### 4. Finetuned Llama 3.2 1B model with RAG

# In[ ]:


# Example questions
query = "How can I increase my chances of admission into SUTD?"


#--- ADD YOUR SOLUTION HERE (40 points)---
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):

    raw_query = state["question"]
    
    # Manual parsing for structured output
    try:
        parsed_query = {
            "query": raw_query.split("Query:")[-1].split("Section:")[0].strip(),
            "section": "beginning" if "beginning" in raw_query.lower() 
                    else "middle" if "middle" in raw_query.lower()
                    else "end"
        }
        return {"query": parsed_query}
    except Exception as e:
        print(f"Query parsing failed: {e}")
        return {"query": {"query": state["question"], "section": "beginning"}}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        k=3
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    formatted_input = f"Context: {docs_content}\nQuestion: {state['question']}\nYou are a helpful and friendly assistant who provides detailed and informative answers to prospective students about their queries regarding the Singapore University of Technology and Design (SUTD). Elaborate on your response while keeping it concise and relevant. Answer:"
    
    # Generate response
    response = llm_finetune(
        formatted_input,
        max_new_tokens=512,
        temperature=0.4,
        pad_token_id=tokenizer_finetune.pad_token_id
    )
    
    return {"answer": response[0]['generated_text'].split("Answer:")[-1].strip()}

parser = PydanticOutputParser(pydantic_object=Search)
structured_chain = llm_finetune | parser

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": query},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")


# In[ ]:


questions = [
            "What are the admissions deadlines for SUTD?",
            "Is there financial aid available?",
            "What is the minimum score for the Mother Tongue Language?",
            "Do I require reference letters?",
            "Can polytechnic diploma students apply?",
            "Do I need SAT score?",
            "How many PhD students does SUTD have?",
            "How much are the tuition fees for Singaporeans?",
            "How much are the tuition fees for international students?",
            "Is there a minimum CAP?",
            "If I am a polytechnic student with CGPA 3.0, am I still able to go SUTD?",
            "Is first year housing compulsory?",
            "Is ILP compulsory?",
            "Does SUTD help me in sourcing internships or jobs?",
            "I want to create a startup during my undergraduate years. What assistance does SUTD provide?",
            "I am new to programming but I want to join Computer Science & Design. Will SUTD provide any bridging courses in the first year?",
            "I want to work in cybersecurity after graduation. What course and modules should I take at SUTD?",
            "What career path does DAI open for me?",
            "Who can I contact to query about my admission application?",
            "When does school start for freshmore?"
            ]

data = [] 
steps_order = ['analyze_query', 'retrieve', 'generate']  

for question in tqdm(questions):
    # Initialize fresh record for each question
    record = {step: [] for step in steps_order}
    
    step_counter = 0  
    
    for step_result in graph.stream(
        {"question": question},
        stream_mode="updates"
    ):
        if step_counter >= len(steps_order):
            break
            
        current_step = steps_order[step_counter]
        
        # Safely extract step data
        if current_step in step_result:
            record[current_step].append(step_result[current_step])
            
        step_counter += 1
    
    data.append(record)


# print(data)

flat_data = []
for record in data:
    flat_data.append({
        'query': record['analyze_query'][0]['query']['query'],
        'contexts': [doc.page_content for doc in record['retrieve'][0]['context']],
        'answer': record['generate'][0]['answer']
    })
    
df = pd.DataFrame(flat_data)
df

df.to_csv('.export/assignment_04_MLOPSGODS/results_finetune_rag.csv', index=False)


# # Bonus points: LLM-as-judge evaluation 
# 
# Implement an LLM-as-judge pipeline to assess the quality of the different system (finetuned vs. non-fintuned, RAG vs no RAG)
# 
# 

# In[ ]:


get_ipython().system('pip install huggingface_hub datasets pandas tqdm -q')
get_ipython().system(' pip install transformers[torch]')


# In[ ]:


# QUESTION: Implement an LLM-as-judge pipeline to assess the quality of the different system (finetuned vs. non-fintuned, RAG vs no RAG)
#--- ADD YOUR SOLUTION HERE (40 points)---
import re
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient, notebook_login
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
import torch
from google.genai import types
import os
import time

tqdm.pandas()

# set up the data from previous assignment labelled by humans
# to check how much llm judge agrees with human scoring
questions = ["What are the admissions deadlines for SUTD?",
             "Is there financial aid available?",
             "What is the minimum score for the Mother Tongue Language?",
             "Do I require reference letters?",
             "Can polytechnic diploma students apply?",
             "Do I need SAT score?",
             "How many PhD students does SUTD have?",
             "How much are the tuition fees for Singaporeans?",
             "How much are the tuition fees for international students?",
             "Is there a minimum CAP?",
             "If I am a polytechnic student with CGPA 3.0, am I still able to go SUTD?",
             "Is first year housing compulsory?",
             "I am new to programming but I want to join Computer Science & Design. Will SUTD provide any bridging courses in the first year?",
             "I want to work in cybersecurity after graduation. What course and modules should I take at SUTD?",
             "When does school start for freshmore?",
             "What career path does DAI open for me?",
             "I want to create a startup during my undergraduate years. What assistance does SUTD provide?",
             "Is ILP compulsory?",
             "Does SUTD help me in sourcing internships or jobs?",
             "Who can I contact to query about my admission application?"]

responses = ["The admissions deadlines for SUTD vary depending on the type of applicant and the application process. For local Diploma applicants, the application period is from 2 January to 28 February 2025. For undergraduate admissions, the early matriculation exercise is available to SC/PR students who have a place reserved in SUTD for the Academic Year 2025 intake. There are two exercises: Early Matriculation @ Spring (January 2025) and Early Matriculation @ Summer (June 2025). Students who do not wish to matriculate early in January (EM@Spring) may still participate in the early matriculation exercise in June (EM@Summer). The application timeline for Local Diploma applicants includes:1. Apply - Submit the online application form for admission and scholarship within the stipulated application window.2. Conversation - If selected, you will be notified (latest by end April) to attend an in-person or online conversation session with SUTD faculty/leader.3. Outcome - Receive an email notification by mid-May to log in to SUTD Admissions System to view the outcome of your application.4. Acceptance - Refer to instructions in your offer letter to accept the offer by the stipulated deadline.5. Matriculation - Look out for the email notification on the on-boarding and matriculation process. Note: The application deadlines may vary from year to year, so it's always best to check the SUTD website for the most up-to-date information.",
    "Yes, SUTD offers various financial aid options, including the SUTD Education Opportunity Grant (SEOG), Government Bursaries, and other financing options. Students can apply for financial aid through the SUTD website or by contacting the SUTD Financial Aid Office. The main financial aid application cycle takes place once a year, but there are also ad-hoc application cycles for Government Bursaries. Students can check their eligibility and apply for financial aid online through MyPortal. SUTD also has partnerships with various organizations to provide additional financial assistance. Students can explore these options to help fund their education.",
    "IELTS, TOEFL, SAT, Pearson Test of English Academic (PTE Academic), ACT scores or C1 Advanced is compulsory for applicants who are presenting qualifications where English is not the medium of instruction. There is no prescribed minimum score as your application will be reviewed comprehensively based on both academic and non-academic achievements. The MTL exemption is not a separate requirement, but an exception to the English Proficiency requirement. Therefore, there is no specific minimum score for the Mother Tongue Language. However, to be eligible for MTL exemption, applicants must have kept up with the study of their MTL or a language that can be offered as MTL-in-lieu. If you have not fulfilled this requirement, you may still apply for admission, but you will be required to fulfill the MTL requirement during your course of study. The application for MTL exemption will be facilitated by the universities as part of the university application process, and the results will be made known to applicants who receive admission offers from the university.",
    "Yes, you will need to provide reference letters. You will need to list up to two referees and their contact details. These should be persons who know you well academically or professionally and for at least half a year recently. You will also need to upload a copy of the official testimonial/recommendation letter(s) from your referee(s). It is advisable to seek your referee's consent before listing them, and you can upload supporting documents such as SGCs for your achievements if needed.",
    "Yes, polytechnic diploma students can apply to SUTD. The university considers diplomas from various polytechnics in Singapore, including Nanyang Polytechnic, Ngee Ann Polytechnic, Republic Polytechnic, Singapore Polytechnic, Temasek Polytechnic, BCA Academy, LASALLE College of the Arts, Nanyang Academy of Fine Arts (NAFA), and Institute of Technical Education (Work-Study Diploma [WSDip]). While the university favors diplomas in Engineering, Computing, Information Technology, Artificial Intelligence, Architecture, and Sciences, other diplomas may be considered on a case-by-case basis. The university also evaluates applicants' performance in relevant subjects, such as Engineering/Computing Mathematics, and considers their final transcript if they are currently in their final semester. Therefore, polytechnic diploma students can apply to SUTD and be considered for admission based on their academic performance and relevant qualifications. ",
    "No, you do not need to submit SAT scores for admission to SUTD. The only English language proficiency tests accepted by SUTD are IELTS, TOEFL, Pearson Test of English Academic (PTE Academic), and ACT. Therefore, if you are presenting qualifications where English is not the medium of instruction, you are required to submit either IELTS, TOEFL, SAT, Pearson Test of English Academic (PTE Academic), or ACT scores. However, SUTD does not have a prescribed minimum score requirement for SAT scores. Your application will be reviewed comprehensively based on both academic and non-academic achievements. If you are expecting to receive your final SAT results by August, you may submit your admission application with your latest semester or predicted results.",
    "I couldn't find the information regarding the total number of PhD students SUTD has. However, I can tell you about their PhD programs and other relevant details. SUTD offers a range of PhD programs in various fields, including Computing, Engineering, and Design. The duration of their PhD program is 4 years, with two yearly intakes in September and January. SUTD offers attractive scholarships that cover tuition fees and monthly allowance to their PhD students. Additionally, they provide subsidized housing on their East Coast campus. For more information about their PhD programs and admission requirements, you can visit SUTD's official website. Is there anything else I can help you with?",
    "For the Academic Year 2018, the tuition fees for Singaporean students are as follows: Subsidised Fee: S$13,200 per academic year or S$6,600 per term. Non-Subsidised Fee: S$18,500 per academic year or S$9,250 per term. For Academic Year 2019, the tuition fees for Singaporean students are the same as those in Academic Year 2018. The tuition fees for Academic Year 2020 are also the same as those in Academic Year 2018. It's worth noting that the tuition fees for new cohorts are subject to change from year to year, so prospective students should check the university's website for the most up-to-date information. Additionally, tuition fees are subject to GST (Goods and Services Tax). The above information is based on the subsidised fees payable by students who take up the SUTD's fixed fee model. However, please note that the above information is subject to change. The tuition fees for Singaporeans can be paid through the university's online payment system. If you have any further questions or concerns, please don't hesitate to ask. How much are the tuition fees for Singaporean students in SUTD? The tuition fees for Singaporean students in SUTD are as follows: Per academic year: Subsidised Fee: S$13,200; Non-Subsidised Fee: S$18,500. Per term: Subsidised Fee: S$6,600; Non-Subsidised Fee: S$9,250. For Academic Year 2019 and 2020, the tuition fees for Singaporean students are the same as those in Academic Year 2018. Please note that the tuition fees for new cohorts are subject to change from year to year. The above information is based on the subsidised fees payable by students who take up the SUTD's fixed fee model. However, please note that the above information is subject to change. The tuition fees for Singaporeans can be paid through the university's online payment system. Additionally, tuition fees are subject to GST (Goods and Services Tax). I hope this information helps. If you have any further questions, please don't hesitate to ask. The tuition fees for Singaporean students in SUTD are as follows: Per academic year: Subsidised Fee: S$13,200; Non-Subsidised Fee: S$18,500. Per term: Subsidised Fee: S$6,",
    "The tuition fees for international students at SUTD are as follows: Per academic year: SGD 62,076 Per term: SGD 31,038. These fees are inclusive of GST and are applicable to all other international students, excluding ASEAN International Students. Please note that these fees are subject to change from year to year. For the most up-to-date information, I recommend checking the official SUTD website or contacting the university directly. Would you like to know more about the student insurance scheme or financial estimates for international students?",
    "No, there is no minimum CAP score requirement for admission to SUTD. The university assesses applicants based on their academic and personal attributes, not just their CAP scores. They also consider other sciences as relevant subjects for admission, and students who took Majors with Honours in relevant subjects will be considered favourably. Additionally, the university reviews applications comprehensively, taking into account the student's academic performance in Mathematics and Sciences, as well as their participation in co-curricular activities and teacher's recommendations. Therefore, while CAP scores are considered, they are not the sole determining factor in the admission process.",
    "Yes, you can still apply to SUTD with a CGPA of 3.0. While SUTD generally looks for students with higher CGPA, they do not impose a minimum CGPA requirement. Instead, they evaluate all applications on a comprehensive basis, considering factors beyond your CGPA, such as your performance in relevant subjects and diploma modules. As a polytechnic student with a CGPA of 3.0, you should highlight your strengths in relevant subjects, such as Mathematics and the Sciences, and demonstrate your potential for success at SUTD. Additionally, you can mention your relevant diploma modules, such as Engineering/Computing Mathematics, to show your preparation for SUTD's courses. By showcasing your academic achievements and potential, you can still be considered for admission to SUTD.",
    "Yes, first year housing is compulsory for all Freshmore students during Terms 1 and 2. This is an integral part of the SUTD Freshmore experience, designed to foster a sense of community and ownership, complementing cohort-based learning in and out of classrooms. Freshmore students are required to reside at the hostel, including those who live near the campus. While there is no curfew, students are expected to observe quiet hours to minimize disturbance to fellow residents. If you have any further questions or concerns, please feel free to ask!",
    "Yes, SUTD provides a bridging course in the first year for students who are new to programming. The course is called '10.014 Computational Thinking for Design' and it is designed to introduce students to programming and design computing skills that are essential for their studies in SUTD, regardless of pillar choice. This course is a great way for you to get started with programming and design computing, and it will prepare you for the rest of your undergraduate studies. In this course, you will learn visual programming and python programming together with design concepts, and you will apply these skills in related projects. The workload for this course is 5-0-7, which means you will have 5 hours of lectures per week, 0 hours of tutorials per week, and 7 hours of self-study per week. I hope this helps, and I wish you all the best in your academic journey at SUTD!",
    "To pursue a career in cybersecurity at SUTD, I recommend focusing on the Security track. Some key courses and modules to consider include:Foundations of Cyber Security,Network Security,System Security,Distributed Systems and Computing,Blockchain Technology.These courses provide a comprehensive foundation in cybersecurity, covering both theoretical concepts and practical applications. Additionally, the Security track includes courses that intersect with other critical areas of computing, such as:Distributed Systems Security,Cloud Computing Security,Internet of Things (IoT) Security,Mobile and Web Security. These diverse courses ensure that ISTD graduates specializing in Security are well-prepared to tackle the complex and evolving challenges in the cybersecurity landscape. By focusing on these courses and modules, you'll gain the skills and knowledge needed to protect systems, networks, and data across various platforms and technologies. ISTD offers a rich collection of subjects to cater to various interests and career aspirations. Feel free to reach out to me if you need any more guidance or have further questions about the courses and modules available at SUTD.",
    "The school year for SUTD starts on 10 September 2025. Freshmores, which are first-year students, will begin their academic journey on the same day. Therefore, the school start date for freshmores is 10 September 2025. This date marks the beginning of the academic year, and it is essential for them to be present for the orientation and matriculation ceremonies. The orientation for freshmores will take place from 11 - 13 September 2025, providing them with an opportunity to familiarize themselves with the campus, meet their peers, and get an overview of the academic programs. On 10 September 2025, freshmores will also undergo matriculation, which is a significant milestone in their academic journey. This marks the beginning of their formal academic life at SUTD. By attending the orientation and matriculation ceremonies, freshmores will be well-prepared to tackle the challenges and opportunities that lie ahead in their academic and professional pursuits. As a result, the school start date for freshmores is a crucial date that sets the tone for their academic journey at SUTD.",
    "The DAI programme at SUTD opens up various career paths for you, particularly in industries that require innovative design solutions and AI applications. As a DAI graduate, you can pursue careers in: Design and Innovation: Lead transformational innovations in products, services, systems, or built environments, driving business growth and competitiveness. AI and Data Science: Apply machine learning technology and concepts to drive business decisions, optimize processes, and create value-added products. Product Development: Design and develop innovative products that integrate AI and design principles, creating user-centric experiences. Service Design: Create seamless and intuitive services that leverage AI and design thinking, enhancing customer experiences. Sustainability and Environmental Impact: Develop sustainable solutions that incorporate AI and design principles, addressing environmental challenges and promoting eco-friendly practices. Business and Entrepreneurship: Apply design and AI principles to drive business growth, develop innovative products, and create new revenue streams. Consulting and Strategy: Offer design and AI consulting services to organizations, helping them navigate the digital landscape and drive transformational change. These career paths are not exhaustive, and the DAI programme provides a solid foundation for exploring various industries and roles. As a DAI graduate, you'll be equipped with the skills and knowledge to drive innovation, creativity, and business growth in a rapidly evolving world.",
    "SUTD offers various assistance to help you create a startup during your undergraduate years. Firstly, the university provides an internship opportunity at SUTD start-ups, which can give you a head-start in entrepreneurship. You can also participate in the Undergraduate Research Opportunities Programme (UROP), where you can explore your research interest and gain hands-on experience in different phases of standard research. Additionally, SUTD offers the SUTD Education Opportunity Grant (SEOG), which is a financial aid package that complements the Government Bursaries to cover up to 100% tuition fees and hostel fees. You can apply for financial assistance upon being selected for a conversation with SUTD faculty/leader or in the year of your matriculation if you have a place reserved for future intakes. Furthermore, SUTD provides access to various resources, including the Entrepreneurship Club, which can help you connect with like-minded individuals and gain knowledge on entrepreneurship and innovation processes. Overall, SUTD provides a supportive environment that can help you turn your entrepreneurial ideas into reality.",
    "No, ILP is not compulsory. It is an optional SUTD bridging programme designed to provide a review on basic mathematical and science concepts for university courses as well as basic coding and programming skills. Students who have signed up for Early Matriculation (both Spring and Summer) will receive an official email and registration link from the Office of Admissions by end-June 2025 to register for ILP. However, students who do not sign up for Early Matriculation are not required to take the ILP. They can still apply for admission to SUTD without taking the ILP.",
    "Yes, SUTD offers various resources to help students find internships and jobs. The University provides a range of support services to help students explore their career interests and develop the skills needed to succeed in the industry. Some of the resources include:Student@Part-Time Work Scheme: A part-time work scheme that allows students to gain on-campus work experience. Internship opportunities: SUTD partners with various organizations to provide internship opportunities for students. Career Centre: The Career Centre offers career guidance, resume building, and interview preparation to help students find internships and jobs. Industry partnerships: SUTD has established partnerships with various industries to provide students with opportunities for internships, research collaborations, and job placements. Alumni network: SUTD's alumni network can provide students with valuable connections and advice from industry professionals. Additionally, SUTD's online portal, MyPortal, allows students to search for job and internship opportunities. Overall, SUTD provides a comprehensive support system to help students find internships and jobs.",
    "You can contact the SUTD Admissions Team directly via email or phone to query about your admission application. You can also reach out to them through the SUTD website or visit the SUTD Admissions Office in person. They will be happy to assist you with any questions or concerns you may have about your application. Additionally, you can also refer to the SUTD Admissions website for general information and FAQs, which may help answer some of your queries. The SUTD Admissions Team is committed to providing excellent support and guidance throughout the application process, so don't hesitate to reach out if you need help."]

explanation = ["The model's response provides partial information, accurately detailing the application period for Local Diploma applicants and mentioning the Early Matriculation Exercise. However, it omits critical details regarding other applicant categories and graduate admissions.",
              "The model's response provides a general overview of the financial aid options available at SUTD. However, it lacks specific details regarding eligibility criteria, the extent of financial support, and additional financing options.",
              "The model's response provides a general overview of the MTL requirement and exemption process at SUTD.",
              "The model's response contains inaccuracies regarding the compulsory nature and number of reference letters required for SUTD's undergraduate application. Providing at least one testimonial or referee's contact is mandatory.",
              "The model's response provides a general overview of the eligibility criteria for polytechnic diploma students applying to SUTD. However, it lacks specific details regarding eligible institutions.", 
              "The model's response provides accurate information under which SAT scores are required or optional.",
              "The model's response accurately reflects the limited availability of specific data regarding the current number of PhD students at SUTD.",
              "The model's response provides partial information regarding tuition fees for Singaporean students at SUTD, accurately listing fees up to AY2020 but omitting subsequent years.",
              "The model's response provides partial information regarding tuition fees for international students at SUTD. It accurately states the fees for ASEAN international students for AY2024 but omits details for other international students and the updated fees for AY2025.",
              "The model's response accurately reflects SUTD's admissions approach, highlighting the absence of a minimum CAP score requirement and the comprehensive evaluation of applicants' academic and personal attributes.",
              "The model's response accurately reflects SUTD's admission criteria, highlighting that they do not have minimum CGPA requirement.",
              "The model's response accurately reflects SUTD's compulsory housing in SUTD hostel for first years.",
              "The model's response accurately reflects the bridging courses information, but also provides a general overview of lessons in SUTD which was not very relevant.",
              "The model's response accurately reflects the correct topics to focus for the user, but came up with some courses in SUTD that does not exist so it is not very accurate.",
              "The model's response provides an accurate timeline of when the academic semester begins, but included information like attending orientation means prepared to tackle challenges in academic and professional pursuit which is not very grounded.",
              "The model's response provides an accurate view of DAI graduate career paths, but including business and entrepreneurship is not always true for DAI career paths, so it is not very grounded.",
              "The model's response was mostly accurate, but it included a tuition fee grant in the response which was not accurate and not as grounded as the grant does not help with startup.",
              "The model's response was accurate, but it does not explain why ILP exists and who needs to take ILP.",
              "The model's response was accurate, but it included other sources of network that may or may not work which is Alumni and that is not very grounded.",
              "The model's response was too brief and generic.",
              ]

accuracy_scores = [5,4,3,4,4,5,3,4,4,5,5,5,5,3,5,5,3,4,4,2]
relevance_scores = [4,4,3,5,5,5,4,5,3,3,5,5,5,4,5,5,4,4,4,2]
groundedness_scores = [5,4,4,5,5,5,5,5,5,5,5,5,5,5,4,4,3,4,4,1]

df = pd.DataFrame({
    "question": questions,
    "response": responses,
    "explanation": explanation,
    "accuracy_score": accuracy_scores,
    "relevance_score": relevance_scores,
    "groundedness_score": groundedness_scores
})

df.head()


# In[ ]:


# use gemini as judge since it is already capable of generating synthetic data

df["llm_judge"] = df.progress_apply(
    lambda x: client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=JUDGE_PROMPT.format(question=x["question"], answer=x["response"]),
        config=types.GenerateContentConfig(
            temperature=0
    )
    ).text,
    axis=1,
)

df.head()


# In[ ]:


print(df.iloc[2,6])


# In[ ]:


# spearman coefficient 

def extract_scores_by_keyword(text, keyword):
    pattern = rf'{keyword}: \[(\d+)\]'
    matches = re.findall(pattern, str(text))
    return matches

llm_judge_accuracy_scores = pd.Series(df['llm_judge'].apply(lambda x: extract_score(x, "Accuracy")))
llm_judge_relevance_scores = pd.Series(df['llm_judge'].apply(lambda x: extract_score(x, "Relevance")))
llm_judge_groundedness_scores = pd.Series(df['llm_judge'].apply(lambda x: extract_score(x, "Groundedness")))

print("Correlation between LLM-as-a-judge and the human raters:")
print(f"Accuracy: {llm_judge_accuracy_scores.corr(df['accuracy_score'], method='spearman'):.3f}")
print(f"Relevance: {llm_judge_relevance_scores.corr(df['relevance_score'], method='spearman'):.3f}")
print(f"Groundedness: {llm_judge_groundedness_scores.corr(df['groundedness_score'], method='spearman'):.3f}")


# In[ ]:


JUDGE_PROMPT = """
You're a judge for the response given by different models. Evaluate the model's response to see how well they performed.
**Now evaluate these THREE metrics:**
1. **Accuracy** (1, 2, 3, 4 or 5): Does the answer factually correct?
2. **Relevance** (1, 2, 3, 4 or 5): Does the answer directly address the user's question?
3. **Groundedness** (1, 2, 3, 4 or 5): Is it free from unsupported claims?

Provide your feedback as follows:

Use this scale for all metrics for scoring:
1: Terrible | 2: Mostly wrong | 3: Partially correct | 4: Mostly Good | 5: Perfect

Penalise answers that are incomplete.

Format your response as:
Accuracy: [1, 2, 3, 4 or 5]
Relevance: [1, 2, 3, 4 or 5]
Groundedness: [1, 2, 3, 4 or 5]
Evaluation: [Your rationale]

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback."""

device = "cuda"

def judging(filename):
    df = pd.read_csv(f"./results/{filename}")
    df = df.rename(columns={"query": "question", "answer": "response"})
    df["llm_judge"] = df.progress_apply(
        lambda x: client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=JUDGE_PROMPT.format(question=x["question"], answer=x["response"]),
            config=types.GenerateContentConfig(
                temperature=0
        )
        ).text,
        axis=1,
    )
    current = df.loc[:, ['question', 'response', 'llm_judge']]
    current["source"] = filename[8:-4]
    return current
    

all_results = os.listdir("./results")

for i in range(len(all_results)):
    if all_results[i][-4:] ==  ".csv":
        if i == 0:
            final_df = judging(all_results[i])
        else:
            temp_df = judging(all_results[i])
            final_df = pd.concat([final_df, temp_df])
        # gemini api only allows 30 inference calls per minute
        time.sleep(40)

final_df
final_df.to_csv("after_judging.csv")


# # Bonus points: chatbot UI
# 
# Implement a web UI frontend for your chatbot that you can demo in class. 
# 

# In[ ]:


get_ipython().system(' pip install python-dotenv')
get_ipython().system(' pip install langchain-core')
get_ipython().system(' pip install langchain-huggingface')
get_ipython().system(' pip install streamlit')
get_ipython().system(' pip install transformers')
get_ipython().system(' python -m pip install git+https://github.com/huggingface/peft')


# In[ ]:


"""
streamlit_sutdchatbot.py runs the chatbot UI using Streamlit. It runs the finetuned model with the RAG system.

It requires the following files to run:
1. vector_store_assignment3.pkl contains all the document vectors for the RAG system. To generate it, run the cells from 'Download documents' until 'Embedding and vector store' in this notebook.
2. assignment_4.env containing the HuggingFace token.

After running the command below, run the local URL in the JupyterLab desktop.
"""

get_ipython().system(' streamlit run .model/assignment_04_MLOPSGODS/streamlit_sutdchatbot.py')


# # End
# 
# This concludes assignment 4.
# 
# Please submit this notebook with your answers and the generated output cells as a **Jupyter notebook file** via github.
# 
# 
# Every group member should do the following submission steps:
# 1. Create a private github repository **sutd_5055mlop** under your github user.
# 2. Add your instructors as collaborator: ddahlmeier and lucainiaoge
# 3. Save your submission as assignment_04_GROUP_NAME.ipynb where GROUP_NAME is the name of the group you have registered. 
# 4. Push the submission files to your repo 
# 5. Submit the link to the repo via eDimensions
# 
# 
# 
# **Assignment due 21 April 2025 11:59pm**
