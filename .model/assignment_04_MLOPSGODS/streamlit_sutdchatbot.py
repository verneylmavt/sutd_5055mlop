from peft import PeftConfig, PeftModel
import pickle
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

# #TODO: add your huggingface token
# load_dotenv("assignment4.env") 
# access_token_read = os.getenv("HF_ACCESS_TOKEN")  
# login(token = access_token_read)


# App title and configuration
st.set_page_config(page_title="Llama-3.2-1B SUTDQA Chatbot")
st.title("Llama-3.2-1B SUTDQA Chatbot")
# st.logo("sutd-logo.png", size="large")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vector store only once
@st.cache_resource
def load_vectorstore():
    # Documents for RAG retrieval
    with open("vector_store_assignment3.pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

# Load model and tokenizer only once
@st.cache_resource
def load_model(max_new_tokens:int=512):
        # Step 1: Load PEFT config
        peft_model_id = "reenee1601/llama-3.2-1B-sutdqa"
        peft_config = PeftConfig.from_pretrained(peft_model_id)
    
        # Step 2: Load base model and tokenizer
        model_base = "meta-llama/Llama-3.2-1B"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_base,
            torch_dtype=torch.float16,
            device_map=0 if device == "cuda" else "cpu"
        )
    
        # Step 3: Load LoRA adapter on top of the base model
        model = PeftModel.from_pretrained(base_model, peft_model_id)
    
        # Step 4: Merge adapter into base model
        model = model.merge_and_unload()
    
        # Step 5: Create pipeline using the merged model
        llm_finetune = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch.float16,
            device_map=0 if device == "cuda" else "cpu"
        )
        return llm_finetune
    
# pipe = load_model()
# vector_store = load_vectorstore()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user's message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_docs = vector_store.similarity_search(
        user_input,
        k=5
    )
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # print(docs_content)
    formatted_input = f"""Context: {docs_content} \nQuestion: {user_input} \nInstructions: - Answer in 2-3 concise sentences (under 200 words) - Focus on SUTD-specific details from the context - Avoid repetition \nAnswer:"""


    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipe(formatted_input)[0]["generated_text"]
            # Extract only the assistant's response
            bot_reply = response.split("Answer:")[-1].strip()
            st.markdown(bot_reply)
            st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
