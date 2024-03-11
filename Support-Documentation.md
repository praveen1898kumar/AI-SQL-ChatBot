# AI SQL ChatBot

## Overview
This Python script implements an AI-powered chatbot capable of generating SQL queries based on provided metadata. It uses OpenAI's GPT-3.5 model for natural language processing and a retrieval chain to provide relevant responses.

## Setup
1. Install required dependencies:
    ```
    pip install openai langchain
    ```

2. Set up OpenAI API key:
    ```python
    import os

    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    ```

## Usage
1. Ensure you have a metadata file (e.g., `Info.txt`) containing SQL metadata.
2. Modify the script to specify the path to your metadata file in the `TextLoader` initialization.
3. Run the script:
    ```
    python chatbot.py
    ```
4. Enter your queries/prompts when prompted. Type 'quit', 'q', or 'exit' to exit the chatbot.

## Configuration
- `PERSIST`: Set to `True` to persist the index to disk.
- `query`: Set to `None` initially. Can be passed as a command-line argument to the script.

## Components
- `langchain`: A library for natural language processing tasks.
- `ChatOpenAI`: Chat model using OpenAI's GPT-3.5 model.
- `DirectoryLoader`, `TextLoader`: Loaders for metadata from directories or text files.
- `OpenAIEmbeddings`: Embedding function for OpenAI's GPT-3.5 model.
- `VectorstoreIndexCreator`, `VectorStoreIndexWrapper`: Indexing utilities.
- `ConversationalRetrievalChain`: Chain for conversational retrieval.

## Example
```python
import os 
import sys 

import openai 
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import TextLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.indexes.vectorstore import VectorStoreIndexWrapper 
from langchain.vectorstores import Chroma 

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Configuration
PERSIST = False 
query = None 

# Initialize components
loader = TextLoader("Info.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None
