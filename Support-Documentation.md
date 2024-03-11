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

```markdown
# Code Explanation

## Import Necessary Modules

```python
import os 
import sys 
```

These lines import the `os` and `sys` modules, which are used for interacting with the operating system and system-specific parameters and functions, respectively.

## Import OpenAI Library

```python
import openai 
```

This line imports the `openai` library, which provides access to OpenAI's API for natural language processing tasks.

## Import Classes and Functions from langchain Library

```python
from langchain.chains import ConversationalRetrievalChain, RetrievalQA 
from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import DirectoryLoader, TextLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.indexes.vectorstore import VectorStoreIndexWrapper 
from langchain.llms import OpenAI 
from langchain.vectorstores import Chroma 
```

These lines import various classes and functions from the `langchain` library, which appears to be a custom library for handling natural language processing tasks, including conversation handling, document loading, embeddings, indexing, and vector stores.

## Set OpenAI API Key

```python
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

This line sets the OpenAI API key as an environment variable, allowing access to the OpenAI API.

## Set up Flag for Persistence

```python
PERSIST = False 
```

This line initializes a flag `PERSIST` to `False`, indicating whether to persist the index. It can be changed to `True` to enable index persistence.

## Initialize Query Variable

```python
query = None 
```

This line initializes the variable `query` to `None`, which will be used to store user input or command-line arguments.

## Check if Command-Line Arguments are Provided

```python
if len(sys.argv) > 1: 
    query = sys.argv[1]
```

This block checks if command-line arguments are provided and assigns the first argument to the `query` variable if available.

## Check if Persistence Flag is Enabled and Index Exists

```python
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("/Users/praveen18kumar/Downloads/Info.txt")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])
```

This block checks if index persistence is enabled and if the index exists. If persistence is enabled and the index exists, it reuses the existing index. Otherwise, it initializes a new index using the specified metadata file.

## Create Conversational Retrieval Chain

```python
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)
```

This line creates a conversational retrieval chain using the specified chat model and retriever, which utilizes the initialized index.

## Initialize Chat History

```python
chat_history = []
```

This line initializes an empty list `chat_history` to store the conversation history.

## Main Loop for Interaction

```python
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append((query, result['answer']))
    query = None
```

This block starts a main loop for interaction. It prompts the user for input if no query is provided, checks if the user wants to quit, retrieves a response from the chain based on the user query, prints the response, adds the query and response to the chat history, and resets the query for the next iteration.

