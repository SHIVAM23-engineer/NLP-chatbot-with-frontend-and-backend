##Data Ingestion
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# Step 2: Load the scraped data JSON file

loader = JSONLoader(
    file_path='C:/Users/yahoo/OneDrive/Desktop/project sample ready/Data Ingestion/scraped_data.json',
    jq_schema='.content',
    text_content=False)

text_documents = loader.load()
# Step 4: Split the documents into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(text_documents)
documents[:5]

# Step 5: Generate embeddings using OllamaEmbeddings

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="gemma",
)
# Step 6: Create a FAISS vectorstore to store and retrieve embeddings
import faiss
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Step 7: Create a retriever to fetch relevant documents based on queries

from langchain.llms import Ollama
from langchain.chains import RetrievalQA

llm = Ollama(model="gemma")  # Replace "valid-model-name" with an actual model name
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# Step 8: Define a custom prompt for the chatbot's response
custom_prompt = """
You are a highly knowledgeable chatbot, and your task is to answer user questions based on the following documents.
Respond in a concise and informative manner.

Documents:
{context}

Question: {question}

Answer:
"""
prompt_template = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)
# Step 10: Set up the chatbot with OpenAI for response generation using the custom prompt
llm = Ollama(model="gemma")  
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Using the "stuff" chain type
    retriever=retriever,
    return_source_documents=True,  # This will also return the source documents for reference
    chain_type_kwargs={"prompt": prompt_template}  # Pass the custom prompt here
)
def chat_with_bot(query):
    # Call qa_chain directly as it supports multiple outputs
    response = qa_chain({"query": query})  # or use qa_chain([{"query": query}])
    
    # Extract the answer from the 'result' key
    answer = response['result']
    
    # Optionally, you can also access the source documents if needed
    source_documents = response.get('source_documents', [])
    
    # Return just the answer for now
    return answer
# Example usage of the chat_with_bot function
user_input = "Tell me about the text."
response = chat_with_bot(user_input)

# Print the chatbot's response
if isinstance(response, list):
    # Assuming the first item in the list is the response dictionary
    response_dict = response[0]
    answer = response_dict.get('result', 'No result found')
    print(answer)
else:
    print("Unexpected response format:", response)