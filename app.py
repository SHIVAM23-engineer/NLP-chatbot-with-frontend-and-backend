from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a Flask app
app = Flask(__name__)

# Define the QA chain (initialize this once and use it in the /send route)
qa_chain = None

def initialize_chatbot():
    global qa_chain

    # Step 1: Load the scraped data JSON file
    from langchain_community.document_loaders import JSONLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.llms import Ollama  # Updated import statement
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate

    loader = JSONLoader(
        file_path='C:/Users/yahoo/OneDrive/Desktop/project sample ready/Data Ingestion/scraped_data.json',
        jq_schema='.content',
        text_content=False
    )
    text_documents = loader.load()

    # Step 2: Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(text_documents)

    # Step 3: Generate embeddings using OllamaEmbeddings
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="gemma")

    # Step 4: Create a FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 5: Create a retriever and an LLM instance
    llm = Ollama(model="gemma")  # This import is now from langchain_community

    # Step 6: Create a custom prompt template
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

    # Step 7: Initialize the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Using the "stuff" chain type
        retriever=retriever,
        return_source_documents=True,  # This will also return the source documents for reference
        chain_type_kwargs={"prompt": prompt_template}  # Pass the custom prompt here
    )

# Step 8: Create a function to interact with the chatbot
def chat_with_bot(query):
    if qa_chain is None:
        return "Chatbot is not initialized"
    
    # Call the QA chain with the user's query
    response = qa_chain({"query": query})
    
    # Extract the answer from the 'result' key
    answer = response['result']
    
    # Optionally, return the source documents as well
    source_documents = response.get('source_documents', [])
    
    return answer

# Define the Flask route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    data = request.json
    user_message = data.get('message')

    # Use the chat_with_bot function to get a response from the chatbot
    bot_response = chat_with_bot(user_message)

    # Return the response as JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    initialize_chatbot()  # Initialize the chatbot on startup
    app.run(debug=True)
