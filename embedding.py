import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir: ",current_dir)
file_path = os.path.join(current_dir, "books", "war_and_peace.txt")
print("file_path: ",file_path)
persistent_directory = os.path.join(current_dir, "db", "chroma_db_war_and_peace")
print("persistent_directory: ",persistent_directory)
# Define the directory containing the text file
# print("file path : ", type(file_path))
        
# print("file path : ", (file_path))

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # # Read the text content from the file
    loader = TextLoader(file_path,encoding="utf-8") 
    # documents = loader.load()
    # try:
    documents = loader.load()
    # except RuntimeError as e:
    #     print(f"Error loading file: {e}")


    # Split the document into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # docs = text_splitter.split_documents(documents)
    rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
    rec_char_docs = rec_char_splitter.split_documents(documents)
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(rec_char_docs)}")
    print(f"Sample chunk:\n{rec_char_docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        rec_char_docs, embeddings, persist_directory=persistent_directory)  
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
