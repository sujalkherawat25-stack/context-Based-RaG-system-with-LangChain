import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Load environment variables from .env
load_dotenv()
# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir: ",current_dir)
file_path = os.path.join(current_dir, "books", "war_and_peace.txt")
print("file_path: ",file_path)
if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # # Read the text content from the file
loader = TextLoader(file_path,encoding="utf-8") 
    # documents = loader.load()
documents = loader.load()
print("documents :" , type(documents))
# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_war_and_peace")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
# query = "How can I learn more about LangChain?"


# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
# relevant_docs = retriever.invoke(query)


# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use six sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)
# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continue_chat():
    """
    Interactive chat function that answers queries based on embeddings or AI's own knowledge.
    Includes the ability to summarize the chat history.
    """
    print("Start chatting with the AI! Type 'exit' to end the conversation or 'summarize chat' to get a summary of the chat or 'summarize document' to get a summary of the provided document.")
    chat_history = []  # List to maintain chat history

    while True:
        query = input("You: ")
        
        if query.lower() == "exit":
            print("Ending the chat. Goodbye!")
            break

        if query.lower() == "summarize chat":
            # Summarize the chat history
            summarize_chat(chat_history)
            # print(f"Summary of chat:\n{summary}")
            continue
        # if query.lower() == "summarize document":
        #     # Summarize the chat history
        #     summarize_document()
        #     # print(f"Summary of chat:\n{summary}")
        #     continue

        # Retrieve documents using embeddings
        relevant_docs = retriever.invoke(query)

        if relevant_docs:  # If documents are found
            combined_input = (
                f"Here are some documents that might help answer the question: {query}\n\n"
                + "Relevant Documents:\n"
                + "\n\n".join([doc.page_content for doc in relevant_docs])
                + "\n\nPlease provide an answer based on the provided documents. "
                + "If the answer is not found in the documents, provide your answer with a note like: "
                + "'Result not found in provided data but here's my response based on my knowledge: your answer.'"
            )
        else:  # If no documents are found
            combined_input = (
                f"Query: {query}\n\n"
                + "No relevant documents found. "
                + "Please provide a response based on your own knowledge."
            )

        # Process the query through the retrieval chain
        result = rag_chain.invoke({"input": combined_input, "chat_history": chat_history})
        
        # Get the AI's response
        ai_response = result.get("answer", "Sorry, I couldn't process that query.")

        # Display the AI's response
        print(f"AI: {ai_response}")

        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=ai_response))


def summarize_chat(chat_history):
    """
    Summarizes the chat history by extracting and combining key points from user and AI exchanges.
    """
    result = rag_chain.invoke({"input": "summarise our chat", "chat_history": chat_history})
        # Display the AI's response
    print(f"AI: {result['answer']}")

# def summarize_document():
#     combined_input = "summarise below document :" +"\n\n"+str(documents)
#     messages = [
#     SystemMessage(content="You are a helpful summariser."),
#     HumanMessage(content=combined_input),
#     ]
#     result  = llm.invoke(messages)
#     print("AI : ",result)


if __name__ == "__main__":
    continue_chat()