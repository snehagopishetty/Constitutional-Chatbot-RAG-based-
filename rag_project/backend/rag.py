import os
from dotenv import load_dotenv

from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

def load_qa_chain(use_memory=True):
    llm = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"), model="command-r")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(
        r"c:\Users\g.sneha2\Desktop\rag_project\backend\constitution_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if use_memory:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # <-- Explicitly tell memory which output key to store
        )
    else:
        memory = None

    conversational_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

    def is_relevant_question(question):
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return False
        content = " ".join([doc.page_content for doc in docs])
        return len(content.strip()) > 50

    return conversational_qa_chain, is_relevant_question


if __name__ == "__main__":
    chain, is_relevant_question = load_qa_chain()

    while True:
        query = input("You: ")
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not is_relevant_question(query):
            print("Sorry, I can't answer that question.\n")
            continue

        result = chain({"question": query})
        print("Chatbot:", result["answer"], "\n")
