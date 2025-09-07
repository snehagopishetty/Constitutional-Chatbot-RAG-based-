from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from load_document import load_constitution
import os
from dotenv import load_dotenv

load_dotenv()


def build_index():
    try:
        docs = load_constitution(r"c:\Users\g.sneha2\Desktop\rag_project\data\constitution.pdf")
        print(f"Loaded {len(docs)} documents")

        # Load local embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(docs, embeddings)
        db.save_local("constitution_index")
        print("FAISS index created and saved locally at 'constitution_index'")
    except Exception as e:
        print("Error building index:", e)


if __name__ == "__main__":
    build_index()


