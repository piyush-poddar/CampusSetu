import os
import psycopg2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import json
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini AI Client
client = genai.Client(api_key=f"{os.getenv('GEMINI_API_KEY')}")

def get_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Fetch embedding vector using Google Gemini API.
    """
    try:
        print("Fetching embeddings from Gemini API...")
        embeddings = []

        # Split chunks into batches of 100
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
            )
            embeddings.extend([embedding.values for embedding in result.embeddings])
        print("Embeddings fetched successfully.")
        return embeddings
    
    except Exception as e:
        print(f"❌ Error fetching embedding: {e}")
        return None

def process_pdf(file_name: str) -> list[str]:
    """
    Process PDF file and return text chunks.
    """
    try:
        print("Processing PDF...")
        file_path = os.path.join("documents", file_name)
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        print(f"PDF processed into {len(chunks)} chunks.")
        return [chunk.page_content for chunk in chunks]
    
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return None

def add_document_to_db(
    file_name: str,
    doc_name: str,
    branch: Optional[str] = "all",
    year: Optional[str] = "all",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None
):
    """
    Process PDF, get embeddings, and insert into the database.
    """
    try:
        print("Adding document to DB...")
        chunks = process_pdf(file_name)
        if not chunks:
            print("No chunks to process.")
            return

        embeddings = get_embeddings(chunks)
        if not embeddings:
            print("No embeddings fetched.")
            return
        print(f"Fetched {len(embeddings)} embeddings.")
        from db import insert  # Import here to avoid circular dependency
        for chunk, embedding in zip(chunks, embeddings):
            insert(
                content=chunk,
                embedding=embedding,
                doc_name=doc_name,
                branch=branch,
                year=year,
                valid_from=valid_from,
                valid_to=valid_to
            )
        print("Document added to DB successfully.")
    
    except Exception as e:
        print(f"❌ Error adding document to DB: {e}")

if __name__ == "__main__":
    file_name = "tt.pdf"
    add_document_to_db(
        file_name=file_name,
        doc_name="Time Table Odd 2025",
        branch="all",
        year="all"
    )