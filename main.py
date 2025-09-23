import os
from google import genai
from db import fetch_similar_documents
from ingestion import get_embeddings
from agent import get_gemini_response
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini AI Client
client = genai.Client(api_key=f"{os.getenv('GEMINI_API_KEY')}")

def answer_query(query: str, branch: Optional[str] = "all", year: Optional[str] = "all") -> str:
    """
    Answer user query using context from the database and Google Gemini API.
    """
    try:
        # Step 1: Get embedding for the query
        query_embedding = get_embeddings([query])[0]
        
        # Step 2: Fetch similar documents from the database
        similar_docs = fetch_similar_documents(query_embedding, top_k=7, branch=branch, year=year)
        
        if not similar_docs:
            return "No relevant context found in the database."
        
        # Step 3: Prepare context for Gemini API
        context = [{"content": doc["content"], "doc_name": doc["doc_name"]} for doc in similar_docs]
        
        # Step 4: Get response from Gemini API
        response = get_gemini_response(context, query)
        
        return response
    
    except Exception as e:
        print(f"❌ Error answering query: {e}")
        return "Error processing your query."
    
if __name__ == "__main__":
    user_query = "What courses in fifth semester?"
    branch = "all"
    year = "all"
    
    answer = answer_query(user_query, branch, year)
    print(f"💡 Answer: {answer}")