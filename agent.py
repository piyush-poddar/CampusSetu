import os
from google import genai
from ingestion import get_embeddings
from db import fetch_similar_documents
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini AI Client
client = genai.Client(api_key=f"{os.getenv('GEMINI_API_KEY')}")

def get_gemini_response(context: list[dict], query: str) -> str:
    """
    Fetch response from Google Gemini API.
    """
    prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.
        Simplify the terms and answer in easy-to-understand language.
        If the context does not contain the answer, say "I don't know".
        Context: {context}
        Question: {query}
        Answer:
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    
    except Exception as e:
        print(f"❌ Error fetching response: {e}")
        return "Error fetching response from Gemini API."