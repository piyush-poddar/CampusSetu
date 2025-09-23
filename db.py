import psycopg2
from dotenv import load_dotenv
import os
from typing import Optional
import uuid

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

def get_db_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
# Connect to the database
# try:
#     connection = get_db_connection()
        
#     print("Connection successful!")
    
#     # Create a cursor to execute SQL queries
#     cursor = connection.cursor()
    
#     # Example query
#     cursor.execute("SELECT NOW();")
#     result = cursor.fetchone()
#     print("Current Time:", result)

#     # Close the cursor and connection
#     cursor.close()
#     connection.close()
#     print("Connection closed.")

# except Exception as e:
#     print(f"Failed to connect: {e}")

def insert(
    content: str,
    embedding: list[float],
    doc_name: str,
    branch: Optional[str] = "all",
    year: Optional[str] = "all",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None
):
    """
    Insert document metadata into the database.
    """
    try:
        doc_id = str(uuid.uuid4())  # Convert UUID to string
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO documents (content, embedding, doc_name, branch, year, valid_from, valid_to, doc_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (content, embedding, doc_name, branch, year, valid_from, valid_to, doc_id)
        )
        connection.commit()
        print("Document metadata inserted successfully.")

    except Exception as e:
        print(f"❌ Error inserting document metadata: {e}")

    finally:
        cursor.close()

def fetch_similar_documents(
    query_embedding: list[float], 
    top_k: int = 5,
    branch: Optional[str] = "all",
    year: Optional[str] = "all"
) -> list[dict]:
    """
    Fetch top_k similar documents based on the query embedding with similarity scoring.
    """
    if query_embedding is None:
        print("⚠️ Skipping search due to embedding fetch failure.")
        return []

    # Build the base SQL query with similarity scoring
    sql_query = """
        SELECT content, doc_name, branch, year, valid_from, valid_to, 
               1 - (embedding <=> %s::vector) AS similarity 
        FROM documents
    """
    
    # Build WHERE clause conditions
    conditions = []
    params: list = [query_embedding]
    
    if branch and branch != "all":
        conditions.append("(branch = %s OR branch = 'all')")
        params.append(branch)
    
    if year and year != "all":
        conditions.append("(year = %s OR year = 'all')")
        params.append(year)
    
    # Add WHERE clause if there are conditions
    if conditions:
        sql_query += " WHERE " + " AND ".join(conditions)
    
    # Add ORDER BY and LIMIT
    sql_query += " ORDER BY similarity DESC LIMIT %s;"
    params.append(top_k)

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        print(f"found {len(results)} results")
        print([row[0] for row in results])
        documents = [
            {
                "content": row[0],
                "doc_name": row[1],
                "branch": row[2],
                "year": row[3],
                "valid_from": row[4],
                "valid_to": row[5],
                "similarity": float(row[6])
            }
            for row in results
        ]
        
        # Print Results Neatly (optional - uncomment if needed)
        # if documents:
        #     print("\n🔍 **Search Results (Dense Search - PGVector):**\n")
        #     for idx, doc in enumerate(documents, 1):
        #         content_preview = doc["content"][:50] + "..." if len(doc["content"]) > 50 else doc["content"]
        #         print(f"{idx}. 📝 {content_preview}  (Similarity: {doc['similarity']:.4f})")
        # else:
        #     print("⚠️ No relevant results found.")
        
        return documents

    except Exception as e:
        print(f"❌ Database error: {e}")
        return []

    finally:
        cursor.close()