from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import shutil
from datetime import datetime

# Import your existing modules
from ingestion import process_pdf, get_embeddings
from main import answer_query
from db import insert, fetch_similar_documents

# Initialize FastAPI app
app = FastAPI(
    title="CampusSetu API",
    description="AI-powered document assistant for academic institutions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    branch: Optional[str] = "all"
    year: Optional[str] = "all"
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    query: str
    branch: str
    year: str
    timestamp: str
    context_used: int

class DocumentUploadResponse(BaseModel):
    message: str
    doc_name: str
    chunks_processed: int
    branch: str
    year: str
    doc_id: str

class SimilarDocument(BaseModel):
    content: str
    doc_name: str
    branch: str
    year: str
    similarity: float
    valid_from: Optional[str]
    valid_to: Optional[str]

class SearchResponse(BaseModel):
    documents: List[SimilarDocument]
    total_found: int
    query_embedding_success: bool

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CampusSetu API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        from db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "api_status": "healthy",
        "database_status": db_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_name: str = Form(...),
    branch: str = Form(default="all"),
    year: str = Form(default="all"),
    valid_from: Optional[str] = Form(default=""),
    valid_to: Optional[str] = Form(default="")
):
    """
    Upload a PDF document and process it into the knowledge base
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create documents directory if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Save uploaded file temporarily
        temp_file_path = os.path.join("documents", file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF and get chunks
        chunks = process_pdf(file.filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to process PDF. Please check the file.")
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        # Process and validate dates
        def process_date(date_str):
            if not date_str or not date_str.strip() or date_str.lower() in ["", "null", "none", "string"]:
                return None
            # Basic date validation - you can make this more sophisticated
            try:
                # Try to parse the date to validate format
                from datetime import datetime
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                return None
        
        processed_valid_from = process_date(valid_from)
        processed_valid_to = process_date(valid_to)
        
        # Insert into database
        doc_id = None
        for chunk, embedding in zip(chunks, embeddings):
            result = insert(
                content=chunk,
                embedding=embedding,
                doc_name=doc_name,
                branch=branch,
                year=year,
                valid_from=processed_valid_from,
                valid_to=processed_valid_to
            )
            if not doc_id:
                doc_id = "generated"  # You might want to return the actual doc_id from insert function
        
        # Clean up temporary file (optional)
        # os.remove(temp_file_path)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            doc_name=doc_name,
            chunks_processed=len(chunks),
            branch=branch,
            year=year,
            doc_id=doc_id or "generated"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base and get AI-powered answers
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get answer using your existing function
        answer = answer_query(request.query, request.branch, request.year)
        
        if not answer:
            raise HTTPException(status_code=404, detail="No relevant context found in the database")
        
        # Get embedding for context count (optional)
        from ingestion import get_embeddings
        query_embedding = get_embeddings([request.query])[0] if get_embeddings([request.query]) else None
        context_count = 0
        
        if query_embedding:
            similar_docs = fetch_similar_documents(
                query_embedding, 
                top_k=request.top_k or 5, 
                branch=request.branch or "all", 
                year=request.year or "all"
            )
            context_count = len(similar_docs)
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            branch=request.branch or "all",
            year=request.year or "all",
            timestamp=datetime.now().isoformat(),
            context_used=context_count
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search-similar/", response_model=SearchResponse)
async def search_similar_documents(request: QueryRequest):
    """
    Search for similar documents without generating an AI answer
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get embedding for the query
        from ingestion import get_embeddings
        query_embeddings = get_embeddings([request.query])
        
        if not query_embeddings or not query_embeddings[0]:
            return SearchResponse(
                documents=[],
                total_found=0,
                query_embedding_success=False
            )
        
        query_embedding = query_embeddings[0]
        
        # Fetch similar documents
        similar_docs = fetch_similar_documents(
            query_embedding,
            top_k=request.top_k or 5,
            branch=request.branch or "all",
            year=request.year or "all"
        )
        
        # Convert to response format
        documents = [
            SimilarDocument(
                content=doc["content"],
                doc_name=doc["doc_name"],
                branch=doc["branch"],
                year=doc["year"],
                similarity=doc["similarity"],
                valid_from=doc.get("valid_from"),
                valid_to=doc.get("valid_to")
            )
            for doc in similar_docs
        ]
        
        return SearchResponse(
            documents=documents,
            total_found=len(documents),
            query_embedding_success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/documents/stats/")
async def get_document_stats():
    """
    Get statistics about uploaded documents
    """
    try:
        from db import get_db_connection
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Get total documents count
        cursor.execute("SELECT COUNT(DISTINCT doc_name) FROM documents")
        result = cursor.fetchone()
        total_docs = result[0] if result else 0
        
        # Get total chunks count
        cursor.execute("SELECT COUNT(*) FROM documents")
        result = cursor.fetchone()
        total_chunks = result[0] if result else 0
        
        # Get documents by branch
        cursor.execute("""
            SELECT branch, COUNT(DISTINCT doc_name) as doc_count 
            FROM documents 
            GROUP BY branch 
            ORDER BY doc_count DESC
        """)
        branch_stats = cursor.fetchall()
        
        # Get documents by year
        cursor.execute("""
            SELECT year, COUNT(DISTINCT doc_name) as doc_count 
            FROM documents 
            GROUP BY year 
            ORDER BY doc_count DESC
        """)
        year_stats = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "documents_by_branch": {branch: count for branch, count in branch_stats},
            "documents_by_year": {year: count for year, count in year_stats},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.get("/documents/list/")
async def list_documents(
    branch: Optional[str] = None,
    year: Optional[str] = None,
    limit: Optional[int] = 50
):
    """
    List uploaded documents with optional filtering
    """
    try:
        from db import get_db_connection
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Build query with optional filters
        query = """
            SELECT DISTINCT doc_name, branch, year, valid_from, valid_to, 
                   COUNT(*) as chunk_count,
                   MIN(CASE WHEN valid_from IS NOT NULL THEN valid_from END) as earliest_valid,
                   MAX(CASE WHEN valid_to IS NOT NULL THEN valid_to END) as latest_valid
            FROM documents
        """
        
        conditions = []
        params = []
        
        if branch and branch != "all":
            conditions.append("branch = %s")
            params.append(branch)
        
        if year and year != "all":
            conditions.append("year = %s")
            params.append(year)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY doc_name, branch, year, valid_from, valid_to"
        query += " ORDER BY doc_name"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        documents = [
            {
                "doc_name": row[0],
                "branch": row[1],
                "year": row[2],
                "valid_from": row[3],
                "valid_to": row[4],
                "chunk_count": row[5]
            }
            for row in results
        ]
        
        cursor.close()
        connection.close()
        
        return {
            "documents": documents,
            "total_found": len(documents),
            "filters_applied": {
                "branch": branch,
                "year": year,
                "limit": limit
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
