import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import Optional
import time

# Import your existing modules
from ingestion import add_document_to_db, process_pdf, get_embeddings
from main import answer_query
from db import insert

# Page configuration
st.set_page_config(
    page_title="CampusSetu - Document Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'upload_success' not in st.session_state:
    st.session_state.upload_success = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 CampusSetu Document Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["📤 Upload Documents", "💬 Ask Questions", "📊 Query History"]
        )
    
    if page == "📤 Upload Documents":
        upload_page()
    elif page == "💬 Ask Questions":
        query_page()
    elif page == "📊 Query History":
        history_page()

def upload_page():
    st.markdown("## 📤 Upload PDF Documents")
    st.markdown("Upload your academic documents and add them to the knowledge base.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload academic documents, syllabi, timetables, etc."
        )
        
        if uploaded_file is not None:
            st.success(f"📄 File uploaded: {uploaded_file.name}")
            
            # Document metadata form
            with st.form("document_metadata"):
                st.markdown("### Document Metadata")
                
                doc_name = st.text_input(
                    "Document Name",
                    value=uploaded_file.name.replace('.pdf', ''),
                    help="Give a descriptive name for this document"
                )
                
                col1_form, col2_form = st.columns(2)
                
                with col1_form:
                    branch = st.selectbox(
                        "Branch",
                        ["all", "CSE", "ECE", "ME", "CE", "EE", "IT", "BBA", "MBA"],
                        help="Select the relevant branch or 'all' for general documents"
                    )
                    
                    year = st.selectbox(
                        "Year",
                        ["all", "2024", "2025", "2026", "2027"],
                        help="Select the relevant year or 'all' for general documents"
                    )
                
                with col2_form:
                    valid_from = st.date_input(
                        "Valid From",
                        value=None,
                        help="Optional: When this document becomes valid"
                    )
                    
                    valid_to = st.date_input(
                        "Valid To",
                        value=None,
                        help="Optional: When this document expires"
                    )
                
                submitted = st.form_submit_button("🚀 Process and Upload Document", type="primary")
                
                if submitted:
                    if doc_name.strip():
                        process_and_upload_document(
                            uploaded_file, doc_name, branch, year, 
                            valid_from, valid_to
                        )
                    else:
                        st.error("Please provide a document name!")
    
    with col2:
        st.markdown("### 📋 Upload Guidelines")
        st.info("""
        **Supported Files:** PDF only
        
        **Document Types:**
        - Academic syllabi
        - Timetables
        - Course materials
        - Exam schedules
        - Academic policies
        
        **Metadata Tips:**
        - Use descriptive names
        - Set branch to 'all' for general docs
        - Set year to 'all' for evergreen content
        """)

def process_and_upload_document(uploaded_file, doc_name, branch, year, valid_from, valid_to):
    """Process and upload document to database"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create documents directory if it doesn't exist
        if not os.path.exists("documents"):
            os.makedirs("documents")
        
        # Save uploaded file temporarily
        temp_path = os.path.join("documents", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_text.text("📄 Processing PDF...")
        progress_bar.progress(25)
        
        # Process PDF and get chunks
        chunks = process_pdf(uploaded_file.name)
        if not chunks:
            st.error("❌ Failed to process PDF. Please check the file.")
            return
        
        status_text.text("🔤 Generating embeddings...")
        progress_bar.progress(50)
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        if not embeddings:
            st.error("❌ Failed to generate embeddings.")
            return
        
        status_text.text("💾 Saving to database...")
        progress_bar.progress(75)
        
        # Convert dates to strings if provided
        valid_from_str = valid_from.strftime("%Y-%m-%d") if valid_from else None
        valid_to_str = valid_to.strftime("%Y-%m-%d") if valid_to else None
        
        # Insert into database
        for chunk, embedding in zip(chunks, embeddings):
            insert(
                content=chunk,
                embedding=embedding,
                doc_name=doc_name,
                branch=branch,
                year=year,
                valid_from=valid_from_str,
                valid_to=valid_to_str
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Upload completed successfully!")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        st.success(f"""
        🎉 **Document uploaded successfully!**
        
        **Details:**
        - Document: {doc_name}
        - Chunks processed: {len(chunks)}
        - Branch: {branch}
        - Year: {year}
        - Valid from: {valid_from_str or 'Not specified'}
        - Valid to: {valid_to_str or 'Not specified'}
        """)
        
        st.session_state.upload_success = True
        
    except Exception as e:
        st.error(f"❌ Error uploading document: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def query_page():
    st.markdown("## 💬 Ask Questions")
    st.markdown("Ask questions about your uploaded documents and get AI-powered answers.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the T1 examination schedule? When are the CSE courses offered?",
            help="Ask questions about your uploaded documents"
        )
        
        # Filter options
        col1_filter, col2_filter, col3_filter = st.columns(3)
        
        with col1_filter:
            query_branch = st.selectbox(
                "Filter by Branch",
                ["all", "CSE", "ECE", "ME", "CE", "EE", "IT", "BBA", "MBA"],
                key="query_branch"
            )
        
        with col2_filter:
            query_year = st.selectbox(
                "Filter by Year",
                ["all", "2024", "2025", "2026", "2027"],
                key="query_year"
            )
        
        with col3_filter:
            top_k = st.slider(
                "Number of sources",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of similar documents to consider"
            )
        
        if st.button("🔍 Get Answer", type="primary"):
            if query.strip():
                get_answer(query, query_branch, query_year, top_k)
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.markdown("### 💡 Query Tips")
        st.info("""
        **Example Questions:**
        - What is the exam schedule?
        - When are CSE courses offered?
        - What are the prerequisites for ME courses?
        - Show me the timetable for ECE
        
        **Filters:**
        - Use branch filter for specific departments
        - Use year filter for time-specific info
        - Adjust sources for more/less context
        """)

def get_answer(query, branch, year, top_k):
    """Get answer for user query"""
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Get answer from your existing function
            answer = answer_query(query)
            
            if answer and answer != "No relevant context found in the database.":
                st.markdown("### 🎯 Answer:")
                st.markdown(f"""
                <div class="success-box">
                {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.query_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "branch": branch,
                    "year": year,
                    "answer": answer
                })
                
                st.success("✅ Answer generated successfully!")
                
            else:
                st.warning("⚠️ No relevant information found in the database. Try uploading more documents or rephrasing your question.")
                
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

def history_page():
    st.markdown("## 📊 Query History")
    
    if st.session_state.query_history:
        st.markdown(f"**Total queries:** {len(st.session_state.query_history)}")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.query_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Display history in reverse order (newest first)
        for i, entry in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"**Query {len(st.session_state.query_history) - i}:** {entry['query'][:50]}..."):
                st.markdown(f"**🕒 Time:** {entry['timestamp']}")
                st.markdown(f"**🏢 Branch:** {entry['branch']}")
                st.markdown(f"**📅 Year:** {entry['year']}")
                st.markdown(f"**❓ Question:** {entry['query']}")
                st.markdown(f"**💬 Answer:** {entry['answer']}")
    else:
        st.info("📝 No queries yet. Go to the 'Ask Questions' page to start asking!")

if __name__ == "__main__":
    main()
