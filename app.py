import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI
import openai                     # <-- ADD
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
import fitz  # PyMuPDF
import hashlib
import tempfile
from typing import List, Dict
import json
from functools import partial
import socket          # NEW – for DNS check
from openai import APIConnectionError  # for granular catch
from httpx import HTTPStatusError
import ssl, http.client   # NEW – raw HTTPS test
from urllib.parse import urlparse          # already added

from pathlib import Path

# Load environment variables ── and remember which .env was found
ENV_PATH = find_dotenv(filename=".env", usecwd=True)
load_dotenv(ENV_PATH)          # pass empty string if nothing found
print(f"[DEBUG] dotenv loaded from: {ENV_PATH or 'Not found'}")

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
# Keep a *per‑index* record of which files we have already parsed and uploaded,
# so the same file can be re‑processed when the user chooses a new index.
if 'indexed_documents' not in st.session_state:
    # maps index_name ➜ set{file_name, …}
    st.session_state.indexed_documents = {}
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None
if 'available_indexes' not in st.session_state:
    st.session_state.available_indexes = []

# Azure configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "rag-demo-index")

# Page configuration
st.set_page_config(
    page_title="RAG Demo - Azure o3",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📚 RAG Application with Azure o3")
st.markdown("העלה קבצי PDF ושאל שאלות בעברית")

# Debug info (remove in production)
with st.expander("🔧 Debug Information", expanded=False):
    st.write("Azure Search Endpoint:", AZURE_SEARCH_ENDPOINT)
    st.write("Azure Search Key configured:", "Yes" if AZURE_SEARCH_KEY else "No")
    st.write("Current Index:", st.session_state.selected_index or "None selected")
    st.write("Loaded .env file:", ENV_PATH or "Not found")
    st.write("OpenAI Endpoint (runtime):", os.getenv("AZURE_OPENAI_ENDPOINT"))
    st.write("OpenAI Deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

# Initialize Azure OpenAI client
# --------------------------- patch init_openai_client --------------------------
@st.cache_resource
def init_openai_client():
    """Return AzureOpenAI client (API-key or AAD)."""
    # --- read env values ----------------------------------------------------
    raw_ep      = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key     = os.getenv("AZURE_OPENAI_KEY", "").strip()
    # ------------------------------------------------------------------------

    # basic validation
    if not (raw_ep.startswith("https://") and deployment):
        return None

    base_ep = raw_ep.split("/openai")[0]
    host    = urlparse(base_ep).hostname or base_ep.replace("https://", "")

    # best-effort TLS probe (fail-soft)
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
            s.settimeout(2)
            s.connect((host, 443))
    except Exception:
        pass  # probe failure is not fatal

    # choose auth mode
    kwargs = dict(azure_endpoint=base_ep, api_version=api_version)
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

    client = AzureOpenAI(**kwargs)
    return client
# ------------------------------------------------------------------------------

# Initialize Azure Search client
@st.cache_resource
def init_search_client(index_name=None):
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        st.warning("Azure Search credentials not configured. Please add AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY to .env file")
        return None, None
    
    # ---- NEW: explicit DNS resolution test ---------------------------------
    try:
        host = AZURE_SEARCH_ENDPOINT.replace("https://", "").split("/")[0]
        socket.gethostbyname(host)
    except Exception as dns_err:
        st.error(
            f"❌ Cannot resolve host “{host}”.\n"
            "• Verify the service name is spelled exactly as shown in Azure Portal\n"
            "• The endpoint must be reachable from your network\n"
            f"DNS error: {dns_err}"
        )
        return None, None
    # ------------------------------------------------------------------------

    try:
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        
        # Test connection first
        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=credential
        )
        
        # Try to list indexes to test connection
        try:
            available_indexes = list(index_client.list_indexes())
            st.session_state.available_indexes = [idx.name for idx in available_indexes]
        except Exception as conn_error:
            st.error(f"Failed to connect to Azure Search. Please verify:")
            st.error("1. The service name in the endpoint URL is correct")
            st.error("2. The Azure Search service exists in your Azure subscription")
            st.error("3. The API key is valid")
            st.error(f"Connection error: {str(conn_error)}")
            return None, None
        
        # Use the provided index name or the selected one
        current_index = index_name or st.session_state.selected_index
        
        if current_index:
            search_client = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=current_index,
                credential=credential
            )
        else:
            search_client = None
        
        return search_client, index_client
    except Exception as e:
        st.error(f"Error initializing Search client: {str(e)}")
        return None, None

def create_search_index(index_client):
    try:
        # Check if index exists
        existing_indexes = [index.name for index in index_client.list_indexes()]
        if AZURE_SEARCH_INDEX in existing_indexes:
            st.info(f"Using existing search index: {AZURE_SEARCH_INDEX}")
            return
        
        # Create index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True)
        ]
        
        index = SearchIndex(
            name=AZURE_SEARCH_INDEX,
            fields=fields
        )
        
        index_client.create_index(index)
        st.success(f"Created search index: {AZURE_SEARCH_INDEX}")
    except Exception as e:
        st.error(f"Error creating search index: {str(e)}")

def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """Extract text from PDF file page by page"""
    documents = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        pdf_document = fitz.open(tmp_file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            if text.strip():
                doc_id = hashlib.md5(f"{pdf_file.name}_{page_num}".encode()).hexdigest()
                documents.append({
                    "id": doc_id,
                    "content": text,
                    "title": pdf_file.name,
                    "page": page_num + 1,
                    "source": pdf_file.name
                })
        
        pdf_document.close()
    finally:
        os.unlink(tmp_file_path)
    
    return documents

def index_documents(search_client, documents):
    """Index documents in Azure Search"""
    if not search_client:
        st.error("Search client not initialized")
        return
    
    try:
        result = search_client.upload_documents(documents)
        st.success(f"Successfully indexed {len(documents)} pages")
    except Exception as e:
        st.error(f"Error indexing documents: {str(e)}")

# --------------------------------- NEW helpers ---------------------------------
def _extract_doc_fields(hit: dict) -> dict:
    """
    Figure-out which keys in an Azure Search hit correspond to:
    text-content, title and path.  Returns a mapping like
    {'text': 'content_text', 'title': 'document_title', 'path': 'content_path'}
    """
    candidates = {
        'text':  ["content", "content_text", "text", "chunk", "page_content"],
        'title': ["title", "document_title", "file_name"],
        'path':  ["source", "content_path", "path"]
    }
    mapping = {}
    for logical, names in candidates.items():
        mapping[logical] = next((n for n in names if n in hit), None)
    return mapping
# ------------------------------------------------------------------------------

def search_documents(search_client, query: str, top: int = 25) -> List[Dict]:
    """Search documents using Azure Search"""
    if not search_client:
        return []
    try:
        results = list(
            search_client.search(search_text=query, top=top, include_total_count=True)
        )
        if not results:
            return []

        # Detect field names from the first hit
        field_map = _extract_doc_fields(results[0])

        documents = []
        for r in results:
            documents.append({
                "content": r.get(field_map['text'], ""),     # fallbacks safe
                "title":   r.get(field_map['title'], ""),
                "page":    r.get("page", None),
                "source":  r.get(field_map['path'], r.get("source", ""))
            })
        return documents
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def generate_answer(openai_client, query: str, context: str) -> str:
    """Generate answer and expose the exact REST URL used by the SDK."""
    if not openai_client:
        return "OpenAI client not initialized"

    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "N/A").rstrip("/")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "N/A")
    api_ver    = os.getenv("AZURE_OPENAI_API_VERSION", "N/A")

    # derive the exact URL the SDK will hit
    base = endpoint if endpoint.endswith("/openai") else f"{endpoint}/openai"
    api_url = f"{base}/deployments/{deployment}/chat/completions?api-version={api_ver}"

    try:
        response = openai_client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "אתה עוזר AI שעונה על שאלות בעברית בהתבסס על המסמכים שניתנו לך.\nענה רק על סמך המידע שבמסמכים. אם אין מספיק מידע לענות על השאלה, אמור זאת בבירור.\nענה בעברית תמיד."},
                {"role": "user",   "content": f"בהתבסס על המסמכים הבאים:\n\n{context}\n\nענה על השאלה הבאה: {query}"}
            ],
            max_completion_tokens=20000
        )
        return response.choices[0].message.content

    except APIConnectionError as ce:
        # Surface the root cause (DNS fail, timeout, proxy refusal, etc.)
        root_cause = repr(getattr(ce, "__cause__", ce))
        st.error(
            f"Azure OpenAI network error:\n"
            f"• REST URL: {api_url}\n"
            f"• Exception: {root_cause}"
        )
        return f"שגיאת רשת מול Azure OpenAI: {root_cause}"

    except openai.OpenAIError as e:
        code    = getattr(e, "status_code", "N/A")
        message = getattr(e, "message", str(e))
        st.error(
            f"Azure OpenAI API error ({code}): {message}\n"
            f"• REST URL: {api_url}"
        )
        return f"שגיאה מ-Azure OpenAI ({code}): {message}"

    except Exception as e:
        st.error(
            f"Unexpected error contacting Azure OpenAI:\n"
            f"• REST URL: {api_url}\n"
            f"• Exception: {e}"
        )
        return f"שגיאה ביצירת תשובה: {e}"

# ---------- helper to rebuild the cached OpenAI client ----------
def refresh_openai_client():
    init_openai_client.clear()
    return init_openai_client()
# ----------------------------------------------------------------


# --------------------  NEW: helpers to refresh cached clients  --------------------
def refresh_search_client():
    """Clear Streamlit cache for init_search_client so we can reconnect."""
    init_search_client.clear()
    return init_search_client()          # returns (search_client, index_client)
# -------------------------------------------------------------------------------

# Sidebar for configuration, index selection and file-upload
with st.sidebar:
    st.header("⚙️  Azure Search Settings")

    # --- NEW: editable endpoint & key ---
    st.text_input("Endpoint",
                  value=AZURE_SEARCH_ENDPOINT,
                  key="custom_endpoint",
                  help="Full URL, e.g. https://<service>.search.windows.net")
    st.text_input("Admin Key",
                  value=AZURE_SEARCH_KEY,
                  key="custom_key",
                  type="password")

    # On-click: update env vars → reconnect
    if st.button("🔄 Connect / Reconnect"):
        os.environ["AZURE_SEARCH_ENDPOINT"] = st.session_state.custom_endpoint.strip()
        os.environ["AZURE_SEARCH_KEY"]      = st.session_state.custom_key.strip()

        # Update module-level vars so rest of code sees the new values
        AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")

        # Re-create clients
        search_client, index_client = refresh_search_client()
        if search_client:
            st.success("Connected to Azure Search service")
        else:
            st.error("Failed to connect. Check endpoint / key and try again.")

    st.divider()
    
    # Ensure we have an index_client object for the UI below
    _, sidebar_index_client = init_search_client()   # <- always returns a value (may be None)

    st.divider()
    
    # Index selection
    if st.session_state.available_indexes:
        st.subheader("📋 Select Search Index")
        selected_index = st.selectbox(
            "Choose an existing index:",
            options=[""] + st.session_state.available_indexes,
            index=0 if not st.session_state.selected_index else st.session_state.available_indexes.index(st.session_state.selected_index) + 1 if st.session_state.selected_index in st.session_state.available_indexes else 0,
            help="Select an index to search in existing documents"
        )
        
        if selected_index:
            st.session_state.selected_index = selected_index
            st.success(f"Using index: {selected_index}")
            
            # Show index info
            if sidebar_index_client:
                try:
                    index = sidebar_index_client.get_index(selected_index)
                    with st.expander("Index Information", expanded=False):
                        st.write(f"Fields: {', '.join([field.name for field in index.fields])}")
                except Exception as e:
                    st.error(f"Error getting index info: {str(e)}")
    
    st.divider()
    
    st.header("📁 Upload New Files")
    st.info("You can also upload new PDF files to create a new index or add to existing one")
    
    uploaded_files = st.file_uploader(
        "בחר קבצי PDF",
        type=['pdf'],
        accept_multiple_files=True,
        help="ניתן להעלות מספר קבצי PDF בו זמנית"
    )
    
    if uploaded_files:
        openai_client = init_openai_client()
        
        # Option to create new index or use selected one
        if st.session_state.selected_index:
            use_existing = st.checkbox(f"Add to existing index: {st.session_state.selected_index}", value=True)
            if not use_existing:
                new_index_name = st.text_input("New index name:", value="rag-demo-index-new")
                st.session_state.selected_index = new_index_name
        else:
            new_index_name = st.text_input("New index name:", value=AZURE_SEARCH_INDEX)
            st.session_state.selected_index = new_index_name
        
        search_client, index_client = init_search_client(st.session_state.selected_index)
        
        if st.button("🔄 עבד וצור אינדקס", type="primary"):
            with st.spinner("מעבד קבצים..."):
                all_documents = []

                # Ensure we have a bucket for the current index
                idx = st.session_state.selected_index
                if idx not in st.session_state.indexed_documents:
                    st.session_state.indexed_documents[idx] = set()

                for uploaded_file in uploaded_files:
                    # Re‑process the PDF if it has **not** yet been sent to *this* index
                    if uploaded_file.name not in st.session_state.indexed_documents[idx]:
                        documents = extract_text_from_pdf(uploaded_file)
                        all_documents.extend(documents)
                        st.session_state.indexed_documents[idx].add(uploaded_file.name)

                if all_documents and search_client:
                    # Create index if needed
                    if st.session_state.selected_index not in st.session_state.available_indexes:
                        create_search_index(index_client)
                    index_documents(search_client, all_documents)
                    # Flatten all filenames from every index for display purposes
                    flat_files = {f for files in st.session_state.indexed_documents.values() for f in files}
                    st.session_state.uploaded_files = sorted(flat_files)
    
    # Display indexed files
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Recently Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.text(f"• {file}")

# Main chat interface
st.header("💬 שאל שאלה")

# Initialize clients
openai_client = init_openai_client()
search_client, index_client = init_search_client(st.session_state.selected_index)

# Show current index status
if st.session_state.selected_index:
    st.info(f"🔍 Searching in index: **{st.session_state.selected_index}**")
else:
    st.warning("⚠️ Please select an index from the sidebar to search")

# Chat input
user_question = st.text_input(
    "הקלד את שאלתך כאן:",
    placeholder="למשל: מה אומר המסמך על...",
    key="user_input"
)

if user_question and st.button("🔍 חפש תשובה", type="primary"):
    if not st.session_state.selected_index:
        st.error("Please select an index first")
    elif not search_client:
        st.error("Azure Search לא מוגדר כראוי")
    else:
        with st.spinner("מחפש במסמכים..."):
            # Search for relevant documents
            search_results = search_documents(search_client, user_question, top=25)
            # Fallback – try a semantically‑close variant of the question
            # if not search_results:
            #     alt_query = (user_question
            #                  .replace("חברי", "הרכב")
            #                  .replace("התמיכות", "הוועדה"))
            #     if alt_query != user_question:
            #         search_results = search_documents(search_client,
            #                                           alt_query,
            #                                           top=25)
            
            if search_results:
                # Prepare context from search results
                context = "\n\n".join([
                    f"מקור: {doc['source']} "
                    f"{'(עמוד ' + str(doc['page']) + ')' if doc['page'] else ''}\n"
                    f"{doc['content'][:2000]}..."
                    for doc in search_results
                ])

                # 🔎 Optional debug panels to inspect what the RAG pipeline is sending
                with st.expander("🔎 Debug: RAG context sent to the LLM", expanded=False):
                    st.code(context, language="markdown")

                with st.expander("🔎 Debug: raw search hits", expanded=False):
                    st.json(search_results)
                
                # Generate answer
                with st.spinner("מייצר תשובה..."):
                    answer = generate_answer(openai_client, user_question, context)
                
                # Display results
                st.divider()
                
                # Answer section
                st.subheader("📝 תשובה")
                if answer and str(answer).strip():
                    # show as rich markdown
                    st.markdown(str(answer))
                else:
                    st.warning("⚠️ המודל החזיר תשובה ריקה או שלא חזרה תשובה.")
                    with st.expander("Debug: raw answer", expanded=False):
                        st.text(repr(answer))
                
                # Sources section
                with st.expander("📚 מקורות רלוונטיים", expanded=False):
                    for i, doc in enumerate(search_results):
                        st.markdown(f"**{i+1}. {doc['source']} - עמוד {doc['page']}**")
                        st.text(doc['content'][:300] + "...")
                        st.divider()
            else:
                st.info("לא נמצאו מסמכים רלוונטיים לשאלתך")

# Footer
st.divider()
st.caption("RAG Demo with Azure o3 - Built with Streamlit")
