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
import ssl, http.client   # NEW – for raw HTTPS test
import httpx              # ← ADD
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
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "o3"  # default to o3

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
st.markdown("Upload PDF files and ask questions")

# Debug info (remove in production)
with st.expander("🔧 Debug Information", expanded=False):
    st.write("Azure Search Endpoint:", AZURE_SEARCH_ENDPOINT)
    st.write("Azure Search Key configured:", "Yes" if AZURE_SEARCH_KEY else "No")
    st.write("Current Index:", st.session_state.selected_index or "None selected")
    st.write("Loaded .env file:", ENV_PATH or "Not found")
    st.write("OpenAI Endpoint (runtime):",
             st.session_state.get(f"dbg_endpoint_{st.session_state.selected_model}",
                                  os.getenv("AZURE_OPENAI_ENDPOINT")))
    st.write("OpenAI Deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

# Initialize Azure OpenAI client
# --------------------------- patch init_openai_client --------------------------
def _check_tls(host: str) -> tuple[bool, str]:
    """Return (is_open, error_text) for port 443 TLS handshake."""
    try:
        with httpx.Client(http2=False, verify=True, timeout=2) as c:
            c.get(f"https://{host}", headers={"User-Agent": "probe"})
        return True, ""
    except Exception as ex:
        return False, str(ex)

@st.cache_resource
def init_openai_client(model_type: str = "o3"):
    """Return AzureOpenAI client for the selected model (o3 / 4o)."""
    suffix = "_4o" if model_type == "4o" else ("_41" if model_type == "41" else "")
    ep_var = f"AZURE_OPENAI_ENDPOINT{suffix}"

    raw_ep = os.getenv(ep_var, "").strip()

    # --- NEW auto-fix for “https:/” typo ---------------------------------
    if raw_ep.startswith("https:/") and not raw_ep.startswith("https://"):
        raw_ep = "https://" + raw_ep[len("https:/"):].lstrip("/")
    raw_ep = raw_ep.rstrip("/")
    # ---------------------------------------------------------------------

    deployment  = os.getenv(f"AZURE_OPENAI_DEPLOYMENT{suffix}", "").strip()
    api_version = os.getenv(f"AZURE_OPENAI_API_VERSION{suffix}", "2025-04-01-preview")
    api_key     = os.getenv(f"AZURE_OPENAI_KEY{suffix}", "").strip()

    # -----------------------------------------------------------------------
    # Enhanced validation & feedback
    incomplete = {"https://", "http://", "https:/", "http:/", "https:", "http:"}
    if raw_ep in incomplete:
        st.error(
            f"{ep_var} looks incomplete (**{raw_ep}**). "
            "Paste the full resource URL, e.g. "
            "`https://myresource.openai.azure.com`"
        )
        return None
    if not raw_ep:
        st.error(f"{ep_var} is empty.  Please set it in .env or the sidebar.")
        return None
    if not raw_ep.startswith("https://"):
        st.error(f"{ep_var} must start with https://  (got: {raw_ep})")
        return None
    if not deployment:
        st.error(f"AZURE_OPENAI_DEPLOYMENT{suffix} is empty.")
        return None
    # -----------------------------------------------------------------------

    # Extract the base endpoint without accidentally truncating the domain
    parsed = urlparse(raw_ep)
    if parsed.path.lower().startswith("/openai"):
        # Endpoint was provided with the "/openai" suffix – strip it safely
        base_ep = f"{parsed.scheme}://{parsed.netloc}"
    else:
        # Endpoint came without the suffix; just make sure there is no trailing slash
        base_ep = raw_ep.rstrip("/")

    host = urlparse(base_ep).hostname or ""
    if host in {"https:", "http:"} or not host:
        st.error(
            f"{ep_var} host part looks invalid (**{raw_ep}**). "
            "Expected format: https://<resource>.openai.azure.com"
        )
        return None
    # -----------------------------------------------------

    ok, err = _check_tls(host)
    if not ok:
        st.error(f"❌ Cannot reach {host}: {err}")
        return None

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
    """Generate answer with model-specific parameters."""
    if not openai_client:
        return "OpenAI client not initialized"

    model_type = st.session_state.selected_model
    suffix = "_4o" if model_type == "4o" else ("_41" if model_type == "41" else "")
    
    endpoint   = os.getenv(f"AZURE_OPENAI_ENDPOINT{suffix}", "N/A").rstrip("/")
    deployment = os.getenv(f"AZURE_OPENAI_DEPLOYMENT{suffix}", "N/A")
    api_ver    = os.getenv(f"AZURE_OPENAI_API_VERSION{suffix}", "N/A")

    # derive the exact URL the SDK will hit
    base = endpoint if endpoint.endswith("/openai") else f"{endpoint}/openai"
    api_url = f"{base}/deployments/{deployment}/chat/completions?api-version={api_ver}"

    try:
        # Base parameters
        params = {
            "model": deployment,
            "messages": [
                {"role": "system", "content": "אתה עוזר AI שעונה על שאלות בעברית בהתבסס על המסמכים שניתנו לך.\nענה רק על סמך המידע שבמסמכים. אם אין מספיק מידע לענות על השאלה, אמור זאת בבירור.\nענה בעברית תמיד."},
                {"role": "user", "content": f"בהתבסס על המסמכים הבאים:\n\n{context}\n\nענה על השאלה הבאה: {query}"}
            ]
        }
        
        # Model-specific parameters
        if model_type == "o3":
            params["max_completion_tokens"] = 20000
        else:  # 4o
            params["temperature"] = 0.7
            params["max_tokens"] = 4000
        
        response = openai_client.chat.completions.create(**params)
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
def refresh_openai_client(model_type="o3"):
    # Clear all cached versions
    init_openai_client.clear()
    return init_openai_client(model_type)
# ----------------------------------------------------------------


# --------------------  NEW: helpers to refresh cached clients  --------------------
def refresh_search_client():
    """Clear Streamlit cache for init_search_client so we can reconnect."""
    init_search_client.clear()
    return init_search_client()          # returns (search_client, index_client)
# -------------------------------------------------------------------------------

# Sidebar for configuration, index selection and file-upload
with st.sidebar:
    # Model selector - place at the top
    st.header("🤖 Model Selection")
    # Allow picking o3, 4o or the new 41 deployment
    MODEL_CHOICES = ["o3", "4o", "41"]
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(st.session_state.selected_model)
               if st.session_state.selected_model in MODEL_CHOICES else 0,
        help="o3 – reasoning model | 4o – GPT‑4o | 41 – GPT‑4.1 preview"
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Refresh client when model changes
        refresh_openai_client(selected_model)
        st.success(f"Switched to {selected_model} model")
    
    # Show current OpenAI configuration
    suffix = "_4o" if selected_model == "4o" else ("_41" if selected_model == "41" else "")
    with st.expander(f"🔧 {selected_model} OpenAI Settings", expanded=False):
        st.text_input(
            "Endpoint",
            value=os.getenv(f"AZURE_OPENAI_ENDPOINT{suffix}", ""),
            key=f"openai_endpoint_{selected_model}",
            help="e.g. https://myresource.openai.azure.com"
        )
        st.text_input(
            "Deployment",
            value=os.getenv(f"AZURE_OPENAI_DEPLOYMENT{suffix}", ""),
            key=f"openai_deployment_{selected_model}"
        )
        st.text_input(
            "API Key",
            value=os.getenv(f"AZURE_OPENAI_KEY{suffix}", ""),
            key=f"openai_key_{selected_model}",
            type="password"
        )
        st.text_input(
            "API Version",
            value=os.getenv(f"AZURE_OPENAI_API_VERSION{suffix}", ""),
            key=f"openai_version_{selected_model}",
            help="e.g. 2025-01-01-preview"
        )
        
        if st.button(f"Update {selected_model} Settings"):
            # Update environment variables
            os.environ[f"AZURE_OPENAI_ENDPOINT{suffix}"] = st.session_state[f"openai_endpoint_{selected_model}"]
            os.environ[f"AZURE_OPENAI_DEPLOYMENT{suffix}"] = st.session_state[f"openai_deployment_{selected_model}"]
            os.environ[f"AZURE_OPENAI_KEY{suffix}"] = st.session_state[f"openai_key_{selected_model}"]
            os.environ[f"AZURE_OPENAI_API_VERSION{suffix}"] = st.session_state[f"openai_version_{selected_model}"]
            
            # Refresh the client
            openai_client = refresh_openai_client(selected_model)
            if openai_client:
                st.success(f"Updated {selected_model} settings successfully")
            else:
                st.error(f"Failed to create {selected_model} client with new settings")
    
    st.divider()
    
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

            # ------- FIXED: index-info block ----------
            if sidebar_index_client:
                try:
                    index = sidebar_index_client.get_index(selected_index)
                    with st.expander("Index Information", expanded=False):
                        st.write(", ".join(f"{f.name} ({f.type})" for f in index.fields))
                except Exception as e:
                    st.error(f"Error getting index info: {e}")
            # ------------------------------------------

    st.divider()

    # ---------------- FIXED: upload-files block ----------------
    st.header("📁 Upload New Files")
    st.info("Upload one or more PDF files to add pages to the current index "
            "or create a new one.")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )

    if uploaded_files:
        # target index
        if not st.session_state.selected_index:
            st.error("Choose or create an index first (see dropdown above).")
        else:
            search_client, index_client = init_search_client(
                st.session_state.selected_index
            )
            if st.button("🔄 Process & Index", type="primary"):
                with st.spinner("Extracting text and uploading…"):
                    new_docs = []
                    idx = st.session_state.selected_index
                    st.session_state.indexed_documents.setdefault(idx, set())

                    for f in uploaded_files:
                        if f.name in st.session_state.indexed_documents[idx]:
                            continue
                        new_docs.extend(extract_text_from_pdf(f))
                        st.session_state.indexed_documents[idx].add(f.name)

                    if new_docs:
                        if idx not in st.session_state.available_indexes:
                            create_search_index(index_client)
                        index_documents(search_client, new_docs)
                        st.session_state.uploaded_files = sorted(
                            {fn for s in st.session_state.indexed_documents.values()
                                   for fn in s}
                        )
    # -----------------------------------------------------------

# ----------------------- Main chat interface -----------------------
st.header("💬 Ask a Question")

# Show current model in use
st.info(f"🤖 Using model: **{st.session_state.selected_model}**")

# Initialize (or refresh) clients
openai_client = init_openai_client(st.session_state.selected_model)
search_client, index_client = init_search_client(st.session_state.selected_index)

# Index status message
if st.session_state.selected_index:
    st.info(f"🔍 Searching in index: **{st.session_state.selected_index}**")
else:
    st.warning("⚠️ Please select an index from the sidebar to search")

# User query
user_question = st.text_input(
    "Type your question here:",
    placeholder="e.g.: What does the document say about…",
    key="user_input"
)

if user_question and st.button("🔍 Search Answer", type="primary"):
    if not st.session_state.selected_index:
        st.error("Please select an index first")
    elif not search_client:
        st.error("Azure Search is not properly configured")
    else:
        with st.spinner("מחפש במסמכים..."):
            search_results = search_documents(search_client, user_question, top=25)

        if search_results:
            # ---------- fixed block ----------
            def _fmt(doc):
                page = f"(page {doc['page']})" if doc.get("page") else ""
                return (f"Source: {doc['source']} {page}\n"
                        f"{doc['content'][:2000]}...")
            context = "\n\n".join(_fmt(d) for d in search_results)
            # ----------------------------------

            with st.expander("🔎 Debug: RAG context sent to the LLM", expanded=False):
                st.code(context, language="markdown")
            with st.expander("🔎 Debug: raw search hits", expanded=False):
                st.json(search_results)

            with st.spinner("מייצר תשובה..."):
                answer = generate_answer(openai_client, user_question, context)

            st.divider()
            st.subheader("📝 Answer")
            if answer and answer.strip():
                st.markdown(answer)
            else:
                st.warning("⚠️ המודל החזיר תשובה ריקה או שלא חזרה תשובה.")
                with st.expander("Debug: raw answer", expanded=False):
                    st.text(repr(answer))

            with st.expander("📚 Relevant Sources", expanded=False):
                for i, d in enumerate(search_results, 1):
                    st.markdown(f"**{i}. {d['source']} - page {d['page']}**")
                    st.text(d['content'][:300] + "…")
                    st.divider()
        else:
            st.info("No relevant documents were found for your question")

# ----------------------------- Footer ------------------------------
st.divider()
st.caption("RAG Demo with Azure o3 - Built with Streamlit")
