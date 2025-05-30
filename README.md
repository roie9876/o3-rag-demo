# RAG Demo with Azure o3

A Retrieval-Augmented Generation (RAG) application using Azure OpenAI o3 model and Azure AI Search.

## Features

- Upload multiple PDF files
- Ask questions in Hebrew
- Works with Azure OpenAI deployments **o3**, **4o**, or **41** (selectable in the UI)
- Azure AI Search for document retrieval
- Streamlit web interface

## Setup

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd o3-rag-demo
   ```

2. Create a Python virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your Azure credentials:
   - **OpenAI**: endpoint, key, API version, and deployment name for each model you plan to use  
     (variables: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_ENDPOINT_4o`, `AZURE_OPENAI_ENDPOINT_41`, etc.)
   - **Azure AI Search**: endpoint, key, and index name (optional but recommended)
## Environment setup

Use this template and fill in **your own** values:

```env
# ----- o3 -----
AZURE_OPENAI_ENDPOINT=<YOUR_O3_ENDPOINT>
AZURE_OPENAI_KEY=<YOUR_O3_KEY>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=o3

# ----- 4o -----
AZURE_OPENAI_ENDPOINT_4o=<YOUR_4O_ENDPOINT>
AZURE_OPENAI_KEY_4o=<YOUR_4O_KEY>
AZURE_OPENAI_API_VERSION_4o=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_4o=gpt-4o

# ----- 41 -----
AZURE_OPENAI_ENDPOINT_41=<YOUR_41_ENDPOINT>
AZURE_OPENAI_KEY_41=<YOUR_41_KEY>
AZURE_OPENAI_API_VERSION_41=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_41=gpt-4.1

# ----- (optional) Azure Cognitive Search -----
AZURE_SEARCH_ENDPOINT=<YOUR_SEARCH_ENDPOINT>
AZURE_SEARCH_KEY=<YOUR_SEARCH_KEY>
AZURE_SEARCH_INDEX=<YOUR_SEARCH_INDEX>
```

> **Note:** Never commit your real keys or endpoints.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload PDF files using the sidebar
2. Click "עבד וצור אינדקס" to process files
3. Ask questions in Hebrew in the main interface
4. View answers with source references

## Security

- `.env` file is excluded from git via `.gitignore`
- Never commit API keys or secrets
Make sure your `.env` follows the template above and is kept out of version control.

## Deactivating the Environment

When you're done working, deactivate the virtual environment:
```bash
deactivate
```
