# RAG Demo with Azure o3

A Retrieval-Augmented Generation (RAG) application using Azure OpenAI o3 model and Azure AI Search.

## Features

- Upload multiple PDF files
- Ask questions in Hebrew
- Uses Azure OpenAI o3 model for generation
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

4. Configure Azure services in `.env`:
   - Add your Azure Search endpoint and key
   - OpenAI credentials are already configured

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

## Deactivating the Environment

When you're done working, deactivate the virtual environment:
```bash
deactivate
```
