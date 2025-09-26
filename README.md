# Document Search AI (RAG-based)

A Streamlit application for semantic document search leveraging FAISS, Sentence Transformers, and large language models like Groq LLaMA or OpenAI GPT.

---

## Features

- Upload documents (PDF, DOCX, TXT) to create searchable semantic chunks
- Use vector similarity search with FAISS
- Integrate with Groq or OpenAI LLMs for question answering
- Configurable and extensible architecture

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/document-search-ai.git
cd document-search-ai
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set up environment variables

Rename .env.example to .env

Fill in your API keys inside .env

env
Copy code
GROQ_API_KEY=your-groq-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
Run the app

bash
Copy code
streamlit run main.py
Usage
Upload your documents through the Streamlit UI

The system will create semantic embeddings and store them in FAISS

Ask natural language questions to search through your documents

Supported Models
Groq LLaMA models

OpenAI GPT models

Sentence Transformers for embeddings
