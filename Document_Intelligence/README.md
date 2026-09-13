# 10 AI Document Intelligence

Enterprise document RAG with ingestion, retrieval, audit logs and analytics.

## Advanced architecture
- Authentication with hashed passwords
- SQLite database
- PDF ingestion
- RAG-style document retrieval with source/chunk tracking
- IBM watsonx.ai foundation-model generation
- Specialist AI agent roles
- Analytics dashboard
- Polished responsive UI

## Setup
`python -m venv venv`
`venv\\Scripts\\activate` on Windows
`pip install -r requirements.txt`
Copy `.env.example` to `.env`, add IBM credentials, then run `python app.py`.

## Production RAG
The starter includes a working retrieval layer. For production, replace the local retriever with IBM's persistent vector-store/RAG extension or a connected Milvus/Elasticsearch/Chroma store. IBM documents vector embeddings, retrieval, reranking and foundation-model generation as the standard RAG pattern.
