# config_example.py — copy to config.py and fill with real values.
NEO4J_URI="neo4j+ssc://dummy_uri"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="dummy_password"

GROQ_API_KEY ="dummy_api_key"
PINECONE_API_KEY = "dummy_api_key" # your Pinecone API key
PINECONE_ENV = "dummy_env"   # example
PINECONE_INDEX_NAME = "dummy_index_name"
PINECONE_VECTOR_DIM = 768   # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
GROQ_API_KEY="dummy_api_key"
GROQ_MODEL="openai/gpt-oss-120b"
TOP_K = 5
LANGSMITH_API_KEY="dummy_api_key"  # your LangSmith API key for observability
