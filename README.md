# Hybrid AI Travel Assistant

This is a Hybrid AI Travel Assistant that uses a combination of Pinecone vector search, Neo4j graph database, and a Large Language Model (LLM) to provide intelligent travel recommendations.

## Features

- **Hybrid Search:** Combines semantic search with graph-based search to provide more accurate and context-aware recommendations.
- **Interactive Chat:** A command-line interface to interact with the travel assistant.
- **Cost-Effective:** Uses open-source alternatives to paid services to reduce costs.
- **Scalable:** Built with an asynchronous architecture to handle concurrent requests.
- **Graph Visualization:** An interactive visualization of the Neo4j graph database.

## Technologies Used

- **Python 3.12**
- **Pinecone:** For vector search.
- **Neo4j:** For graph database.
- **Groq:** For the Large Language Model.
- **HuggingFace:** For sentence embeddings.
- **uv:** For dependency management.
- **pyvis:** For graph visualization.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/neo4-hybrid-chat.git
   ```
2. **Install dependencies:**
   ```bash

   uv pip install -r requirements.txt
   or install dependencies from project.toml
   by uv syncaiohttp>=3.13.1
   asyncio>=4.0.0
   dotenv>=0.9.9
   groq>=0.32.0
   langchain>=1.0.0
   langchain-google-genai>=3.0.0
   langchain-huggingface>=1.0.0
   neo4j>=5.15.0
   networkx==3.1
   pinecone>=7.3.0
   python-dotenv>=1.1.1
   pyvis==0.3.1
   sentence-transformers>=5.1.1
   tqdm>=4.67.1
   langsmith

   ```
3. **Set up environment variables:**
   Create a `config.py file with these varibale to run hybrid_chat.py` file in the root directory and add the following environment variables:
   ```
   NEO4J_URI = "neo4j+ssc://dummy-host.databases.neo4j.io"
   NEO4J_USERNAME = "dummy_user"
   NEO4J_PASSWORD = "dummy_password"

   GROQ_API_KEY = "gsk_dummykey"

   PINECONE_API_KEY = "pcsk_dummyapikey"
   PINECONE_ENV = "us-east1"
   PINECONE_INDEX_NAME = "dummy-index"
   PINECONE_VECTOR_DIM = 768

   GROQ_MODEL = "openai/dummy-model"
   TOP_K = 5
   LANGSMITH_API_KEY=<"your langsmith Api key">

   ```

## Usage

1. **Load data to Neo4j:**
   ```bash
   python load_to_neo4j.py
   ```
2. **Upload data to Pinecone:**
   ```bash
   python pinecone_upload.py
   ```
3. **Start the interactive chat:**
   ```bash
   python hybrid_chat.py
   ```
4. **Visualize the graph:**
   ```bash
   python visualize_graph.py
   ```
   This will create an `neo4j_viz.html` file in the root directory. Open this file in your browser to see the interactive graph visualization.

## Configuration

The `config.py` file contains the configuration for the project. You can change the following settings:

- `PINECONE_INDEX_NAME`: The name of the Pinecone index.
- `PINECONE_DIMENSION`: The dimension of the vectors.
- `GROQ_MODEL`: The name of the Groq model to use.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
