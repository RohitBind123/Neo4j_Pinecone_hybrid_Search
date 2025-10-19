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
   ```
3. **Set up environment variables:**
   Create a `.env` file in the root directory and add the following environment variables:
   ```
   PINECONE_API_KEY="your-pinecone-api-key"
   NEO4J_URI="your-neo4j-uri"
   NEO4J_USERNAME="your-neo4j-username"
   NEO4J_PASSWORD="your-neo4j-password"
   GROQ_API_KEY="your-groq-api-key"
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
