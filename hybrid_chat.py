# hybrid_chat_async.py
import asyncio
import aiohttp
import os
import json
from typing import List
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# This script implements an asynchronous version of the hybrid chat system.
# It uses asyncio to run the I/O-bound operations concurrently, which improves the responsiveness of the application.

# -----------------------------
# Initialize Groq client
# -----------------------------
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


# -----------------------------
# Config
# -----------------------------
TOP_K =config.TOP_K
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
# Initialize the Hugging Face embedding model.
client = client =HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the Pinecone client.
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))


# -----------------------------
# Helper functions
# -----------------------------
# Create a cache for the embeddings.
embedding_cache = {}

async def embed_text(text: str) -> List[float]:
    """Get embedding for a text string."""
    # Check if the embedding is already in the cache.
    if text in embedding_cache:
        return embedding_cache[text]
    # If the embedding is not in the cache, generate it and add it to the cache.
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, client.embed_documents, [text])
    embedding_cache[text] = resp
    return resp

async def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    # Get the embedding for the query text.
    vec = await embed_text(query_text)
    # Query the Pinecone index.
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, lambda: index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    ))
    print("DEBUG: Pinecone top 5 results:")
    print(len(res["matches"]))
    return res["matches"]

async def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    loop = asyncio.get_event_loop()
    with driver.session() as session:
        for nid in node_ids:
            # Query the Neo4j database to get the neighbors of the node.
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = await loop.run_in_executor(None, lambda: session.run(q, nid=nid))
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def search_summary(pinecone_matches):
    """Summarize the top nodes from the Pinecone search."""
    summary = []
    for m in pinecone_matches:
        meta = m["metadata"]
        summary.append(f"- {meta.get('name', '')} ({meta.get('type', '')} in {meta.get('city', 'N/A')}) - Score: {m.get('score', 0):.2f}")
    return "\n".join(summary)

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    # The system prompt provides instructions to the language model.
    system = (
        "You are a helpful travel assistant. Your task is to answer the user's query based on the provided context."
        "First, summarize the user's query to understand their intent."
        "Then, analyze the semantic search results and the graph facts to identify the key entities and relationships."
        "Finally, generate a concise and helpful response that directly answers the user's question, citing node ids when referencing specific places or attractions."
        "Think step-by-step."
    )

    # The user prompt contains the user's query and the context from the semantic search and graph search.
    summary = search_summary(pinecone_matches)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         f"Top semantic matches (from vector DB):\n{summary}\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
    ]
    return prompt

async def call_chat(prompt_messages):


    """Calling groq chat model."""
    # Call the Groq chat model to get a response.
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(model= config.GROQ_MODEL, messages=prompt_messages))
    return resp.choices[0].message.content

# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():
    """Run an interactive chat session."""
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        # Get the user's query.
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")
        query = query.strip()
        if not query or query.lower() in ("exit","quit"):
            break

        # Get the semantic search results from Pinecone.
        matches = await pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        # Get the graph context from Neo4j.
        graph_facts = await fetch_graph_context(match_ids)

        # Build the prompt for the language model.
        prompt = build_prompt(query, matches, graph_facts)
        # Get the answer from the language model.
        answer = await call_chat(prompt)
        # Print the answer.
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    asyncio.run(interactive_chat())
