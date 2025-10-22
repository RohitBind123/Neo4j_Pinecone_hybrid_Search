
import streamlit as st
import asyncio
import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langsmith import traceable
# Load environment variables
load_dotenv()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Hybrid Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

# ----------------------------
# Initialize Groq client
# ----------------------------
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ----------------------------
# Langsmith settings
# ----------------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY

# ----------------------------
# Config
# ----------------------------
TOP_K = config.TOP_K
INDEX_NAME = config.PINECONE_INDEX_NAME

# ----------------------------
# Initialize clients
# ----------------------------
@st.cache_resource
def get_clients():
    """Get clients for HuggingFace, Pinecone, and Neo4j."""
    # Initialize the Hugging Face embedding model.
    client = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize the Pinecone client.
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Connect to Pinecone index
    if INDEX_NAME not in pc.list_indexes().names():
        st.info(f"Creating managed index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=config.PINECONE_REGION)
        )

    index = pc.Index(INDEX_NAME)

    # Connect to Neo4j
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    
    return client, index, driver

client, index, driver = get_clients()

# ----------------------------
# Helper functions
# ----------------------------
embedding_cache = {}

async def embed_text(text: str) -> List[float]:
    """Get embedding for a text string."""
    if text in embedding_cache:
        return embedding_cache[text]
    
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, client.embed_documents, [text])
    embedding_cache[text] = resp[0]
    return resp[0]

@traceable
async def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = await embed_text(query_text)
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, lambda: index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    ))
    return res["matches"]

@traceable
async def fetch_graph_context(node_ids: List[str]):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    loop = asyncio.get_event_loop()
    with driver.session() as session:
        for nid in node_ids:
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
    system = (
        "You are a helpful travel assistant. Your task is to answer the user's query based on the provided context."
        "First, summarize the user's query to understand their intent."
        "Then, analyze the semantic search results and the graph facts to identify the key entities and relationships."
        "Finally, generate a concise and helpful response that directly answers the user's question, citing node ids when referencing specific places or attractions."
        "Think step-by-step."
    )

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

@traceable
async def call_chat(prompt_messages):
    """Calling groq chat model."""
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(model=config.GROQ_MODEL, messages=prompt_messages))
    return resp.choices[0].message.content

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Hybrid Travel Assistant ✈️")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your travel question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async function to get the answer
                matches = loop.run_until_complete(pinecone_query(prompt, top_k=TOP_K))
                match_ids = [m["id"] for m in matches]
                graph_facts = loop.run_until_complete(fetch_graph_context(match_ids))
                chat_prompt = build_prompt(prompt, matches, graph_facts)
                answer = loop.run_until_complete(call_chat(chat_prompt))
                
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Close the event loop
                loop.close()
