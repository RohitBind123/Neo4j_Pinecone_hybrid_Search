# pinecone_upload.py
import os 
import json
import time
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
import config
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# This script loads data from a JSON file, generates embeddings for the data, and uploads them to a Pinecone index.

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset .json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # using gemini-embedding-001 model with 768 dimensions

# -----------------------------
# Initialize clients
# -----------------------------
# Initialize the Hugging Face embedding model.
client =HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Initialize the Pinecone client.
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
# Check if the index already exists.
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    # If the index does not exist, create it.
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,           # dimensionality of dense model embeddings
        metric="dotproduct",     # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts):
    """Generate embeddings using hugging face embedding model ."""
    resp = client.embed_documents(texts)
    return resp

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    """Load data from the JSON file, generate embeddings, and upload them to the Pinecone index."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        # Get the semantic text from the node, or use the description if it is not available.
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        # Create the metadata for the vector.
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    # Upload the data in batches.
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        # Generate embeddings for the texts.
        embeddings = get_embeddings(texts)

        # Create the vectors to be uploaded.
        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        # Upload the vectors to the Pinecone index.
        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
