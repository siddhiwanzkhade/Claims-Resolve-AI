import os
import hashlib

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.rag.embeddings import get_embedding_model, EMBEDDING_DIMENSION


def create_pinecone_index_if_needed(index_name: str):
    api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Pinecone index already exists: {index_name}")


def get_vectorstore(index_name: str, namespace: str):
    embeddings = get_embedding_model()

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )


def build_stable_ids(chunks):
    ids = []

    for chunk in chunks:
        key = "_".join(
            str(chunk.metadata.get(k, ""))
            for k in (
                "source_file",
                "section_index",
                "section_piece_index",
                "chunk_id"
            )
        )

        ids.append(hashlib.md5(key.encode("utf-8")).hexdigest())

    return ids

#clears old chunks
def clear_namespace(index_name: str, namespace: str):
    api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    index.delete(
        delete_all=True,
        namespace=namespace
    )

    print(f"Cleared Pinecone namespace: {namespace}")

def store_chunks(chunks, index_name: str, namespace: str):
    vectorstore = get_vectorstore(index_name, namespace)

    ids = build_stable_ids(chunks)

    vectorstore.add_documents(
        chunks,
        ids=ids
    )

    print(f"Stored {len(chunks)} chunks in Pinecone namespace: {namespace}")

