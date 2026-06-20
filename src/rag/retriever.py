import os
from dotenv import load_dotenv

from src.rag.pinecone_store import get_vectorstore


load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-policy-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "claim-policies")


def retrieve_policy_chunks(
    query: str,
    company: str | None = None,
    policy_type: str | None = None,
    k: int = 5
):
    vectorstore = get_vectorstore(INDEX_NAME, NAMESPACE)

    metadata_filter = {}

    if company:
        metadata_filter["company"] = company

    if policy_type:
        metadata_filter["policy_type"] = policy_type

    if metadata_filter:
        return vectorstore.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter
        )

    return vectorstore.similarity_search(
        query=query,
        k=k
    )