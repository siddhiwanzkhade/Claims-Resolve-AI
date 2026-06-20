#uses the chunking method to every policy doc
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader

from src.rag.chunking import chunk_documents
from src.rag.pinecone_store import (
    create_pinecone_index_if_needed,
    clear_namespace,
    store_chunks
)

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-policy-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "claim-policies")

POLICY_DIR = "data/policy"


POLICY_METADATA = {
    "damaged_delivery_policy.pdf": {
        "company": "walmart",
        "policy_type": "damaged_delivery"
    },
    "replacement_policy.pdf": {
        "company": "bestbuy",
        "policy_type": "replacement"
    },
    "refund_policy.pdf": {
        "company": "ebay",
        "policy_type": "refund"
    },
    "return_policy.pdf": {
        "company": "ebay",
        "policy_type": "return_condition"
    },
    "manual_review_policy.pdf": {
        "company": "ebay",
        "policy_type": "manual_review"
    },
    "escalation_policy.pdf": {
        "company": "ebay",
        "policy_type": "fraud_escalation"
    },
    "evidence_requirements_policy.pdf": {
        "company": "ebay",
        "policy_type": "evidence_requirements"
    },
}


def load_policy_documents():
    all_docs = []

    for file_name in os.listdir(POLICY_DIR):
        file_path = os.path.join(POLICY_DIR, file_name)

        if not file_name.lower().endswith(".pdf"):
            print(f"Skipping non-PDF file: {file_name}")
            continue

        print(f"Loading PDF: {file_name}")

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        metadata = POLICY_METADATA.get(file_name, {})

        for doc in docs:
            doc.metadata["company"] = metadata.get("company", "unknown")
            doc.metadata["policy_type"] = metadata.get("policy_type", "unknown")
            doc.metadata["document_type"] = "ecommerce_claim_policy"
            doc.metadata["source_file"] = file_name

        all_docs.extend(docs)

    return all_docs


def print_chunk_preview(chunks, limit=20):
    for chunk in chunks[:limit]:
        print("\n" + "=" * 80)
        print(f"CHUNK {chunk.metadata['chunk_id']}")
        print(f"Source: {chunk.metadata.get('source_file')}")
        print(f"Company: {chunk.metadata.get('company')}")
        print(f"Policy Type: {chunk.metadata.get('policy_type')}")
        print(f"Section: {chunk.metadata.get('section_title')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print("-" * 80)
        print(chunk.page_content)


def main():
    create_pinecone_index_if_needed(INDEX_NAME)

    clear_namespace(INDEX_NAME, NAMESPACE)

    docs = load_policy_documents()
    print(f"Loaded {len(docs)} pages/documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print_chunk_preview(chunks, limit=20)

    store_chunks(chunks, INDEX_NAME, NAMESPACE)


if __name__ == "__main__":
    main()