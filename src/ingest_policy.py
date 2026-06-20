import os
import re
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-policy-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "claim-policy")
POLICY_DIR = "data/policy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Tune these per your actual content. Sections under this size skip the
# secondary recursive split entirely (kept as one atomic chunk).
SECTION_MAX_CHARS = 1200
RECURSIVE_CHUNK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 150

POLICY_METADATA = {
    "damaged_delivery_policy.pdf": {
        "company": "walmart",
        "policy_type": "damaged_delivery",
    },
    "replacement_policy.pdf": {
        "company": "bestbuy",
        "policy_type": "replacement",
    },
    "refund_policy.pdf": {
        "company": "ebay",
        "policy_type": "refund",
    },
    "return_policy.pdf": {
        "company": "ebay",
        "policy_type": "return_condition",
    },
    "manual_review_policy.pdf": {
        "company": "ebay",
        "policy_type": "manual_review",
    },
    "escalation_policy.pdf": {
        "company": "ebay",
        "policy_type": "fraud_escalation",
    },
    "evidence_requirements_policy.pdf": {
        "company": "ebay",
        "policy_type": "evidence_requirements",
    },
}

# Matches the repeated page header/footer boilerplate, e.g.:
#   NovaCart Commerce Inc.
#   Doc Code: BBY-RTN
#   Best Buy Return & Exchange Policy
#   Page 1
# This pattern is intentionally specific to your doc template. If a new
# source file uses a different header format, extend this pattern or add
# a second one and try both.
BOILERPLATE_PATTERN = re.compile(
    r"NovaCart Commerce Inc\.\s*\n"
    r"Doc Code:\s*[A-Z0-9\-]+\s*\n"
    r".*?\n"          # the "<Company> Return & Exchange Policy" line (varies)
    r"Page\s*\d+\s*\n?",
    re.MULTILINE,
)

# Matches numbered section headers like "1. Return and Exchange Periods"
# Captures the header text in group 1. Requires the header to start a new
# line and be followed by a newline (i.e. it's on its own line).
SECTION_PATTERN = re.compile(r"\n(\d{1,2}\.\s[A-Z][^\n]{3,90})\n")


def create_pinecone_index_if_needed():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"Pinecone index already exists: {INDEX_NAME}")


def load_policy_documents():
    all_docs = []
    for file_name in os.listdir(POLICY_DIR):
        file_path = os.path.join(POLICY_DIR, file_name)
        if file_name.lower().endswith(".pdf"):
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
        else:
            print(f"Skipping non-PDF file: {file_name}")
    return all_docs


def strip_boilerplate(text):
    return BOILERPLATE_PATTERN.sub("", text)


def split_by_sections(text):
    """
    Splits text on numbered section headers (e.g. "1. Return Periods").
    Returns (preamble, sections) where sections is a list of
    (header, body) tuples. If no section headers are found, sections
    will be empty and the caller should fall back to plain recursive
    splitting on the full text.
    """
    parts = SECTION_PATTERN.split(text)
    if len(parts) < 3:
        # No section headers matched at all.
        return text, []

    preamble = parts[0].strip()
    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((header, body))
    return preamble, sections


def chunk_documents(docs):
    """
    Section-aware chunking:
      1. Groups per-page Documents back into one full text per source file
         (so sections that span a page break aren't pre-fragmented).
      2. Strips repeated page-header/footer boilerplate.
      3. Splits on numbered section headers, keeping header + body together.
      4. Only recursively splits a section further if it exceeds
         SECTION_MAX_CHARS, so short sections (and short tables) stay atomic.
      5. Falls back to plain recursive splitting for any source file where
         no section headers are detected, so this never silently drops content.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Group pages by source file, preserving page order.
    by_source = {}
    for doc in docs:
        by_source.setdefault(doc.metadata["source_file"], []).append(doc)

    all_chunks = []

    for source_file, page_docs in by_source.items():
        page_docs_sorted = sorted(page_docs, key=lambda d: d.metadata.get("page", 0))
        base_metadata = dict(page_docs_sorted[0].metadata)

        full_text = "\n".join(
            strip_boilerplate(d.page_content) for d in page_docs_sorted
        )

        preamble, sections = split_by_sections(full_text)

        if not sections:
            # No numbered-section structure detected in this doc.
            # Fall back to plain recursive splitting on the whole (cleaned) text.
            print(
                f"  [no section headers detected in {source_file}, "
                f"using plain recursive split]"
            )
            sub_chunks = recursive_splitter.split_text(full_text)
            for sub in sub_chunks:
                meta = dict(base_metadata)
                meta["section_title"] = None
                all_chunks.append(Document(page_content=sub, metadata=meta))
            continue

        for section_index, (header, body) in enumerate(sections):
            combined = f"{header}\n{body}".strip()
            if len(combined) <= SECTION_MAX_CHARS:
                pieces = [combined]
            else:
                pieces = recursive_splitter.split_text(combined)

            for piece_index, piece in enumerate(pieces):
                meta = dict(base_metadata)
                meta["section_title"] = header
                meta["section_index"] = section_index
                meta["section_piece_index"] = piece_index
                all_chunks.append(Document(page_content=piece, metadata=meta))

    # Stable, deterministic chunk_id derived from content + position,
    # rather than a pure incrementing counter. Useful for debugging and
    # paired with the stable Pinecone vector IDs below.
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    return all_chunks


def build_stable_ids(chunks):
    """
    Deterministic IDs based on source file + section + piece, so re-running
    ingestion overwrites existing vectors instead of creating duplicates.
    """
    ids = []
    for chunk in chunks:
        key = "_".join(
            str(chunk.metadata.get(k, ""))
            for k in ("source_file", "section_index", "section_piece_index", "chunk_id")
        )
        ids.append(hashlib.md5(key.encode("utf-8")).hexdigest())
    return ids


def store_chunks_in_pinecone(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )
    ids = build_stable_ids(chunks)
    vectorstore.add_documents(chunks, ids=ids)
    print(f"Stored {len(chunks)} chunks in Pinecone namespace: {NAMESPACE}")


def print_chunk_preview(chunks, limit=None):
    """
    Same inspection format you've been using to debug chunk quality.
    Call with limit=None to print all chunks, or e.g. limit=15 for a quick look.
    """
    preview_chunks = chunks if limit is None else chunks[:limit]
    for chunk in preview_chunks:
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
    create_pinecone_index_if_needed()
    docs = load_policy_documents()
    print(f"Loaded {len(docs)} pages/documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Inspect before storing 
    print_chunk_preview(chunks, limit=20)

    store_chunks_in_pinecone(chunks)


if __name__ == "__main__":
    main()