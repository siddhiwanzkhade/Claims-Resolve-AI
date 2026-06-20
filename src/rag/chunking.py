import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SECTION_MAX_CHARS = 1400
RECURSIVE_CHUNK_SIZE = 1200
RECURSIVE_CHUNK_OVERLAP = 200


BOILERPLATE_PATTERN = re.compile(
    r"NovaCart Commerce Inc\.\s*\n"
    r"Doc Code:\s*[A-Z0-9\-]+\s*\n"
    r".*?\n"
    r"Page\s*\d+\s*\n?",
    re.MULTILINE,
)

SECTION_PATTERN = re.compile(r"\n(\d{1,2}\.\s[A-Z][^\n]{3,90})\n")


def strip_boilerplate(text: str) -> str:
    return BOILERPLATE_PATTERN.sub("", text)


def split_by_sections(text: str):
    parts = SECTION_PATTERN.split(text)

    if len(parts) < 3:
        return text, []

    preamble = parts[0].strip()
    sections = []

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((header, body))

    return preamble, sections


def chunk_documents(docs):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    by_source = {}

    for doc in docs:
        by_source.setdefault(doc.metadata["source_file"], []).append(doc)

    all_chunks = []

    for source_file, page_docs in by_source.items():
        page_docs_sorted = sorted(
            page_docs,
            key=lambda d: d.metadata.get("page", 0)
        )

        base_metadata = dict(page_docs_sorted[0].metadata)

        full_text = "\n".join(
            strip_boilerplate(d.page_content) for d in page_docs_sorted
        )

        preamble, sections = split_by_sections(full_text)

        if not sections:
            print(f"[No section headers detected in {source_file}; using recursive split]")

            sub_chunks = recursive_splitter.split_text(full_text)

            for sub in sub_chunks:
                meta = dict(base_metadata)
                meta["section_title"] = None
                all_chunks.append(
                    Document(page_content=sub, metadata=meta)
                )

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

                all_chunks.append(
                    Document(page_content=piece, metadata=meta)
                )

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    return all_chunks