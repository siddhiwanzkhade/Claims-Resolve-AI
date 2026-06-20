from src.rag.retriever import retrieve_policy_chunks


def main():
    query = """
    Customer says their headphones arrived damaged during shipping.
    They want a replacement.
    """

    results = retrieve_policy_chunks(
        query=query,
        company="bestbuy",
        policy_type="replacement",
        k=3
    )

    for i, doc in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print(f"RESULT {i}")
        print("Source:", doc.metadata.get("source_file"))
        print("Company:", doc.metadata.get("company"))
        print("Policy Type:", doc.metadata.get("policy_type"))
        print("Section:", doc.metadata.get("section_title"))
        print("-" * 80)
        print(doc.page_content[:1200])


if __name__ == "__main__":
    main()