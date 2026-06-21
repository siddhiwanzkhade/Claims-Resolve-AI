# Claims-Resolve AI

Claims-Resolve AI is an agentic, multimodal solution designed to streamline and automate e-commerce refund and replacement claims. By leveraging specialized AI agents through LangGraph, it enhances efficiency, accuracy, and policy compliance in claims management.

## Features

* **Automated Claim Processing**: Utilizes AI capabilities for complaint analysis, image-based damage assessment, policy retrieval, and risk/mismatch detection to resolve claims efficiently.
* **Image Analysis**: Employs Vision Language Models (Groq Vision) to evaluate uploaded product/package images for damage type, severity, and evidence quality.
* **Natural Language Processing**: Analyzes customer complaint text to extract issue type, requested action, sentiment, and urgency relevant to the claim.
* **Policy-RAG Retrieval**: Retrieves retailer-specific policy clauses via Pinecone + SentenceTransformer embeddings to ground every decision in actual policy text rather than guesswork.
* **Risk & Mismatch Detection**: Flags inconsistencies between complaint, order, and evidence signals, and scores overall claim risk to safeguard against potentially fraudulent or unsupported claims.
* **Schema-Validated Outputs**: Every agent's output is validated against a Pydantic schema before moving forward, preventing malformed or hallucinated outputs from propagating.
* **Manual Review Fallback**: Routes unclear, unsupported, missing-evidence, or high-risk claims to manual review instead of blindly approving them.
* **Gradio Dashboard**: Provides a local interface for submitting complaints, product images, retailer selection, and order metadata.

## Agent Architecture

* **LangGraph Supervisor**: Controls claim routing and decides which step runs next.
* **Complaint Agent**: Extracts issue type, requested action, sentiment, urgency, and summary from customer text.
* **Order Agent**: Evaluates order status, delivery status, return window, order value, and claim history.
* **Vision Agent**: Analyzes product/package images for visible damage evidence.
* **Policy RAG Agent**: Retrieves relevant retailer-specific policy clauses from Pinecone.
* **Risk Scoring Agent**: Computes claim risk and identifies mismatch/risk flags.
* **Resolution Agent**: Generates the final claim decision using complaint, order, vision, policy, and risk signals.


```mermaid
flowchart LR
    A["Claim submitted<br/>Item arrived cracked, replacement requested"] --> B["Complaint Agent<br/>issue: damaged, action: replacement"]
    B --> C["Order Agent<br/>delivered, within return window"]
    C --> D["Vision Agent<br/>crack detected, evidence quality: high"]
    D --> E["Schema validation<br/>VisionAnalysis"]
    E --> F["Policy RAG Agent<br/>retrieves replacement policy clause"]
    F --> G["Risk Scoring Agent<br/>low risk, no mismatch flags"]
    G --> H["Resolution Agent<br/>fuses all agent outputs"]
    H --> I["approve_replacement"]

    style A fill:#f1efe8,stroke:#888780
    style I fill:#eaf3de,stroke:#639922
```
## Example Claim Paths

```text
Damaged item with replacement request
→ Complaint Agent → Order Agent → Vision Agent → Policy RAG Agent → Risk Scoring Agent → Resolution Agent
→ approve_replacement

Damaged item with refund request
→ Complaint Agent → Order Agent → Vision Agent → Policy RAG Agent → Risk Scoring Agent → Resolution Agent
→ approve_refund

Damaged item without image evidence
→ Input Validation → LangGraph Supervisor
→ request_visual_evidence

High-risk or unclear claim
→ Complaint Agent → Order Agent → Vision Agent → Policy RAG Agent → Risk Scoring Agent → Resolution Agent
→ manual_review
```

## Tech Stack

| Component | Tool |
|---|---|
| Agent orchestration | LangGraph, LangChain |
| Complaint analysis (LLM) | Groq-hosted LLM |
| Image analysis (VLM) | Groq Vision |
| Policy embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Pinecone |
| Output validation | Pydantic |
| PDF ingestion | PyPDF |
| Dashboard | Gradio |



## Supported Models

Claims-Resolve AI uses `llama-3.1-8b-instant` through Groq for complaint understanding and `meta-llama/llama-4-scout-17b-16e-instruct` through Groq Vision for product/package image analysis.

The Complaint Agent extracts issue type, requested action, sentiment, urgency, and summary from customer text. The Vision Agent detects visible damage, damage type, severity, evidence quality, and confidence from uploaded images.

For Policy-RAG retrieval, the system uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with Pinecone as the vector database. An optional local MLX vision backend using `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` is included as an extension, but Groq Vision is the primary working backend.

The workflow is orchestrated using LangGraph and structured outputs are validated using Pydantic.

## Installation

1. Clone the Repository:
