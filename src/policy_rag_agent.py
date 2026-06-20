import json
from typing import Optional

from src.rag.retriever import retrieve_policy_chunks


def infer_policy_type(issue_type: str, requested_action: str) -> Optional[str]:
    """
    Maps claim intent to the most relevant policy_type metadata.
    This narrows Pinecone retrieval to the most relevant policy subset.
    """

    issue_type = (issue_type or "").lower()
    requested_action = (requested_action or "").lower()

    if "damage" in issue_type or "damaged" in issue_type or "defective" in issue_type:
        if requested_action == "replacement":
            return "replacement"
        if requested_action == "refund":
            return "refund"
        return "damaged_delivery"

    if "wrong" in issue_type or "incorrect" in issue_type:
        return "replacement"

    if "missing" in issue_type or "not_received" in issue_type:
        return "refund"

    if "return" in issue_type or "return" in requested_action:
        return "return_condition"

    if "manual" in requested_action or "escalation" in requested_action:
        return "manual_review"

    return None


def build_policy_query(
    complaint_result: dict,
    order_result: dict,
    vision_result: dict
) -> str:
    """
    Converts outputs from complaint/order/vision agents into a retrieval query.
    """

    issue_type = complaint_result.get("issue_type", "unclear")
    requested_action = complaint_result.get("requested_action", "unclear")
    sentiment = complaint_result.get("sentiment", "neutral")
    urgency = complaint_result.get("urgency", "low")
    summary = complaint_result.get("summary", "")

    return_window_status = order_result.get("return_window_status", "unknown")
    delivery_issue = order_result.get("delivery_issue", "unknown")
    order_value_risk = order_result.get("order_value_risk", "unknown")
    customer_claim_risk = order_result.get("customer_claim_risk", "unknown")

    damage_detected = vision_result.get("damage_detected", "unknown")
    damage_type = vision_result.get("damage_type", "unknown")
    damage_severity = vision_result.get("damage_severity", "unknown")
    visible_evidence = vision_result.get("visible_evidence", "")
    vision_confidence = vision_result.get("confidence", "unknown")

    query = f"""
    E-commerce visual claim policy lookup.

    Customer complaint:
    issue_type: {issue_type}
    requested_action: {requested_action}
    sentiment: {sentiment}
    urgency: {urgency}
    summary: {summary}

    Order context:
    return_window_status: {return_window_status}
    delivery_issue: {delivery_issue}
    order_value_risk: {order_value_risk}
    customer_claim_risk: {customer_claim_risk}

    Visual evidence:
    damage_detected: {damage_detected}
    damage_type: {damage_type}
    damage_severity: {damage_severity}
    visible_evidence: {visible_evidence}
    confidence: {vision_confidence}

    Retrieve policy sections about refund eligibility, replacement eligibility,
    damaged or defective item handling, return windows, evidence requirements,
    false claims, manual review, and escalation.
    """

    return query.strip()


def run_policy_rag_agent(
    complaint_result: dict,
    order_result: dict,
    vision_result: dict,
    company: str,
    top_k: int = 3
) -> dict:
    """
    Main Policy RAG Agent.

    This function retrieves relevant policy evidence from Pinecone.
    It assumes Pinecone has already been built using build_policy_index.py.
    """

    issue_type = complaint_result.get("issue_type", "unclear")
    requested_action = complaint_result.get("requested_action", "unclear")

    policy_type = infer_policy_type(
        issue_type=issue_type,
        requested_action=requested_action
    )

    query = build_policy_query(
        complaint_result=complaint_result,
        order_result=order_result,
        vision_result=vision_result
    )

    docs = retrieve_policy_chunks(
        query=query,
        company=company,
        policy_type=policy_type,
        k=top_k
    )

    evidence = []

    for doc in docs:
        evidence.append({
            "source_file": doc.metadata.get("source_file"),
            "company": doc.metadata.get("company"),
            "policy_type": doc.metadata.get("policy_type"),
            "section_title": doc.metadata.get("section_title"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "content": doc.page_content
        })

    return {
        "query": query,
        "company_filter": company,
        "policy_type_filter": policy_type,
        "retrieved_evidence": evidence
    }


if __name__ == "__main__":
    sample_complaint = {
        "issue_type": "damaged_item",
        "requested_action": "replacement",
        "sentiment": "negative",
        "urgency": "high",
        "summary": "Headphones arrived broken and the box was crushed."
    }

    sample_order = {
        "return_window_status": "within_return_window",
        "delivery_issue": "delivered",
        "order_value_risk": "medium",
        "customer_claim_risk": "low"
    }

    sample_vision = {
        "damage_detected": True,
        "damage_type": "broken_component",
        "damage_severity": "medium",
        "visible_evidence": "The earbud component appears detached.",
        "confidence": "high"
    }

    result = run_policy_rag_agent(
        complaint_result=sample_complaint,
        order_result=sample_order,
        vision_result=sample_vision,
        company="bestbuy",
        top_k=3
    )

    print(json.dumps(result, indent=2))