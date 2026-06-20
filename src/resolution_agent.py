import json


def extract_policy_text(policy_result: dict) -> str:
    """
    Combines retrieved policy chunks into one searchable text block.
    """

    evidence = policy_result.get("retrieved_evidence", [])
    policy_text_parts = []

    for item in evidence:
        section = item.get("section_title", "Unknown Section")
        content = item.get("content", "")
        policy_text_parts.append(f"{section}\n{content}")

    return "\n\n".join(policy_text_parts)


def get_policy_basis(policy_result: dict) -> list:
    """
    Keeps retrieved policy evidence in the final output for auditability.
    """

    evidence = policy_result.get("retrieved_evidence", [])
    policy_basis = []

    for item in evidence:
        policy_basis.append({
            "source_file": item.get("source_file"),
            "company": item.get("company"),
            "policy_type": item.get("policy_type"),
            "section_title": item.get("section_title"),
            "chunk_id": int(item.get("chunk_id")) if item.get("chunk_id") is not None else None,
            "content": item.get("content")
        })

    return policy_basis


def resolve_claim(
    complaint_result: dict,
    order_result: dict,
    vision_result: dict,
    policy_result: dict
) -> dict:
    """
    Combines complaint analysis, order metadata analysis, visual evidence analysis,
    and retrieved policy evidence to recommend a final claim resolution.

    This is the policy-grounded decision layer of the system.
    """

    # Extract complaint signals
    issue_type = complaint_result.get("issue_type", "unclear")
    requested_action = complaint_result.get("requested_action", "unclear")
    sentiment = complaint_result.get("sentiment", "neutral")
    urgency = complaint_result.get("urgency", "low")
    complaint_summary = complaint_result.get("summary", "")

    # Extract order signals
    return_window_status = order_result.get("return_window_status", "unknown")
    delivery_issue = order_result.get("delivery_issue", "unknown")
    order_value_risk = order_result.get("order_value_risk", "unknown")
    customer_claim_risk = order_result.get("customer_claim_risk", "unknown")
    order_hint = order_result.get("recommended_action_hint", "policy_check_required")

    # Extract vision/image/video signals
    damage_detected = vision_result.get("damage_detected", False)
    damage_type = vision_result.get("damage_type", "unclear")
    damage_severity = vision_result.get("damage_severity", "unknown")
    visible_evidence = vision_result.get("visible_evidence", "")
    vision_confidence = vision_result.get("confidence", "low")
    vision_backend = vision_result.get("model_backend", "unknown")

    # Extract policy/RAG signals
    policy_text = extract_policy_text(policy_result).lower()
    policy_basis = get_policy_basis(policy_result)

    policy_supports_replacement = (
        "replacement" in policy_text
        and ("damaged" in policy_text or "defective" in policy_text)
    )

    policy_supports_refund = (
        "refund" in policy_text
        or "money back" in policy_text
    )

    policy_mentions_manual_review = (
        "manual review" in policy_text
        or "may be denied" in policy_text
        or "verification" in policy_text
    )

    reasons = []
    risk_flags = []

    # Build explanation reasons
    if issue_type != "unclear":
        reasons.append(f"Complaint indicates issue type: {issue_type}.")

    if requested_action != "unclear":
        reasons.append(f"Customer requested: {requested_action}.")

    if complaint_summary:
        reasons.append(f"Complaint summary: {complaint_summary}")

    if damage_detected:
        reasons.append(
            f"Visual evidence supports damage: {damage_type} with {damage_severity} severity."
        )
    else:
        reasons.append("Visual evidence does not clearly confirm visible damage.")

    if visible_evidence:
        reasons.append(f"Visible evidence: {visible_evidence}")

    if return_window_status == "within_return_window":
        reasons.append("Order is within the return/replacement window.")
    elif return_window_status == "outside_return_window":
        reasons.append("Order is outside the return/replacement window.")

    if customer_claim_risk == "high":
        reasons.append("Customer has high previous-claim risk.")
        risk_flags.append("high_customer_claim_risk")
    elif customer_claim_risk == "medium":
        reasons.append("Customer has medium previous-claim risk.")
        risk_flags.append("medium_customer_claim_risk")
    elif customer_claim_risk == "low":
        reasons.append("Customer has low previous-claim risk.")

    if order_value_risk == "high":
        reasons.append("Order value is high, so manual review may be required.")
        risk_flags.append("high_order_value")

    if vision_confidence == "low":
        risk_flags.append("low_visual_confidence")

    if issue_type == "damaged_item" and damage_detected is False:
        risk_flags.append("visual_claim_mismatch")

    if return_window_status == "outside_return_window":
        risk_flags.append("outside_return_window")

    if policy_basis:
        top_policy = policy_basis[0]
        reasons.append(
            f"Retrieved policy evidence from {top_policy.get('source_file')} "
            f"section: {top_policy.get('section_title')}."
        )
    else:
        reasons.append("No relevant policy evidence was retrieved.")
        risk_flags.append("missing_policy_evidence")

    # Decision rules

    # Rule 1: High-risk customer should go to manual review
    if customer_claim_risk == "high":
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 2: High-value orders should be manually reviewed
    elif order_value_risk == "high":
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 3: Damaged item + replacement request + visual evidence + eligible order + policy support
    elif (
        issue_type == "damaged_item"
        and requested_action == "replacement"
        and damage_detected is True
        and return_window_status == "within_return_window"
        and policy_supports_replacement
    ):
        final_decision = "approve_replacement"
        confidence = "high"

    # Rule 4: Damaged item + refund request + visual evidence + eligible order + policy support
    elif (
        issue_type == "damaged_item"
        and requested_action == "refund"
        and damage_detected is True
        and return_window_status == "within_return_window"
        and policy_supports_refund
    ):
        final_decision = "approve_refund"
        confidence = "high"

    # Rule 5: Complaint says damaged, but visual evidence does not support it
    elif issue_type == "damaged_item" and damage_detected is False:
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 6: Missing item or not delivered cases
    elif issue_type == "missing_item" or delivery_issue == "not_delivered":
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 7: Late delivery refund case
    elif (
        issue_type == "late_delivery"
        and requested_action == "refund"
        and return_window_status == "within_return_window"
    ):
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 8: Outside return window needs policy check/manual review
    elif return_window_status == "outside_return_window":
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 9: If retrieved policy mentions verification/manual-review conditions
    elif policy_mentions_manual_review:
        final_decision = "manual_review"
        confidence = "medium"

    # Rule 10: If order agent says policy check is required
    elif order_hint == "policy_check_required":
        final_decision = "manual_review"
        confidence = "medium"

    # Default fallback
    else:
        final_decision = "manual_review"
        confidence = "low"

    # Escalation logic
    escalation_required = False

    if final_decision == "manual_review":
        escalation_required = True

    if urgency == "high" and final_decision == "manual_review":
        escalation_required = True

    if customer_claim_risk == "high":
        escalation_required = True

    if order_value_risk == "high":
        escalation_required = True

    # Customer-facing message
    if final_decision == "approve_replacement":
        customer_message = (
            "Your replacement request appears eligible based on the submitted visual evidence "
            "and the applicable replacement policy."
        )
    elif final_decision == "approve_refund":
        customer_message = (
            "Your refund request appears eligible based on the submitted visual evidence "
            "and the applicable refund or return policy."
        )
    else:
        customer_message = (
            "Your claim requires manual review because additional verification is needed before "
            "a final refund or replacement decision can be made."
        )

    # Final structured output
    return {
        "final_decision": final_decision,
        "confidence": confidence,
        "escalation_required": escalation_required,
        "customer_sentiment": sentiment,
        "urgency": urgency,
        "reason": " ".join(reasons),
        "customer_message": customer_message,
        "risk_flags": risk_flags,
        "policy_basis": policy_basis,
        "policy_filter_used": {
            "company_filter": policy_result.get("company_filter"),
            "policy_type_filter": policy_result.get("policy_type_filter")
        },
        "signals_used": {
            "issue_type": issue_type,
            "requested_action": requested_action,
            "return_window_status": return_window_status,
            "delivery_issue": delivery_issue,
            "order_value_risk": order_value_risk,
            "customer_claim_risk": customer_claim_risk,
            "damage_detected": damage_detected,
            "damage_type": damage_type,
            "damage_severity": damage_severity,
            "vision_confidence": vision_confidence,
            "vision_backend": vision_backend,
            "policy_supports_replacement": policy_supports_replacement,
            "policy_supports_refund": policy_supports_refund
        }
    }


if __name__ == "__main__":
    sample_complaint_result = {
        "issue_type": "damaged_item",
        "requested_action": "replacement",
        "sentiment": "negative",
        "urgency": "high",
        "summary": "Headphones arrived broken and the box was crushed."
    }

    sample_order_result = {
        "status": "success",
        "order_id": "ORD12345",
        "product_name": "Wireless Headphones",
        "return_window_status": "within_return_window",
        "delivery_issue": "delivered",
        "order_value_risk": "medium",
        "customer_claim_risk": "low",
        "recommended_action_hint": "eligible_for_resolution"
    }

    sample_vision_result = {
        "damage_detected": True,
        "damage_type": "broken_component",
        "damage_severity": "medium",
        "visible_evidence": "The earbud's metal component appears detached.",
        "confidence": "high",
        "model_backend": "groq_vision"
    }

    from src.policy_rag_agent import run_policy_rag_agent

    sample_policy_result = run_policy_rag_agent(
        complaint_result=sample_complaint_result,
        order_result=sample_order_result,
        vision_result=sample_vision_result,
        company="bestbuy",
        top_k=3
    )

    result = resolve_claim(
        complaint_result=sample_complaint_result,
        order_result=sample_order_result,
        vision_result=sample_vision_result,
        policy_result=sample_policy_result
    )

    print(json.dumps(result, indent=2))