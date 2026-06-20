import json


def score_claim_risk(
    complaint_result: dict,
    order_result: dict,
    vision_result: dict,
    policy_result: dict
) -> dict:
    """
    Computes claim risk using complaint, order, vision, and policy signals.

    This agent does not approve or reject claims.
    It only produces risk signals that the Resolution Agent can use.
    """

    issue_type = complaint_result.get("issue_type", "unclear")
    requested_action = complaint_result.get("requested_action", "unclear")
    urgency = complaint_result.get("urgency", "low")

    return_window_status = order_result.get("return_window_status", "unknown")
    order_value_risk = order_result.get("order_value_risk", "unknown")
    customer_claim_risk = order_result.get("customer_claim_risk", "unknown")
    delivery_issue = order_result.get("delivery_issue", "unknown")

    damage_detected = vision_result.get("damage_detected", False)
    damage_severity = vision_result.get("damage_severity", "unknown")
    vision_confidence = vision_result.get("confidence", "unknown")
    visible_evidence = vision_result.get("visible_evidence", "")

    retrieved_evidence = policy_result.get("retrieved_evidence", [])
    policy_type_filter = policy_result.get("policy_type_filter")

    risk_score = 0
    risk_flags = []
    explanations = []

  
    # Order risk
  
    if order_value_risk == "high":
        risk_score += 25
        risk_flags.append("high_order_value")
        explanations.append("Order value is high, so manual verification is recommended.")

    elif order_value_risk == "medium":
        risk_score += 10
        risk_flags.append("medium_order_value")
        explanations.append("Order value is medium.")

    if customer_claim_risk == "high":
        risk_score += 30
        risk_flags.append("high_customer_claim_risk")
        explanations.append("Customer has a high previous-claim risk signal.")

    elif customer_claim_risk == "medium":
        risk_score += 15
        risk_flags.append("medium_customer_claim_risk")
        explanations.append("Customer has a medium previous-claim risk signal.")

    if return_window_status == "outside_return_window":
        risk_score += 30
        risk_flags.append("outside_return_window")
        explanations.append("Order appears to be outside the return/replacement window.")

    elif return_window_status == "unknown":
        risk_score += 10
        risk_flags.append("unknown_return_window")
        explanations.append("Return window status is unknown.")


    # Vision/evidence risk
  
    if issue_type == "damaged_item" and damage_detected is False:
        risk_score += 35
        risk_flags.append("visual_claim_mismatch")
        explanations.append(
            "Complaint reports damage, but visual evidence does not clearly confirm damage."
        )

    if vision_confidence == "low":
        risk_score += 20
        risk_flags.append("low_visual_confidence")
        explanations.append("Vision model confidence is low.")

    elif vision_confidence == "medium":
        risk_score += 8
        risk_flags.append("medium_visual_confidence")
        explanations.append("Vision model confidence is medium.")

    if not visible_evidence:
        risk_score += 15
        risk_flags.append("missing_visible_evidence_description")
        explanations.append("Vision agent did not provide a clear visible evidence description.")

    if damage_severity == "high" and order_value_risk == "high":
        risk_score += 10
        risk_flags.append("high_damage_high_value")
        explanations.append("High damage severity on a high-value order requires stronger verification.")


    # Delivery / claim-type risk
    
    if issue_type == "missing_item" or delivery_issue == "not_delivered":
        risk_score += 25
        risk_flags.append("missing_or_not_delivered_claim")
        explanations.append(
            "Missing item or not-delivered claims usually require carrier/order verification."
        )

    if requested_action == "refund" and customer_claim_risk in ["medium", "high"]:
        risk_score += 10
        risk_flags.append("refund_request_with_claim_history")
        explanations.append("Refund request combined with previous claim risk needs additional review.")

    if urgency == "high" and customer_claim_risk == "high":
        risk_score += 10
        risk_flags.append("high_urgency_high_claim_risk")
        explanations.append("High urgency combined with high claim risk increases review priority.")

  
    # Policy/RAG risk

    if not retrieved_evidence:
        risk_score += 30
        risk_flags.append("missing_policy_evidence")
        explanations.append("No policy evidence was retrieved for this claim.")

    if policy_type_filter is None:
        risk_score += 8
        risk_flags.append("unclear_policy_type")
        explanations.append("The system could not infer a narrow policy type for retrieval.")

    
    # Normalize score
    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "high"
        recommended_action = "manual_review"
    elif risk_score >= 40:
        risk_level = "medium"
        recommended_action = "manual_review"
    elif risk_score >= 20:
        risk_level = "low_medium"
        recommended_action = "proceed_with_policy_grounded_resolution"
    else:
        risk_level = "low"
        recommended_action = "proceed_with_policy_grounded_resolution"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "recommended_action": recommended_action,
        "explanations": explanations,
        "signals_used": {
            "issue_type": issue_type,
            "requested_action": requested_action,
            "urgency": urgency,
            "return_window_status": return_window_status,
            "order_value_risk": order_value_risk,
            "customer_claim_risk": customer_claim_risk,
            "delivery_issue": delivery_issue,
            "damage_detected": damage_detected,
            "damage_severity": damage_severity,
            "vision_confidence": vision_confidence,
            "policy_type_filter": policy_type_filter,
            "policy_evidence_count": len(retrieved_evidence)
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
        "return_window_status": "within_return_window",
        "delivery_issue": "delivered",
        "order_value_risk": "medium",
        "customer_claim_risk": "low"
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

    result = score_claim_risk(
        complaint_result=sample_complaint_result,
        order_result=sample_order_result,
        vision_result=sample_vision_result,
        policy_result=sample_policy_result
    )

    print(json.dumps(result, indent=2))