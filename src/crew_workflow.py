import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from src.complaint_agent import analyze_complaint
from src.order_agent import analyze_order
from src.vision_agent import analyze_product_image
from src.policy_rag_agent import run_policy_rag_agent
from src.risk_scoring import score_claim_risk
from src.resolution_agent import resolve_claim


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "claim_runs.jsonl"


# ---------------------------------------------------------
# Shared state for the agentic graph
# ---------------------------------------------------------

class ClaimState(TypedDict, total=False):
    # Raw user inputs
    complaint_text: str
    order_data: Dict[str, Any]
    image_path: Optional[str]
    company: str

    # Outputs produced by agents/tools
    complaint_result: Dict[str, Any]
    order_result: Dict[str, Any]
    vision_result: Dict[str, Any]
    policy_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    final_resolution: Dict[str, Any]

    # Supervisor / graph control
    next_action: str
    supervisor_reason: str
    completed_steps: List[str]
    route: str

    # Metadata
    started_at: str
    completed_at: str
    runtime_seconds: float


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def add_completed_step(state: ClaimState, step_name: str) -> List[str]:
    completed = state.get("completed_steps", [])
    if step_name not in completed:
        completed.append(step_name)
    return completed


def has_image(state: ClaimState) -> bool:
    image_path = state.get("image_path")
    return image_path is not None and str(image_path).strip() != ""


def get_issue_type(state: ClaimState) -> str:
    return state.get("complaint_result", {}).get("issue_type", "unknown")


def get_requested_action(state: ClaimState) -> str:
    return state.get("complaint_result", {}).get("requested_action", "unknown")


def get_risk_level(state: ClaimState) -> str:
    return state.get("risk_result", {}).get("risk_level", "unknown")


def get_risk_recommended_action(state: ClaimState) -> str:
    return state.get("risk_result", {}).get("recommended_action", "unknown")


# ---------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------

def supervisor_node(state: ClaimState) -> ClaimState:
    """
    The supervisor decides the next agent/tool to call.

    This makes the system agentic:
    - It does not blindly run every step.
    - It checks what evidence exists.
    - It chooses the next best action.
    """

    completed = state.get("completed_steps", [])

    # Step 1: Always understand the complaint first.
    if "complaint_analysis" not in completed:
        return {
            "next_action": "complaint_analysis",
            "supervisor_reason": "Need to understand the claim issue type and requested action first."
        }

    issue_type = get_issue_type(state)
    requested_action = get_requested_action(state)

    # Step 2: If order data is missing, ask for clarification.
    if not state.get("order_data"):
        return {
            "next_action": "ask_for_more_info",
            "supervisor_reason": "Order data is missing, so the system cannot validate eligibility."
        }

    # Step 3: Validate order information.
    if "order_analysis" not in completed:
        return {
            "next_action": "order_analysis",
            "supervisor_reason": "Need to validate delivery status, return window, order value, and customer claim history."
        }

    # Step 4: Use vision only when image evidence is useful.
    # Damaged item claims benefit from image inspection.
    if (
        issue_type in ["damaged_item", "defective_item", "wrong_item"]
        and has_image(state)
        and "vision_analysis" not in completed
    ):
        return {
            "next_action": "vision_analysis",
            "supervisor_reason": "Claim type requires visual evidence, and an image was provided."
        }

    # Step 5: If damaged item claim has no image, still create a vision placeholder.
    # This helps risk scoring understand missing evidence.
    if (
        issue_type in ["damaged_item", "defective_item", "wrong_item"]
        and not has_image(state)
        and "vision_analysis" not in completed
    ):
        return {
            "next_action": "vision_analysis",
            "supervisor_reason": "Claim type requires visual evidence, but no image was provided. Need to record missing evidence."
        }

    # Step 6: For missing package claims, image is usually not needed.
    # Go directly to policy retrieval after order analysis.
    if "policy_retrieval" not in completed:
        return {
            "next_action": "policy_retrieval",
            "supervisor_reason": "Need relevant company policy evidence before deciding the claim."
        }

    # Step 7: Score risk after enough evidence is available.
    if "risk_scoring" not in completed:
        return {
            "next_action": "risk_scoring",
            "supervisor_reason": "Need risk score using complaint, order, vision, and policy signals."
        }

    # Step 8: High or medium risk goes to manual review.
    if get_risk_recommended_action(state) == "manual_review":
        return {
            "next_action": "manual_review",
            "supervisor_reason": "Risk scoring recommends manual review."
        }

    # Step 9: If policy evidence is missing, ask for manual review.
    policy_result = state.get("policy_result", {})
    retrieved_evidence = policy_result.get("retrieved_evidence", [])

    if not retrieved_evidence:
        return {
            "next_action": "manual_review",
            "supervisor_reason": "No policy evidence was retrieved, so automated resolution is unsafe."
        }

    # Step 10: Decide final resolution path.
    if requested_action == "refund":
        return {
            "next_action": "refund_resolution",
            "supervisor_reason": "Claim is low-risk and customer requested a refund."
        }

    if requested_action == "replacement":
        return {
            "next_action": "replacement_resolution",
            "supervisor_reason": "Claim is low-risk and customer requested a replacement."
        }

    if issue_type in ["missing_item", "not_delivered"]:
        return {
            "next_action": "delivery_investigation",
            "supervisor_reason": "Claim is related to missing item or delivery failure."
        }

    return {
        "next_action": "general_resolution",
        "supervisor_reason": "Enough evidence is available for policy-grounded resolution."
    }


# ---------------------------------------------------------
# Agent/tool nodes
# ---------------------------------------------------------

def complaint_analysis_node(state: ClaimState) -> ClaimState:
    complaint_result = analyze_complaint(state["complaint_text"])

    return {
        "complaint_result": complaint_result,
        "completed_steps": add_completed_step(state, "complaint_analysis")
    }


def order_analysis_node(state: ClaimState) -> ClaimState:
    order_result = analyze_order(state["order_data"])

    return {
        "order_result": order_result,
        "completed_steps": add_completed_step(state, "order_analysis")
    }


def vision_analysis_node(state: ClaimState) -> ClaimState:
    image_path = state.get("image_path")

    if not image_path:
        vision_result = {
            "damage_detected": False,
            "damage_type": "unknown",
            "damage_severity": "unknown",
            "visible_evidence": "",
            "confidence": "low",
            "model_backend": "none",
            "note": "No image was provided for visual inspection."
        }
    else:
        vision_result = analyze_product_image(image_path)

    return {
        "vision_result": vision_result,
        "completed_steps": add_completed_step(state, "vision_analysis")
    }


def policy_retrieval_node(state: ClaimState) -> ClaimState:
    # If vision did not run because not needed, give safe default vision_result.
    vision_result = state.get("vision_result", {
        "damage_detected": False,
        "damage_severity": "not_applicable",
        "confidence": "not_applicable",
        "visible_evidence": "Vision analysis was not required for this claim type."
    })

    policy_result = run_policy_rag_agent(
        complaint_result=state["complaint_result"],
        order_result=state["order_result"],
        vision_result=vision_result,
        company=state.get("company", "bestbuy"),
        top_k=3
    )

    return {
        "vision_result": vision_result,
        "policy_result": policy_result,
        "completed_steps": add_completed_step(state, "policy_retrieval")
    }


def risk_scoring_node(state: ClaimState) -> ClaimState:
    vision_result = state.get("vision_result", {
        "damage_detected": False,
        "damage_severity": "not_applicable",
        "confidence": "not_applicable",
        "visible_evidence": "Vision analysis was not required for this claim type."
    })

    risk_result = score_claim_risk(
        complaint_result=state["complaint_result"],
        order_result=state["order_result"],
        vision_result=vision_result,
        policy_result=state["policy_result"]
    )

    return {
        "vision_result": vision_result,
        "risk_result": risk_result,
        "completed_steps": add_completed_step(state, "risk_scoring")
    }


# ---------------------------------------------------------
# Final decision nodes
# ---------------------------------------------------------

def manual_review_node(state: ClaimState) -> ClaimState:
    risk_result = state.get("risk_result", {})

    final_resolution = {
        "decision": "manual_review",
        "customer_message": (
            "Your claim requires additional review before a final decision can be made. "
            "A support specialist will verify the order details, policy eligibility, and submitted evidence."
        ),
        "internal_reason": state.get("supervisor_reason", "Supervisor routed this claim to manual review."),
        "risk_score": risk_result.get("risk_score"),
        "risk_level": risk_result.get("risk_level"),
        "risk_flags": risk_result.get("risk_flags", []),
        "next_step": "Human support specialist should review the claim."
    }

    return {
        "route": "manual_review",
        "final_resolution": final_resolution
    }


def ask_for_more_info_node(state: ClaimState) -> ClaimState:
    final_resolution = {
        "decision": "need_more_information",
        "customer_message": (
            "We need more information to process your claim. "
            "Please provide your order details, including order ID, delivery status, and purchase date."
        ),
        "internal_reason": state.get("supervisor_reason", "Missing required information."),
        "next_step": "Collect missing order information from the customer."
    }

    return {
        "route": "ask_for_more_info",
        "final_resolution": final_resolution
    }


def refund_resolution_node(state: ClaimState) -> ClaimState:
    final_resolution = resolve_claim(
        complaint_result=state["complaint_result"],
        order_result=state["order_result"],
        vision_result=state.get("vision_result", {}),
        policy_result=state["policy_result"],
        risk_result=state["risk_result"]
    )

    # Add explicit route label for readability.
    final_resolution["resolution_path"] = "refund_resolution"

    return {
        "route": "refund_resolution",
        "final_resolution": final_resolution
    }


def replacement_resolution_node(state: ClaimState) -> ClaimState:
    final_resolution = resolve_claim(
        complaint_result=state["complaint_result"],
        order_result=state["order_result"],
        vision_result=state.get("vision_result", {}),
        policy_result=state["policy_result"],
        risk_result=state["risk_result"]
    )

    final_resolution["resolution_path"] = "replacement_resolution"

    return {
        "route": "replacement_resolution",
        "final_resolution": final_resolution
    }


def delivery_investigation_node(state: ClaimState) -> ClaimState:
    final_resolution = {
        "decision": "delivery_investigation",
        "customer_message": (
            "Your claim appears to involve a missing item or delivery issue. "
            "We need to verify delivery and carrier information before issuing a refund or replacement."
        ),
        "internal_reason": state.get("supervisor_reason", "Supervisor routed this claim to delivery investigation."),
        "policy_evidence": state.get("policy_result", {}).get("retrieved_evidence", []),
        "risk_result": state.get("risk_result", {}),
        "next_step": "Verify carrier/order delivery records."
    }

    return {
        "route": "delivery_investigation",
        "final_resolution": final_resolution
    }


def general_resolution_node(state: ClaimState) -> ClaimState:
    final_resolution = resolve_claim(
        complaint_result=state["complaint_result"],
        order_result=state["order_result"],
        vision_result=state.get("vision_result", {}),
        policy_result=state["policy_result"],
        risk_result=state["risk_result"]
    )

    final_resolution["resolution_path"] = "general_resolution"

    return {
        "route": "general_resolution",
        "final_resolution": final_resolution
    }


def logging_node(state: ClaimState) -> ClaimState:
    completed_at = datetime.utcnow().isoformat()

    runtime_seconds = None
    if state.get("started_at"):
        started = datetime.fromisoformat(state["started_at"])
        completed = datetime.fromisoformat(completed_at)
        runtime_seconds = (completed - started).total_seconds()

    logged_result = {
        **state,
        "completed_at": completed_at,
        "runtime_seconds": runtime_seconds
    }

    LOG_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(logged_result, default=str) + "\n")

    return {
        "completed_at": completed_at,
        "runtime_seconds": runtime_seconds
    }


# ---------------------------------------------------------
# Router function
# ---------------------------------------------------------

def route_from_supervisor(
    state: ClaimState
) -> Literal[
    "complaint_analysis",
    "order_analysis",
    "vision_analysis",
    "policy_retrieval",
    "risk_scoring",
    "manual_review",
    "ask_for_more_info",
    "refund_resolution",
    "replacement_resolution",
    "delivery_investigation",
    "general_resolution"
]:
    return state["next_action"]


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_claim_resolution_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("supervisor", supervisor_node)

    graph.add_node("complaint_analysis", complaint_analysis_node)
    graph.add_node("order_analysis", order_analysis_node)
    graph.add_node("vision_analysis", vision_analysis_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("risk_scoring", risk_scoring_node)

    graph.add_node("manual_review", manual_review_node)
    graph.add_node("ask_for_more_info", ask_for_more_info_node)
    graph.add_node("refund_resolution", refund_resolution_node)
    graph.add_node("replacement_resolution", replacement_resolution_node)
    graph.add_node("delivery_investigation", delivery_investigation_node)
    graph.add_node("general_resolution", general_resolution_node)

    graph.add_node("logging", logging_node)

    # Start with supervisor, not a fixed first agent.
    graph.add_edge(START, "supervisor")

    # Supervisor decides which node to call next.
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "complaint_analysis": "complaint_analysis",
            "order_analysis": "order_analysis",
            "vision_analysis": "vision_analysis",
            "policy_retrieval": "policy_retrieval",
            "risk_scoring": "risk_scoring",
            "manual_review": "manual_review",
            "ask_for_more_info": "ask_for_more_info",
            "refund_resolution": "refund_resolution",
            "replacement_resolution": "replacement_resolution",
            "delivery_investigation": "delivery_investigation",
            "general_resolution": "general_resolution"
        }
    )

    # Specialist agents/tools return control back to supervisor.
    graph.add_edge("complaint_analysis", "supervisor")
    graph.add_edge("order_analysis", "supervisor")
    graph.add_edge("vision_analysis", "supervisor")
    graph.add_edge("policy_retrieval", "supervisor")
    graph.add_edge("risk_scoring", "supervisor")

    # Final nodes go to logging, then end.
    graph.add_edge("manual_review", "logging")
    graph.add_edge("ask_for_more_info", "logging")
    graph.add_edge("refund_resolution", "logging")
    graph.add_edge("replacement_resolution", "logging")
    graph.add_edge("delivery_investigation", "logging")
    graph.add_edge("general_resolution", "logging")

    graph.add_edge("logging", END)

    return graph.compile()


claim_resolution_graph = build_claim_resolution_graph()


# ---------------------------------------------------------
# Public function used by app.py / Gradio
# ---------------------------------------------------------

def run_claim_resolution(
    complaint_text: str,
    order_data: dict,
    image_path: Optional[str] = None,
    company: str = "bestbuy"
) -> dict:
    initial_state: ClaimState = {
        "complaint_text": complaint_text,
        "order_data": order_data,
        "image_path": image_path,
        "company": company,
        "completed_steps": [],
        "started_at": datetime.utcnow().isoformat()
    }

    result = claim_resolution_graph.invoke(initial_state)

    return result


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":
    sample_complaint_text = (
        "My headphones arrived broken and the box was crushed. "
        "I want a replacement."
    )

    sample_order_data = {
        "order_id": "ORD-1001",
        "customer_id": "CUST-501",
        "order_value": 149.99,
        "delivery_status": "delivered",
        "days_since_delivery": 5,
        "previous_claims": 0
    }

    result = run_claim_resolution(
        complaint_text=sample_complaint_text,
        order_data=sample_order_data,
        image_path=None,
        company="bestbuy"
    )

    print(json.dumps(result, indent=2, default=str))