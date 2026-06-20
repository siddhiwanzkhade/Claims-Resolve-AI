import json
import time
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


class ClaimState(TypedDict, total=False):
    # Raw inputs
    complaint_text: str
    order_data: Dict[str, Any]
    image_path: Optional[str]
    video_path: Optional[str]
    vision_backend: str
    company: str
    enable_logging: bool

    # Agent outputs
    complaint_analysis: Dict[str, Any]
    order_analysis: Dict[str, Any]
    vision_analysis: Dict[str, Any]
    policy_rag: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    final_resolution: Dict[str, Any]

    # Supervisor/control fields
    next_action: str
    supervisor_reason: str
    completed_steps: List[str]
    route: str

    # Validation
    input_error: bool
    input_error_type: str
    input_error_message: str

    # Metadata
    started_at: str
    completed_at: str
    runtime_seconds: float
    timings: Dict[str, float]


def add_completed_step(state: ClaimState, step_name: str) -> List[str]:
    completed = list(state.get("completed_steps", []))
    if step_name not in completed:
        completed.append(step_name)
    return completed


def has_image(state: ClaimState) -> bool:
    image_path = state.get("image_path")
    return image_path is not None and str(image_path).strip() != ""


def has_video(state: ClaimState) -> bool:
    video_path = state.get("video_path")
    return video_path is not None and str(video_path).strip() != ""


def update_timing(state: ClaimState, key: str, elapsed: float) -> Dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[key] = round(elapsed, 3)
    return timings


def get_requested_action(state: ClaimState) -> str:
    return state.get("complaint_analysis", {}).get("requested_action", "unknown")


def get_issue_type(state: ClaimState) -> str:
    return state.get("complaint_analysis", {}).get("issue_type", "unknown")


def get_risk_recommended_action(state: ClaimState) -> str:
    return state.get("risk_analysis", {}).get("recommended_action", "unknown")


# ---------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------

def supervisor_node(state: ClaimState) -> ClaimState:
    """
    Supervisor decides the next node based on current state.

    This is what makes the flow agentic:
    the system does not blindly execute every step;
    it routes based on validation, evidence, policy, and risk.
    """

    completed = state.get("completed_steps", [])

    if "input_validation" not in completed:
        return {
            "next_action": "input_validation",
            "supervisor_reason": "Validate claim inputs before running agents."
        }

    if state.get("input_error"):
        error_type = state.get("input_error_type")

        if error_type == "missing_visual_evidence":
            return {
                "next_action": "request_visual_evidence",
                "supervisor_reason": "Image or video evidence is required for automated visual claim review."
            }

        return {
            "next_action": "ask_for_more_info",
            "supervisor_reason": state.get("input_error_message", "Required claim information is missing.")
        }

    if "complaint_analysis" not in completed:
        return {
            "next_action": "complaint_analysis",
            "supervisor_reason": "Need to understand issue type, requested action, sentiment, and urgency."
        }

    if "order_analysis" not in completed:
        return {
            "next_action": "order_analysis",
            "supervisor_reason": "Need to validate return window, delivery status, order value, and claim history."
        }

    # Current automated visual path supports image.
    # Video-only claims are accepted as visual evidence but routed to manual review until video agent is added.
    if has_video(state) and not has_image(state):
        return {
            "next_action": "video_manual_review",
            "supervisor_reason": "Video evidence was provided, but automated video analysis is not enabled yet."
        }

    if "vision_analysis" not in completed:
        return {
            "next_action": "vision_analysis",
            "supervisor_reason": "Image evidence is available and should be analyzed by the Vision Agent."
        }

    if "policy_retrieval" not in completed:
        return {
            "next_action": "policy_retrieval",
            "supervisor_reason": "Need policy evidence from Pinecone before making a decision."
        }

    if "risk_scoring" not in completed:
        return {
            "next_action": "risk_scoring",
            "supervisor_reason": "Need risk score using complaint, order, vision, and policy signals."
        }

    if get_risk_recommended_action(state) == "manual_review":
        return {
            "next_action": "manual_review",
            "supervisor_reason": "Risk scoring recommends manual review."
        }

    policy_result = state.get("policy_rag", {})
    retrieved_evidence = policy_result.get("retrieved_evidence", [])

    if not retrieved_evidence:
        return {
            "next_action": "manual_review",
            "supervisor_reason": "No policy evidence was retrieved, so automated resolution is unsafe."
        }

    requested_action = get_requested_action(state)

    if requested_action == "replacement":
        return {
            "next_action": "replacement_resolution",
            "supervisor_reason": "Claim is low-risk and customer requested a replacement."
        }

    if requested_action == "refund":
        return {
            "next_action": "refund_resolution",
            "supervisor_reason": "Claim is low-risk and customer requested a refund."
        }

    return {
        "next_action": "general_resolution",
        "supervisor_reason": "Enough evidence is available for policy-grounded resolution."
    }


# ---------------------------------------------------------
# Validation node
# ---------------------------------------------------------

def input_validation_node(state: ClaimState) -> ClaimState:
    complaint_text = state.get("complaint_text", "")
    order_data = state.get("order_data", {})

    if not complaint_text or complaint_text.strip() == "":
        return {
            "input_error": True,
            "input_error_type": "missing_complaint",
            "input_error_message": "Customer complaint text is required.",
            "completed_steps": add_completed_step(state, "input_validation")
        }

    if not order_data:
        return {
            "input_error": True,
            "input_error_type": "missing_order_data",
            "input_error_message": "Order metadata is required.",
            "completed_steps": add_completed_step(state, "input_validation")
        }

    if not has_image(state) and not has_video(state):
        return {
            "input_error": True,
            "input_error_type": "missing_visual_evidence",
            "input_error_message": (
                "Visual evidence is required for automated claim review. "
                "Please upload an image or video of the product/package."
            ),
            "completed_steps": add_completed_step(state, "input_validation")
        }

    return {
        "input_error": False,
        "input_error_type": "",
        "input_error_message": "",
        "completed_steps": add_completed_step(state, "input_validation")
    }


# ---------------------------------------------------------
# Agent/tool nodes
# ---------------------------------------------------------

def complaint_analysis_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    complaint_result = analyze_complaint(state["complaint_text"])

    return {
        "complaint_analysis": complaint_result,
        "completed_steps": add_completed_step(state, "complaint_analysis"),
        "timings": update_timing(state, "complaint_agent_seconds", time.perf_counter() - start)
    }


def order_analysis_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    order_result = analyze_order(state["order_data"])

    return {
        "order_analysis": order_result,
        "completed_steps": add_completed_step(state, "order_analysis"),
        "timings": update_timing(state, "order_agent_seconds", time.perf_counter() - start)
    }


def vision_analysis_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    vision_result = analyze_product_image(
        image_path=state["image_path"],
        backend=state.get("vision_backend", "groq")
    )

    return {
        "vision_analysis": vision_result,
        "completed_steps": add_completed_step(state, "vision_analysis"),
        "timings": update_timing(state, "vision_agent_seconds", time.perf_counter() - start)
    }


def policy_retrieval_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    policy_result = run_policy_rag_agent(
        complaint_result=state["complaint_analysis"],
        order_result=state["order_analysis"],
        vision_result=state["vision_analysis"],
        company=state.get("company", "bestbuy"),
        top_k=3
    )

    return {
        "policy_rag": policy_result,
        "completed_steps": add_completed_step(state, "policy_retrieval"),
        "timings": update_timing(state, "policy_rag_seconds", time.perf_counter() - start)
    }


def risk_scoring_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    risk_result = score_claim_risk(
        complaint_result=state["complaint_analysis"],
        order_result=state["order_analysis"],
        vision_result=state["vision_analysis"],
        policy_result=state["policy_rag"]
    )

    return {
        "risk_analysis": risk_result,
        "completed_steps": add_completed_step(state, "risk_scoring"),
        "timings": update_timing(state, "risk_scoring_seconds", time.perf_counter() - start)
    }


# ---------------------------------------------------------
# Final decision nodes
# ---------------------------------------------------------

def resolve_with_optional_risk(state: ClaimState) -> Dict[str, Any]:
    """
    Calls resolve_claim.

    This supports both versions:
    1. resolve_claim(..., risk_result=...)
    2. resolve_claim(...) without risk_result
    """

    try:
        return resolve_claim(
            complaint_result=state["complaint_analysis"],
            order_result=state["order_analysis"],
            vision_result=state["vision_analysis"],
            policy_result=state["policy_rag"],
            risk_result=state.get("risk_analysis", {})
        )
    except TypeError:
        resolution = resolve_claim(
            complaint_result=state["complaint_analysis"],
            order_result=state["order_analysis"],
            vision_result=state["vision_analysis"],
            policy_result=state["policy_rag"]
        )
        resolution["risk_analysis"] = state.get("risk_analysis", {})
        return resolution


def replacement_resolution_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    final_resolution = resolve_with_optional_risk(state)
    final_resolution["resolution_path"] = "replacement_resolution"

    return {
        "route": "replacement_resolution",
        "final_resolution": final_resolution,
        "timings": update_timing(state, "resolution_agent_seconds", time.perf_counter() - start)
    }


def refund_resolution_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    final_resolution = resolve_with_optional_risk(state)
    final_resolution["resolution_path"] = "refund_resolution"

    return {
        "route": "refund_resolution",
        "final_resolution": final_resolution,
        "timings": update_timing(state, "resolution_agent_seconds", time.perf_counter() - start)
    }


def general_resolution_node(state: ClaimState) -> ClaimState:
    start = time.perf_counter()

    final_resolution = resolve_with_optional_risk(state)
    final_resolution["resolution_path"] = "general_resolution"

    return {
        "route": "general_resolution",
        "final_resolution": final_resolution,
        "timings": update_timing(state, "resolution_agent_seconds", time.perf_counter() - start)
    }


def manual_review_node(state: ClaimState) -> ClaimState:
    risk_result = state.get("risk_analysis", {})

    final_resolution = {
        "final_decision": "manual_review",
        "confidence": "medium",
        "escalation_required": True,
        "customer_message": (
            "Your claim requires manual review before a final refund or replacement decision can be made."
        ),
        "internal_reason": state.get("supervisor_reason", "Supervisor routed this claim to manual review."),
        "risk_score": risk_result.get("risk_score"),
        "risk_level": risk_result.get("risk_level"),
        "risk_flags": risk_result.get("risk_flags", []),
        "policy_basis": state.get("policy_rag", {}).get("retrieved_evidence", []),
        "next_step": "Human support specialist should review the claim."
    }

    return {
        "route": "manual_review",
        "final_resolution": final_resolution
    }


def request_visual_evidence_node(state: ClaimState) -> ClaimState:
    final_resolution = {
        "final_decision": "request_visual_evidence",
        "confidence": "high",
        "escalation_required": False,
        "customer_message": (
            "Visual evidence is required for automated claim review. "
            "Please upload an image or video showing the product, package, or reported issue."
        ),
        "internal_reason": state.get("supervisor_reason"),
        "next_step": "Collect image or video evidence from the customer."
    }

    return {
        "route": "request_visual_evidence",
        "final_resolution": final_resolution
    }


def ask_for_more_info_node(state: ClaimState) -> ClaimState:
    final_resolution = {
        "final_decision": "need_more_information",
        "confidence": "high",
        "escalation_required": False,
        "customer_message": state.get(
            "input_error_message",
            "More information is required to process this claim."
        ),
        "internal_reason": state.get("supervisor_reason"),
        "next_step": "Collect missing claim information."
    }

    return {
        "route": "ask_for_more_info",
        "final_resolution": final_resolution
    }


def video_manual_review_node(state: ClaimState) -> ClaimState:
    final_resolution = {
        "final_decision": "manual_review",
        "confidence": "medium",
        "escalation_required": True,
        "customer_message": (
            "Your video evidence has been received. This claim requires manual review because "
            "automated video analysis is not enabled in the current version."
        ),
        "internal_reason": state.get("supervisor_reason"),
        "next_step": "Human reviewer should inspect the submitted video evidence."
    }

    return {
        "route": "video_manual_review",
        "final_resolution": final_resolution
    }


def logging_node(state: ClaimState) -> ClaimState:
    completed_at = datetime.utcnow().isoformat()

    runtime_seconds = None
    if state.get("started_at"):
        started = datetime.fromisoformat(state["started_at"])
        completed = datetime.fromisoformat(completed_at)
        runtime_seconds = round((completed - started).total_seconds(), 3)

    timings = dict(state.get("timings", {}))
    timings["total_pipeline_seconds"] = runtime_seconds

    logged_result = {
        **state,
        "completed_at": completed_at,
        "runtime_seconds": runtime_seconds,
        "timings": timings
    }

    if state.get("enable_logging", True):
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(logged_result, default=str) + "\n")

    return {
        "completed_at": completed_at,
        "runtime_seconds": runtime_seconds,
        "timings": timings
    }


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

def route_from_supervisor(
    state: ClaimState
) -> Literal[
    "input_validation",
    "complaint_analysis",
    "order_analysis",
    "vision_analysis",
    "policy_retrieval",
    "risk_scoring",
    "manual_review",
    "request_visual_evidence",
    "ask_for_more_info",
    "video_manual_review",
    "replacement_resolution",
    "refund_resolution",
    "general_resolution"
]:
    return state["next_action"]


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

def build_claim_resolution_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("supervisor", supervisor_node)

    graph.add_node("input_validation", input_validation_node)
    graph.add_node("complaint_analysis", complaint_analysis_node)
    graph.add_node("order_analysis", order_analysis_node)
    graph.add_node("vision_analysis", vision_analysis_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("risk_scoring", risk_scoring_node)

    graph.add_node("manual_review", manual_review_node)
    graph.add_node("request_visual_evidence", request_visual_evidence_node)
    graph.add_node("ask_for_more_info", ask_for_more_info_node)
    graph.add_node("video_manual_review", video_manual_review_node)
    graph.add_node("replacement_resolution", replacement_resolution_node)
    graph.add_node("refund_resolution", refund_resolution_node)
    graph.add_node("general_resolution", general_resolution_node)

    graph.add_node("logging", logging_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "input_validation": "input_validation",
            "complaint_analysis": "complaint_analysis",
            "order_analysis": "order_analysis",
            "vision_analysis": "vision_analysis",
            "policy_retrieval": "policy_retrieval",
            "risk_scoring": "risk_scoring",
            "manual_review": "manual_review",
            "request_visual_evidence": "request_visual_evidence",
            "ask_for_more_info": "ask_for_more_info",
            "video_manual_review": "video_manual_review",
            "replacement_resolution": "replacement_resolution",
            "refund_resolution": "refund_resolution",
            "general_resolution": "general_resolution"
        }
    )

    graph.add_edge("input_validation", "supervisor")
    graph.add_edge("complaint_analysis", "supervisor")
    graph.add_edge("order_analysis", "supervisor")
    graph.add_edge("vision_analysis", "supervisor")
    graph.add_edge("policy_retrieval", "supervisor")
    graph.add_edge("risk_scoring", "supervisor")

    graph.add_edge("manual_review", "logging")
    graph.add_edge("request_visual_evidence", "logging")
    graph.add_edge("ask_for_more_info", "logging")
    graph.add_edge("video_manual_review", "logging")
    graph.add_edge("replacement_resolution", "logging")
    graph.add_edge("refund_resolution", "logging")
    graph.add_edge("general_resolution", "logging")

    graph.add_edge("logging", END)

    return graph.compile()


claim_resolution_graph = build_claim_resolution_graph()


# ---------------------------------------------------------
# Public function for app.py
# ---------------------------------------------------------

def run_claim_resolution(
    complaint_text: str,
    order_data: dict,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    vision_backend: str = "groq",
    company: str = "bestbuy",
    enable_logging: bool = True
) -> dict:
    initial_state: ClaimState = {
        "complaint_text": complaint_text,
        "order_data": order_data,
        "image_path": image_path,
        "video_path": video_path,
        "vision_backend": vision_backend,
        "company": company,
        "enable_logging": enable_logging,
        "completed_steps": [],
        "started_at": datetime.utcnow().isoformat(),
        "timings": {}
    }

    return claim_resolution_graph.invoke(initial_state)


if __name__ == "__main__":
    sample_complaint_text = (
        "My headphones arrived broken and the box was crushed. "
        "I want a replacement."
    )

    sample_order_data = {
        "company": "bestbuy",
        "order_id": "ORD12345",
        "product_name": "Wireless Headphones",
        "order_value": 149.99,
        "delivery_status": "delivered",
        "delivery_date": "2026-05-25",
        "return_deadline": "2026-06-24",
        "customer_previous_claims": 1
    }

    sample_image_path = (
        "/Users/siddhiwanzkhade/E-commerce Claim AI/"
        "data/policy/images/test_damage.jpg"
    )

    result = run_claim_resolution(
        complaint_text=sample_complaint_text,
        order_data=sample_order_data,
        image_path=sample_image_path,
        vision_backend="groq",
        company="bestbuy"
    )

    print(json.dumps(result, indent=2, default=str))