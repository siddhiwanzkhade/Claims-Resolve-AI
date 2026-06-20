# Defines what valid agent outputs should look like

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ComplaintAnalysis(BaseModel):
    issue_type: Literal[
        "damaged_item",
        "defective_item",
        "package_damage",
        "late_delivery",
        "missing_item",
        "wrong_item",
        "refund_request",
        "unclear"
    ]

    requested_action: Literal[
        "refund",
        "replacement",
        "return",
        "escalation",
        "unclear"
    ]

    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]
    summary: str


class OrderAnalysis(BaseModel):
    status: Literal["success", "error"] = "success"
    order_id: Optional[str] = None
    product_name: Optional[str] = None

    return_window_status: Literal[
        "within_return_window",
        "outside_return_window",
        "unknown"
    ] = "unknown"

    delivery_issue: Literal[
        "delivered",
        "in_transit",
        "delayed",
        "not_delivered",
        "unknown"
    ] = "unknown"

    order_value_risk: Literal[
        "low",
        "medium",
        "high",
        "unknown"
    ] = "unknown"

    customer_claim_risk: Literal[
        "low",
        "medium",
        "high",
        "unknown"
    ] = "unknown"

    recommended_action_hint: Optional[str] = "policy_check_required"
    message: Optional[str] = None


class VisionAnalysis(BaseModel):
    damage_detected: bool
    damage_type: str = "unknown"

    damage_severity: Literal[
        "low",
        "medium",
        "high",
        "unknown"
    ] = "unknown"

    visible_evidence: str = ""
    confidence: Literal[
        "low",
        "medium",
        "high",
        "unknown"
    ] = "unknown"

    model_backend: str = "unknown"


class PolicyEvidenceChunk(BaseModel):
    source_file: Optional[str] = None
    company: Optional[str] = None
    policy_type: Optional[str] = None
    section_title: Optional[str] = None
    chunk_id: Optional[int] = None
    content: str


class PolicyRAGResult(BaseModel):
    query: str
    company_filter: Optional[str] = None
    policy_type_filter: Optional[str] = None
    retrieved_evidence: List[PolicyEvidenceChunk] = Field(default_factory=list)


class RiskAnalysis(BaseModel):
    risk_score: int = Field(ge=0, le=100)

    risk_level: Literal[
        "low",
        "low_medium",
        "medium",
        "high"
    ]

    risk_flags: List[str] = Field(default_factory=list)

    recommended_action: Literal[
        "manual_review",
        "proceed_with_policy_grounded_resolution"
    ]

    explanations: List[str] = Field(default_factory=list)
    signals_used: Dict[str, Any] = Field(default_factory=dict)


class FinalResolution(BaseModel):
    final_decision: Literal[
        "approve_refund",
        "approve_replacement",
        "manual_review",
        "request_visual_evidence",
        "need_more_information"
    ]

    confidence: Literal["low", "medium", "high"]
    escalation_required: bool
    customer_message: str

    reason: Optional[str] = None
    customer_sentiment: Optional[str] = None
    urgency: Optional[str] = None

    risk_flags: List[str] = Field(default_factory=list)
    policy_basis: List[PolicyEvidenceChunk] = Field(default_factory=list)
    policy_filter_used: Dict[str, Any] = Field(default_factory=dict)
    signals_used: Dict[str, Any] = Field(default_factory=dict)
    risk_analysis: Optional[Dict[str, Any]] = None
    resolution_path: Optional[str] = None


class ClaimStateSchema(BaseModel):
    complaint_text: str
    order_data: Dict[str, Any]

    image_path: Optional[str] = None
    video_path: Optional[str] = None
    vision_backend: str = "groq"
    company: str = "bestbuy"
    enable_logging: bool = True

    complaint_analysis: Optional[ComplaintAnalysis] = None
    order_analysis: Optional[OrderAnalysis] = None
    vision_analysis: Optional[VisionAnalysis] = None
    policy_rag: Optional[PolicyRAGResult] = None
    risk_analysis: Optional[RiskAnalysis] = None
    final_resolution: Optional[FinalResolution] = None

    next_action: Optional[str] = None
    supervisor_reason: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    route: Optional[str] = None

    input_error: bool = False
    input_error_type: str = ""
    input_error_message: str = ""

    timings: Dict[str, float] = Field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    runtime_seconds: Optional[float] = None