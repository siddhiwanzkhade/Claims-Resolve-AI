import json
import gradio as gr

from src.claim_resolution_flow import run_claim_resolution


def build_user_friendly_summary(result: dict) -> str:
    """
    Builds a readable dashboard-style summary for non-technical users.
    """

    final_resolution = result.get("final_resolution", {})
    risk_analysis = result.get("risk_analysis", {})
    policy_rag = result.get("policy_rag", {})
    vision_analysis = result.get("vision_analysis", {})
    timings = result.get("timings", {})

    decision = final_resolution.get("final_decision", "unknown")
    customer_message = final_resolution.get(
        "customer_message",
        "No customer message generated."
    )
    confidence = final_resolution.get("confidence", "unknown")
    escalation_required = final_resolution.get("escalation_required", "unknown")

    risk_score = risk_analysis.get("risk_score", "N/A")
    risk_level = risk_analysis.get("risk_level", "N/A")
    risk_flags = risk_analysis.get("risk_flags", [])

    route = result.get("route", "unknown")
    supervisor_reason = result.get(
        "supervisor_reason",
        "No supervisor reason available."
    )

    damage_detected = vision_analysis.get("damage_detected", "unknown")
    damage_type = vision_analysis.get("damage_type", "unknown")
    damage_severity = vision_analysis.get("damage_severity", "unknown")
    visible_evidence = vision_analysis.get(
        "visible_evidence",
        "No visual evidence summary available."
    )

    retrieved_evidence = policy_rag.get("retrieved_evidence", [])
    top_policy = retrieved_evidence[0] if retrieved_evidence else {}

    policy_section = top_policy.get("section_title", "No policy section retrieved.")
    policy_source = top_policy.get("source_file", "N/A")
    policy_type = top_policy.get("policy_type", "N/A")

    runtime = timings.get(
        "total_pipeline_seconds",
        result.get("runtime_seconds", "N/A")
    )

    completed_steps = result.get("completed_steps", [])
    completed_steps_text = " → ".join(completed_steps) if completed_steps else "N/A"

    risk_flags_text = ", ".join(risk_flags) if risk_flags else "None"

    # Make the decision visually easier to scan.
    if decision in ["approve_replacement", "approve_refund"]:
        decision_label = f"✅ `{decision}`"
    elif decision in ["manual_review", "request_visual_evidence", "need_more_information"]:
        decision_label = f"⚠️ `{decision}`"
    else:
        decision_label = f"`{decision}`"

    return f"""
### Final Decision: {decision_label}

**Customer Message:**  
{customer_message}

---

### Why this decision was made

- **Supervisor route:** `{route}`
- **Supervisor reason:** {supervisor_reason}
- **Confidence:** `{confidence}`
- **Escalation required:** `{escalation_required}`

---

### Visual Evidence Summary

- **Damage detected:** `{damage_detected}`
- **Damage type:** `{damage_type}`
- **Damage severity:** `{damage_severity}`
- **Visual evidence:** {visible_evidence}

---

### Policy Evidence

- **Policy source:** `{policy_source}`
- **Policy type:** `{policy_type}`
- **Policy section:** {policy_section}

---

### Risk Summary

- **Risk score:** `{risk_score}/100`
- **Risk level:** `{risk_level}`
- **Risk flags:** {risk_flags_text}

---

### Runtime and Graph Trace

- **Total processing time:** `{runtime}` seconds
- **Completed steps:** {completed_steps_text}
"""


def build_error_summary(message: str) -> str:
    """
    Builds readable error message for dashboard.
    """

    return f"""
## Claim Decision Dashboard


### Final Decision: ⚠️ `error`

**Message:**  
{message}

Please check the required inputs and try again.
"""


def run_app(
    complaint_text,
    product_image,
    vision_backend,
    company,
    order_id,
    product_name,
    order_value,
    delivery_status,
    delivery_date,
    return_deadline,
    customer_previous_claims
):
    """
    Runs the full E-commerce ClaimAI workflow from the Gradio UI.

    Flow:
    Complaint + Order Metadata + Visual Evidence
        ↓
    claim_resolution_flow.py
        ↓
    LangGraph Supervisor
        ↓
    Input Validation
    Complaint Agent
    Order Agent
    Vision Agent
    Policy RAG Agent
    Risk Scoring Agent
    Resolution Agent / Manual Review
        ↓
    Human-readable dashboard + JSON audit outputs
    """

    try:
        order_data = {
            "company": company,
            "order_id": order_id,
            "product_name": product_name,
            "order_value": order_value,
            "delivery_status": delivery_status,
            "delivery_date": delivery_date,
            "return_deadline": return_deadline,
            "customer_previous_claims": customer_previous_claims
        }

        result = run_claim_resolution(
            complaint_text=complaint_text,
            order_data=order_data,
            image_path=product_image,
            video_path=None,
            vision_backend=vision_backend,
            company=company,
            enable_logging=True
        )

        complaint_result = result.get("complaint_analysis", {})
        order_result = result.get("order_analysis", {})
        vision_result = result.get("vision_analysis", {})
        policy_result = result.get("policy_rag", {})
        risk_result = result.get("risk_analysis", {})
        resolution_result = result.get("final_resolution", {})
        timings_result = result.get("timings", {})

        graph_metadata = {
            "route": result.get("route"),
            "next_action": result.get("next_action"),
            "supervisor_reason": result.get("supervisor_reason"),
            "completed_steps": result.get("completed_steps", []),
            "input_error": result.get("input_error", False),
            "input_error_type": result.get("input_error_type", ""),
            "input_error_message": result.get("input_error_message", ""),
            "runtime_seconds": result.get("runtime_seconds")
        }

        decision_summary = build_user_friendly_summary(result)

        return (
            decision_summary,
            json.dumps(complaint_result, indent=2),
            json.dumps(order_result, indent=2),
            json.dumps(vision_result, indent=2),
            json.dumps(policy_result, indent=2),
            json.dumps(risk_result, indent=2),
            json.dumps(resolution_result, indent=2),
            json.dumps(timings_result, indent=2),
            json.dumps(graph_metadata, indent=2),
            product_image
        )

    except Exception as e:
        error_message = str(e)
        error_summary = build_error_summary(error_message)

        return (
            error_summary,
            json.dumps({"error": error_message}, indent=2),
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            product_image
        )


with gr.Blocks(title="Claims-Resolve AI") as demo:
    gr.Markdown("# Claims-Resolve AI")
    gr.Markdown(
    "**Demo note:** Use BestBuy for replacement claims, eBay for refund/return claims, "
    "and Walmart for damaged-delivery examples. Policy retrieval is filtered by selected retailer."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Customer Inputs")

            complaint_input = gr.Textbox(
                label="Customer Complaint",
                placeholder=(
                    "Example: My headphones arrived broken and the box was crushed. "
                    "I want a replacement."
                ),
                lines=5
            )

            product_image_input = gr.Image(
                label="Upload Product / Package Image",
                type="filepath"
            )

            vision_backend_input = gr.Dropdown(
                label="Vision Backend",
                choices=["groq", "mlx"],
                value="groq"
            )

            company_input = gr.Dropdown(
                label="Retailer / Policy Source",
                choices=["bestbuy", "walmart", "ebay"],
                value="bestbuy"
            )

        with gr.Column():
            gr.Markdown("## Order Metadata")

            order_id_input = gr.Textbox(
                label="Order ID",
                value="ORD12345"
            )

            product_name_input = gr.Textbox(
                label="Product Name",
                value="Wireless Headphones"
            )

            order_value_input = gr.Number(
                label="Order Value",
                value=149.99
            )

            delivery_status_input = gr.Dropdown(
                label="Delivery Status",
                choices=["delivered", "in_transit", "delayed", "not_delivered"],
                value="delivered"
            )

            delivery_date_input = gr.Textbox(
                label="Delivery Date YYYY-MM-DD",
                value="2026-05-25"
            )

            return_deadline_input = gr.Textbox(
                label="Return Deadline YYYY-MM-DD",
                value="2026-06-24"
            )

            customer_previous_claims_input = gr.Number(
                label="Customer Previous Claims",
                value=1
            )

    analyze_button = gr.Button("Analyze Full Claim")

    gr.Markdown("## Claim Decision Dashboard")

    decision_summary_output = gr.Markdown()

    gr.Markdown("## Agent Outputs and Audit Trail")

    with gr.Row():
        complaint_output = gr.Code(
            label="Complaint Agent Output",
            language="json"
        )

        order_output = gr.Code(
            label="Order Agent Output",
            language="json"
        )

    with gr.Row():
        vision_output = gr.Code(
            label="Vision Agent Output",
            language="json"
        )

        image_preview = gr.Image(
            label="Uploaded Product Image Preview"
        )

    policy_output = gr.Code(
        label="Policy RAG Output",
        language="json"
    )

    risk_output = gr.Code(
        label="Risk Scoring Output",
        language="json"
    )

    resolution_output = gr.Code(
        label="Final Resolution Output",
        language="json"
    )

    timings_output = gr.Code(
        label="Timing Metrics Output",
        language="json"
    )

    graph_output = gr.Code(
        label="LangGraph Supervisor Metadata",
        language="json"
    )

    analyze_button.click(
        fn=run_app,
        inputs=[
            complaint_input,
            product_image_input,
            vision_backend_input,
            company_input,
            order_id_input,
            product_name_input,
            order_value_input,
            delivery_status_input,
            delivery_date_input,
            return_deadline_input,
            customer_previous_claims_input
        ],
        outputs=[
            decision_summary_output,
            complaint_output,
            order_output,
            vision_output,
            policy_output,
            risk_output,
            resolution_output,
            timings_output,
            graph_output,
            image_preview
        ]
    )


if __name__ == "__main__":
    demo.launch()