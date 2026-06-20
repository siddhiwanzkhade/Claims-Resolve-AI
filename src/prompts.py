COMPLAINT_ANALYSIS_PROMPT = """
You are an e-commerce customer complaint analysis assistant.

Analyze the customer complaint below and return ONLY valid JSON.

Complaint:
{complaint}

Important rules:
- Choose exactly ONE value for each field.
- Do NOT return multiple labels separated by "|", comma, slash, or "and".
- Do NOT copy the list of allowed values into the output.
- Return only one valid JSON object.
- Do not include markdown, explanations, or extra text.

Allowed issue_type values:
damaged_item, defective_item, package_damage, late_delivery, missing_item, wrong_item, refund_request, unclear

Allowed requested_action values:
refund, replacement, return, escalation, unclear

Allowed sentiment values:
positive, neutral, negative

Allowed urgency values:
low, medium, high

Return JSON in this exact format:
{{
  "issue_type": "damaged_item",
  "requested_action": "replacement",
  "sentiment": "negative",
  "urgency": "high",
  "summary": "one sentence summary of the complaint"
}}
"""