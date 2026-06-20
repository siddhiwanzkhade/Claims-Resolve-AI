import json
from groq import Groq
from pydantic import ValidationError

from src.config import GROQ_API_KEY, GROQ_TEXT_MODEL
from src.prompts import COMPLAINT_ANALYSIS_PROMPT
from src.schemas import ComplaintAnalysis


client = Groq(api_key=GROQ_API_KEY)


VALID_ISSUE_TYPES = {
    "damaged_item",
    "defective_item",
    "package_damage",
    "late_delivery",
    "missing_item",
    "wrong_item",
    "refund_request",
    "unclear"
}

VALID_REQUESTED_ACTIONS = {
    "refund",
    "replacement",
    "return",
    "escalation",
    "unclear"
}

VALID_SENTIMENTS = {
    "positive",
    "neutral",
    "negative"
}

VALID_URGENCY = {
    "low",
    "medium",
    "high"
}


def extract_json(text: str) -> dict:
    """
    Safely extracts JSON from LLM output.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != 0:
            return json.loads(text[start:end])

        raise ValueError(f"Could not parse JSON from model output: {text}")


def normalize_label(value: str, valid_values: set, default: str) -> str:
    """
    Normalizes labels when the LLM returns values like:
    'package_damage | damaged_item'

    It picks the first valid label it finds.
    """

    if not value:
        return default

    value = str(value).strip().lower()

    if value in valid_values:
        return value

    # Handle outputs like "package_damage | damaged_item"
    for part in value.replace("/", "|").split("|"):
        cleaned = part.strip()
        if cleaned in valid_values:
            return cleaned

    # Handle natural language-ish variants
    aliases = {
        "broken_item": "damaged_item",
        "broken product": "damaged_item",
        "damaged product": "damaged_item",
        "replace": "replacement",
        "replacement_request": "replacement",
        "refund_request": "refund",
        "angry": "negative",
        "urgent": "high",
    }

    if value in aliases and aliases[value] in valid_values:
        return aliases[value]

    return default


def normalize_complaint_result(result: dict) -> dict:
    """
    Cleans raw LLM output before Pydantic validation.
    """

    result["issue_type"] = normalize_label(
        result.get("issue_type", "unclear"),
        VALID_ISSUE_TYPES,
        "unclear"
    )

    result["requested_action"] = normalize_label(
        result.get("requested_action", "unclear"),
        VALID_REQUESTED_ACTIONS,
        "unclear"
    )

    result["sentiment"] = normalize_label(
        result.get("sentiment", "neutral"),
        VALID_SENTIMENTS,
        "neutral"
    )

    result["urgency"] = normalize_label(
        result.get("urgency", "low"),
        VALID_URGENCY,
        "low"
    )

    result["summary"] = str(result.get("summary", "")).strip()

    if not result["summary"]:
        result["summary"] = "No summary provided."

    return result


def validate_complaint_result(result: dict) -> dict:
    """
    Validates complaint-agent output using Pydantic.

    This prevents malformed LLM JSON or unexpected labels from
    entering the LangGraph state.
    """

    normalized_result = normalize_complaint_result(result)

    try:
        validated = ComplaintAnalysis(**normalized_result)
        return validated.model_dump()

    except ValidationError as e:
        raise ValueError(
            f"Complaint Agent returned invalid structured output after normalization: {e}"
        )


def analyze_complaint(complaint: str) -> dict:
    """
    Uses Groq LLM to analyze customer complaint text.
    Returns validated structured output.
    """

    if not complaint or complaint.strip() == "":
        fallback_result = {
            "issue_type": "unclear",
            "requested_action": "unclear",
            "sentiment": "neutral",
            "urgency": "low",
            "summary": "No complaint provided."
        }

        return validate_complaint_result(fallback_result)

    prompt = COMPLAINT_ANALYSIS_PROMPT.format(complaint=complaint)

    response = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator for e-commerce complaint analysis. "
                    "Return only valid JSON. For every field, choose exactly one label. "
                    "Never return multiple labels separated by |, comma, slash, or 'and'."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content

    raw_result = extract_json(output_text)

    return validate_complaint_result(raw_result)


if __name__ == "__main__":
    print("Complaint agent started...")

    test_complaint = (
        "My headphones arrived broken and the box was crushed. "
        "I want a replacement."
    )

    print("Analyzing complaint...")
    result = analyze_complaint(test_complaint)

    print("Result:")
    print(json.dumps(result, indent=2))