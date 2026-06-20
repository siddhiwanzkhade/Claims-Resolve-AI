import base64
import json
import mimetypes
import subprocess

from groq import Groq
from pydantic import ValidationError

from src.schemas import VisionAnalysis

from src.config import (
    GROQ_API_KEY,
    GROQ_VISION_MODEL,
    VISION_BACKEND,
    MLX_VISION_MODEL
)



# Groq client setup

# This client is used only when backend="groq".
# Groq is the hosted VLM option, meaning the model runs on Groq's servers.
client = Groq(api_key=GROQ_API_KEY)


def extract_json(text: str) -> dict:
    """
    Extracts the first valid JSON object from model output.

    Some models or CLI tools return extra text/logs after the JSON.
    This function avoids the 'Extra data' JSON parsing error by decoding
    only the first complete JSON object.
    """

    text = text.strip()

    # Try direct JSON parsing first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    
    start = text.find("{")

    if start == -1:
        raise ValueError(f"Could not find JSON object in model output: {text}")

    # Use JSONDecoder to parse only the first complete JSON object.
    decoder = json.JSONDecoder()

    try:
        result, _ = decoder.raw_decode(text[start:])
        return result
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON from model output: {text}")

def validate_vision_result(result: dict) -> dict:
    """
    Validates vision-agent output using Pydantic.

    This ensures the VLM returns the required fields before the result
    enters the LangGraph state.
    """
    try:
        validated = VisionAnalysis(**result)
        return validated.model_dump()

    except ValidationError as e:
        raise ValueError(
            f"Vision Agent returned invalid structured output: {e}"
        )
    
def encode_image_to_data_url(image_path: str) -> str:
    """
    Converts a local image file into a base64 data URL.

    Groq/OpenAI-style vision APIs usually expect images as:
    data:image/jpeg;base64,<encoded_image>

    So this function:
    1. reads the local image
    2. converts it to base64
    3. adds the correct MIME type prefix
    """

    mime_type, _ = mimetypes.guess_type(image_path)

    # If Python cannot detect the image type, default to JPEG.
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def get_damage_prompt() -> str:
    """
    Central prompt used by both Groq and MLX backends.

    This prompt forces the model to choose one value for each field,
    instead of copying all possible labels.
    """

    return """
You are an e-commerce product damage inspection assistant.

Analyze the uploaded product image and return ONLY one valid JSON object.

Do not copy the option lists. Choose exactly one value for each field.

Allowed values:
- damage_detected: true or false
- damage_type: broken_component, cracked_item, crushed_packaging, missing_part, no_visible_damage, unclear
- damage_severity: low, medium, high, unknown
- confidence: low, medium, high

Return JSON in this exact format:
{
  "damage_detected": true,
  "damage_type": "broken_component",
  "damage_severity": "medium",
  "visible_evidence": "briefly describe the visible evidence in the image",
  "confidence": "medium"
}
"""

def analyze_with_groq(image_path: str) -> dict:
    """
    Analyzes product damage using Groq's hosted vision model.

    This is the reliable demo backend:
    - fast
    - easy to run
    - no local model download
    - needs Groq API key
    """

    # Convert local image into base64 data URL for API input.
    image_data_url = encode_image_to_data_url(image_path)

    prompt = get_damage_prompt()

    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator for e-commerce damage inspection. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    # Extract the model's text response.
    output_text = response.choices[0].message.content

    # Convert model text into Python dictionary.
    result = extract_json(output_text)

    # Add metadata so we know which backend produced the result.
    result["model_backend"] = "groq_vision"

    return validate_vision_result(result)


def analyze_with_mlx(image_path: str) -> dict:
    """
    Analyzes product damage using local quantized Qwen2.5-VL through MLX-VLM.

    This is the local quantization backend:
    - uses a 4-bit quantized VLM
    - runs locally on Apple Silicon through MLX
    - does not need paid hosted vision API
    - useful for showing quantization/local inference skills

    Note:
    This uses subprocess to call the MLX-VLM CLI:
    python -m mlx_vlm.generate ...
    """

    prompt = get_damage_prompt()

    command = [
        "python",
        "-m",
        "mlx_vlm.generate",
        "--model",
        MLX_VISION_MODEL,
        "--image",
        image_path,
        "--prompt",
        prompt,
        "--max-tokens",
        "300",
        "--temp",
        "0.0"
    ]

    # Run the MLX-VLM command from Python.
    # capture_output=True stores stdout/stderr instead of printing directly.
    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    # If MLX fails, raise the exact terminal error.
    if completed_process.returncode != 0:
        raise RuntimeError(completed_process.stderr)

    raw_output = completed_process.stdout

    # Convert model output text into dictionary.
    result = extract_json(raw_output)

    # Add backend metadata.
    result["model_backend"] = "mlx_qwen2.5_vl_4bit"

    return validate_vision_result(result)


def analyze_product_image(image_path: str, backend: str = None) -> dict:
    """
    Main function used by the rest of the project.

    Parameters:
    - image_path: path to uploaded product image
    - backend:
        "groq" = hosted Groq vision model
        "mlx"  = local 4-bit quantized Qwen2.5-VL through MLX

    If backend is not provided, it uses VISION_BACKEND from config.py/.env.
    """

    # Graceful fallback if user does not upload an image.
    if not image_path:
        fallback_result = {
            "damage_detected": False,
            "damage_type": "no_image",
            "damage_severity": "unknown",
            "visible_evidence": "No image was provided.",
            "confidence": "low",
            "model_backend": "none"
        }
        return validate_vision_result(fallback_result)

    # Use explicit backend if provided, otherwise use default from .env.
    backend = backend or VISION_BACKEND
    backend = backend.lower()

    if backend == "groq":
        return analyze_with_groq(image_path)

    if backend == "mlx":
        return analyze_with_mlx(image_path)

    raise ValueError(f"Unsupported vision backend: {backend}")




# It will test both backends without using Gradio.
if __name__ == "__main__":
    test_image_path = "/Users/siddhiwanzkhade/E-commerce Claim AI/data/policy/images/test_damage.jpg"

    print("Testing Groq backend...")
    try:
        groq_result = analyze_product_image(test_image_path, backend="groq")
        print(json.dumps(groq_result, indent=2))
    except Exception as e:
        print(f"Groq backend failed: {e}")

    print("\nTesting MLX backend...")
    try:
        mlx_result = analyze_product_image(test_image_path, backend="mlx")
        print(json.dumps(mlx_result, indent=2))
    except Exception as e:
        print(f"MLX backend failed: {e}")