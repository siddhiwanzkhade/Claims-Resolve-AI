# E-commerce-Claim-AI

E-commerce-Claim-AI is a multimodal agentic AI solution designed to streamline and strengthen e-commerce claim investigation. By leveraging AI agents, vision-language models, NLP, and risk scoring, it helps evaluate refund, replacement, damaged-product, wrong-item, and missing-package claims with better accuracy and decision support.

## Features

- **Automated Claim Investigation**: Uses AI agents to assess customer claims, reduce manual review effort, and speed up refund/replacement decisions.

- **Multimodal Image Analysis**: Employs vision-language models to evaluate uploaded product or package images for damage, missing parts, packaging issues, and evidence quality.

- **NLP-Based Complaint Understanding**: Analyzes customer-submitted complaints to extract issue type, urgency, sentiment, requested action, and missing information.

- **Return-Abuse Risk Detection**: Identifies potentially suspicious claims using order value, claim history, return-window status, evidence quality, and image-text mismatch signals.

- **Dynamic Agent Routing**: Uses a Router/Supervisor Agent to call only the agents needed for each claim instead of running every case through the same fixed workflow.

- **Image-Text Mismatch Detection**: Compares customer complaints with uploaded visual evidence to flag inconsistent or unclear claims.

- **Evidence Request Handling**: Requests additional proof when the claim cannot be safely approved or rejected.

- **Decision Support**: Recommends refund, replacement, rejection, evidence request, or human escalation based on claim context and risk level.

## Agent Architecture

E-commerce-Claim-AI uses multiple specialized agents:

- **Router Agent**: Decides which agents should be called for each claim.
- **Complaint Agent**: Extracts claim type, sentiment, urgency, and requested action.
- **Order Agent**: Validates order status, delivery status, return window, product value, and claim history.
- **Vision Agent**: Analyzes uploaded product/package images.
- **Mismatch Agent**: Detects inconsistencies between complaint text, images, and order data.
- **Risk Agent**: Scores return-abuse risk and escalation need.
- **Evidence Agent**: Requests missing or unclear proof.
- **Resolution Agent**: Generates the final claim recommendation.

## Example Claim Paths

```text
Wrong size claim
→ Complaint Agent + Order Agent + Resolution Agent

Damaged item with image
→ Complaint Agent + Order Agent + Vision Agent + Risk Agent + Resolution Agent

Damaged item without image
→ Complaint Agent + Order Agent + Evidence Agent

Image-text mismatch
→ Complaint Agent + Vision Agent + Mismatch Agent + Risk Agent + Human Escalation

High-value repeated claim
→ Complaint Agent + Order Agent + Risk Agent + Human Escalation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/siddhiwanzkhade/E-commerce-Claim-AI.git
```

2.Navigate to the project directory
```bash
cd E-commerce-Claim-AI
```

3. Create and activate virtual environment
 ```bash
python -m venv claimguard_env
source claimguard_env/bin/activate
```
4. Install Dependencies
 ```bash
pip install -r requirements.txt
```
5. Set up environment variables:
Create a .env file in the project root and add the required API keys and model configuration.


## Supported Models
E-commerce-Claim-AI supports LLMs for complaint analysis, routing, risk reasoning, and final recommendation generation. It also supports vision-language models for product/package image analysis.

## Usage
Launch the Gradio dashboard:
python app.py
- Use the interface to submit:
- Customer complaint
- Order details
- Product/package images
- Claim metadata
The system returns a structured claim investigation report with agent outputs, risk signals, and the final recommendation.

## Tech Stack
Python
LangGraph
Pydantic
LLMs
Vision-Language Models
Gradio
Rule-based risk and policy logic

## Goal
E-commerce-Claim-AI helps e-commerce teams make faster, safer, and more evidence-aware claim decisions by combining multimodal analysis, dynamic agent routing, and return-abuse risk detection.

<img width="2534" height="1376" alt="image" src="https://github.com/user-attachments/assets/0b764d6d-00d8-4091-8938-35647c31d741" />
<img width="1266" height="713" alt="image" src="https://github.com/user-attachments/assets/3a68ab96-5454-4c09-8aa3-6d06a439b0fa" />
<img width="1280" height="706" alt="Screenshot 2026-05-31 at 3 55 26 PM" src="https://github.com/user-attachments/assets/b6ab07d8-6f46-4024-94e3-f223b22a6185" />
<img width="1277" height="712" alt="Screenshot 2026-05-31 at 3 55 36 PM" src="https://github.com/user-attachments/assets/4f2acc01-21dd-4c0e-9eaa-f3a6dde61450" />



