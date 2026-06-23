"""
Provenance Guard - Milestone 3

A Flask backend that accepts submitted text and uses a single detection
signal (an LLM judgment from Groq) to classify whether the text appears
AI-generated, human-written, or uncertain.

Endpoints:
  POST /submit  - classify a submitted writing sample
  GET  /log     - return recent structured audit log entries
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from groq import Groq

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
AUDIT_LOG_FILE = "audit_log.json"

app = Flask(__name__)

# Create the Groq client once at startup so we reuse it across requests.
client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Detection Signal 1: LLM score
# ---------------------------------------------------------------------------

def get_llm_score(text):
    """Ask the LLM how likely the text is AI-generated.

    Returns a tuple of (llm_score, explanation):
      - llm_score: float from 0.0 to 1.0 (higher = more likely AI-generated)
      - explanation: short string describing the reasoning
    """
    system_prompt = (
        "You are a detector that estimates how likely a piece of text was "
        "written by an AI language model versus a human. "
        "Respond ONLY with a valid JSON object in this exact shape:\n"
        '{"llm_score": <float between 0.0 and 1.0>, '
        '"explanation": "<short reason>"}\n'
        "A higher llm_score means more likely AI-generated. "
        "A lower llm_score means more likely human-written."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content
    llm_score, explanation = parse_llm_response(raw_content)
    return llm_score, explanation


def parse_llm_response(raw_content):
    """Pull the llm_score and explanation out of the model's reply.

    The model is asked for JSON, but we stay defensive in case it wraps the
    JSON in extra text. Falls back to a neutral 0.5 score if parsing fails.
    """
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        # Try to find a JSON object embedded in the text.
        match = re.search(r"\{.*\}", raw_content or "", re.DOTALL)
        if not match:
            return 0.5, "Could not parse model response; defaulting to uncertain."
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return 0.5, "Could not parse model response; defaulting to uncertain."

    score = data.get("llm_score", 0.5)
    explanation = data.get("explanation", "")

    # Clamp the score into the valid 0.0 - 1.0 range.
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))

    return score, explanation


# ---------------------------------------------------------------------------
# Attribution mapping (temporary Milestone 3 rules)
# ---------------------------------------------------------------------------

def classify(llm_score):
    """Map an llm_score to (attribution, confidence, label)."""
    if llm_score >= 0.75:
        attribution = "likely_ai"
        confidence = llm_score
        label = (
            "Provenance Guard found strong signs that this content may have "
            "been AI-generated."
        )
    elif llm_score <= 0.39:
        attribution = "likely_human"
        confidence = 1 - llm_score
        label = (
            "Provenance Guard found strong signs that this content was likely "
            "written by a human."
        )
    else:
        attribution = "uncertain"
        confidence = 0.5
        label = (
            "Provenance Guard could not confidently determine whether this "
            "content was human-written or AI-generated."
        )

    # Round for clean output.
    confidence = round(confidence, 2)
    return attribution, confidence, label


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------

def append_audit_entry(entry):
    """Append a structured entry to the local JSON audit log file."""
    entries = read_audit_log()
    entries.append(entry)
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def read_audit_log():
    """Read all audit log entries, returning an empty list if none exist."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        # If the file is empty or corrupt, start fresh.
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    creator_id = data.get("creator_id")
    text = data.get("text")

    # Validate required fields.
    if not creator_id or not str(creator_id).strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400
    if not text or not str(text).strip():
        return jsonify({"error": "Field 'text' is required."}), 400

    # Run detection signal 1.
    llm_score, explanation = get_llm_score(text)

    # Map to attribution / confidence / label.
    attribution, confidence, label = classify(llm_score)

    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Write the audit log entry.
    append_audit_entry({
        "event_type": "submission",
        "timestamp": timestamp,
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": round(llm_score, 2),
        "status": "classified",
        "label": label,
    })

    # Build and return the API response.
    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": label,
        "signals": {
            "llm_score": round(llm_score, 2),
            "explanation": explanation,
        },
        "status": "classified",
    })


@app.route("/log", methods=["GET"])
def log():
    """Return recent audit log entries (most recent last)."""
    return jsonify({"entries": read_audit_log()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
