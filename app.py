"""
Provenance Guard - Milestone 4

A Flask backend that accepts submitted text and uses three detection
signals to classify whether the text appears AI-generated, human-written,
or uncertain:

  1. LLM-based classification (Groq llama-3.3-70b-versatile)
  2. Stylometric heuristics (pure Python writing statistics)
  3. Repetition / generic-phrase detection (pure Python)

The three signals are blended into a single combined_score that drives the
final attribution and confidence.

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
# Shared text helpers (used by the pure-Python signals)
# ---------------------------------------------------------------------------

def split_sentences(text):
    """Split text into sentences on ., !, and ? boundaries."""
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def split_words(text):
    """Return a lowercased list of word tokens (letters and apostrophes)."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def clamp(value):
    """Keep a number inside the 0.0 - 1.0 range."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Detection Signal 2: Stylometric heuristics
# ---------------------------------------------------------------------------

def get_stylometric_score(text):
    """Estimate AI-likelihood from measurable writing statistics.

    Returns a float from 0.0 to 1.0:
      - Higher  = text looks uniform / formulaic (more AI-like)
      - Lower   = text looks varied / irregular (more human-like)

    It blends three sub-measurements:
      - sentence length variance (uniform lengths look AI-like)
      - type-token ratio / vocabulary diversity (low diversity looks AI-like)
      - punctuation variety (expressive punctuation looks human-like)
    """
    sentences = split_sentences(text)
    words = split_words(text)

    # Not enough signal to judge; stay neutral.
    if len(words) < 5 or len(sentences) == 0:
        return 0.5

    # --- Sub-measurement A: sentence length variance -----------------------
    # Humans tend to mix short and long sentences. We use the coefficient of
    # variation (std / mean) so the measure is independent of average length.
    lengths = [len(split_words(s)) for s in sentences]
    lengths = [length for length in lengths if length > 0]
    mean_len = sum(lengths) / len(lengths)

    if len(lengths) > 1 and mean_len > 0:
        variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
        std_dev = variance ** 0.5
        coeff_variation = std_dev / mean_len
    else:
        coeff_variation = 0.0

    # A CV of ~0.6+ is quite varied (human); ~0 is perfectly uniform (AI).
    uniformity_score = clamp(1.0 - (coeff_variation / 0.6))

    # --- Sub-measurement B: vocabulary diversity (type-token ratio) --------
    type_token_ratio = len(set(words)) / len(words)
    # TTR >= 0.7 is diverse (human); TTR <= 0.4 is repetitive (AI).
    low_diversity_score = clamp((0.7 - type_token_ratio) / 0.3)

    # --- Sub-measurement C: punctuation variety ----------------------------
    # Humans reach for a wider range of marks (; : ! ? ( ) - —).
    expressive_marks = set(re.findall(r"[;:!?()\-—]", text))
    punctuation_score = clamp(1.0 - (len(expressive_marks) / 4.0))

    # Weighted blend; sentence variance and diversity carry the most weight.
    stylometric_score = (
        0.45 * uniformity_score
        + 0.40 * low_diversity_score
        + 0.15 * punctuation_score
    )
    return round(clamp(stylometric_score), 2)


# ---------------------------------------------------------------------------
# Detection Signal 3: Repetition and generic phrases
# ---------------------------------------------------------------------------

# Common AI-style filler / transition phrases.
GENERIC_PHRASES = [
    "it is important to note",
    "furthermore",
    "moreover",
    "in conclusion",
    "plays a crucial role",
    "rapidly evolving",
    "transformative",
    "in today's world",
    "a testament to",
    "delve into",
]


def get_repetition_score(text):
    """Estimate AI-likelihood from repeated / formulaic language.

    Returns a float from 0.0 to 1.0:
      - Higher = more repeated phrases and stock transitions (more AI-like)
      - Lower  = less formulaic (more human-like)

    It blends three sub-measurements:
      - hits against a list of generic AI-style phrases
      - repeated sentence openings (same first word reused)
      - repeated three-word phrases (trigrams)
    """
    lowered = text.lower()
    words = split_words(text)
    sentences = split_sentences(text)

    # --- Sub-measurement A: generic phrase hits ----------------------------
    phrase_hits = sum(1 for phrase in GENERIC_PHRASES if phrase in lowered)
    # Three or more stock phrases is a strong tell.
    phrase_component = clamp(phrase_hits / 3.0)

    # --- Sub-measurement B: repeated sentence openings ---------------------
    openings = [split_words(s)[0] for s in sentences if split_words(s)]
    if openings:
        repeated_openings = len(openings) - len(set(openings))
        opening_component = clamp(repeated_openings / len(openings))
    else:
        opening_component = 0.0

    # --- Sub-measurement C: repeated three-word phrases --------------------
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        repeated_trigrams = len(trigrams) - len(set(trigrams))
        ngram_component = clamp(repeated_trigrams / len(trigrams))
    else:
        ngram_component = 0.0

    repetition_score = (
        0.50 * phrase_component
        + 0.30 * opening_component
        + 0.20 * ngram_component
    )
    return round(clamp(repetition_score), 2)


# ---------------------------------------------------------------------------
# Ensemble: blend the three signals into an attribution + confidence
# ---------------------------------------------------------------------------

# Human-readable label for each attribution. Each value is a single, complete
# string so the wording is identical in the API response and the audit log.
LABELS = {
    "likely_ai": "Provenance Guard found strong signs that this content may have been AI-generated.",
    "likely_human": "Provenance Guard found strong signs that this content was likely written by a human.",
    "uncertain": "Provenance Guard could not confidently determine whether this content was human-written or AI-generated.",
}


def validate_labels():
    """Startup self-check that guards against label-wording regressions.

    Raises AssertionError if any expected substring is missing, which can
    happen when a label is accidentally split across lines or mangled.
    """
    assert "strong signs that" in LABELS["likely_ai"]
    assert "likely written by" in LABELS["likely_human"]
    assert "human-written or AI-generated" in LABELS["uncertain"]


def classify(llm_score, stylometric_score, repetition_score):
    """Blend the three signals and map the result to a final verdict.

    combined_score = 0.50*llm + 0.30*stylometric + 0.20*repetition

    Interpreting combined_score as AI-likelihood:
      0.00 - 0.39  -> likely_human
      0.40 - 0.74  -> uncertain
      0.75 - 1.00  -> likely_ai

    Returns (attribution, confidence, label, combined_score).
    """
    combined_score = (
        0.50 * llm_score
        + 0.30 * stylometric_score
        + 0.20 * repetition_score
    )

    if combined_score >= 0.75:
        attribution = "likely_ai"
        confidence = combined_score
    elif combined_score < 0.40:
        attribution = "likely_human"
        confidence = 1 - combined_score
    else:
        attribution = "uncertain"
        # Keep uncertain confidence moderate: ~0.50 at the middle of the band
        # and only slightly higher toward the edges, so an uncertain result
        # never reads as high-confidence AI/human.
        confidence = clamp(0.50 + abs(combined_score - 0.575))

    label = LABELS[attribution]

    # Round for clean output.
    confidence = round(confidence, 2)
    combined_score = round(combined_score, 2)
    return attribution, confidence, label, combined_score


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

    # Run all three detection signals.
    llm_score, explanation = get_llm_score(text)
    stylometric_score = get_stylometric_score(text)
    repetition_score = get_repetition_score(text)

    # Blend the signals into the final verdict.
    attribution, confidence, label, combined_score = classify(
        llm_score, stylometric_score, repetition_score
    )

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
        "stylometric_score": stylometric_score,
        "repetition_score": repetition_score,
        "combined_score": combined_score,
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
            "stylometric_score": stylometric_score,
            "repetition_score": repetition_score,
            "explanation": explanation,
        },
        "status": "classified",
    })


@app.route("/log", methods=["GET"])
def log():
    """Return recent audit log entries (most recent last)."""
    return jsonify({"entries": read_audit_log()})


if __name__ == "__main__":
    validate_labels()
    app.run(debug=True, port=5001)
