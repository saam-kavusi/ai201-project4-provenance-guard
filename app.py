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
  POST /appeal  - let a creator contest a classification
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
AUDIT_LOG_FILE = "audit_log.json"

app = Flask(__name__)

# Rate limiter. We attach limits per-endpoint (see /submit) rather than
# applying a global default, so /log and /appeal stay unthrottled.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
)

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
    return llm_score, normalize_whitespace(explanation)


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


def normalize_whitespace(text):
    """Collapse runs of whitespace in the explanation into single spaces.

    One test produced an explanation with a missing/odd space (e.g.
    "generic  language" rendering as "genericlanguage"-style spacing issues).
    This tidies the spacing without changing the wording: runs of whitespace
    become a single space and leading/trailing whitespace is stripped.
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


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


# Multi-modal stretch support: turn an image_metadata object into a plain-text
# block so the existing text-based detector can score it unchanged. We simply
# join each provided field into a readable "Key: value" line.
def metadata_to_text(metadata):
    """Flatten an image-metadata dict into readable text lines."""
    lines = []
    for key, value in metadata.items():
        if value is None or str(value).strip() == "":
            continue
        readable_key = str(key).replace("_", " ").strip().title()
        lines.append(f"{readable_key}: {str(value).strip()}")
    return "\n".join(lines)


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

    Raises AssertionError unless every label matches its canonical wording
    exactly, character for character. Exact equality (rather than substring
    checks) catches collapsed spacing such as "strong signs" -> "strongsigns".
    """
    assert LABELS["likely_ai"] == (
        "Provenance Guard found strong signs that this content may have been "
        "AI-generated."
    )
    assert LABELS["likely_human"] == (
        "Provenance Guard found strong signs that this content was likely "
        "written by a human."
    )
    assert LABELS["uncertain"] == (
        "Provenance Guard could not confidently determine whether this content "
        "was human-written or AI-generated."
    )


def classify(llm_score, stylometric_score, repetition_score):
    """Blend the three signals and map the result to a final verdict.

    combined_score = 0.50*llm + 0.30*stylometric + 0.20*repetition

    Interpreting combined_score as AI-likelihood:
      0.00 - 0.39  -> likely_human
      0.40 - 0.64  -> uncertain
      0.65 - 1.00  -> likely_ai

    Returns (attribution, confidence, label, combined_score).
    """
    combined_score = (
        0.50 * llm_score
        + 0.30 * stylometric_score
        + 0.20 * repetition_score
    )

    if combined_score >= 0.65:
        attribution = "likely_ai"
        confidence = combined_score
    elif combined_score < 0.40:
        attribution = "likely_human"
        confidence = 1 - combined_score
    else:
        attribution = "uncertain"
        # Keep uncertain confidence moderate: ~0.50 at the middle of the
        # 0.40 - 0.65 band and only slightly higher toward the edges, so an
        # uncertain result never reads as high-confidence AI/human.
        confidence = clamp(0.50 + abs(combined_score - 0.525))

    label = LABELS[attribution]

    # Round for clean output.
    confidence = round(confidence, 2)
    combined_score = round(combined_score, 2)
    return attribution, confidence, label, combined_score


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------

def append_audit_entry(entry):
    """Append a structured entry to the local JSON audit log file.

    Guarded by assert_labels_intact() so a corrupted label can never reach
    the file through the append path (e.g. /submit).
    """
    assert_labels_intact([entry])
    entries = read_audit_log()
    entries.append(entry)
    # Re-validate the FULL list we are about to persist, not just the new
    # entry. This catches a corrupted label that may already be present in the
    # existing file (or any other entry) before we write the whole file back.
    assert_labels_intact(entries)
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


def update_audit_log(entries):
    """Overwrite the audit log file with the given list of entries.

    Used when we need to mutate an existing entry in place (e.g. flipping a
    submission's status to 'under_review' after an appeal).

    Guarded by assert_labels_intact() so a corrupted label can never reach
    the file through the overwrite path (e.g. /appeal).
    """
    assert_labels_intact(entries)
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


# Substrings that can only appear when a label's spacing has been mangled
# (e.g. "strong signs" collapsed to "strongsigns"). A correct label never
# contains any of these, so we treat their presence as corruption.
MANGLED_LABEL_MARKERS = ("strongsigns", "writtenby", "human-writtenor")


def assert_labels_intact(entries):
    """Raise unless every entry's label is exactly one of the canonical labels.

    Defensive guard against the class of bug where a label is rebuilt or
    re-spaced before being persisted (e.g. "a human" collapsed to "ahuman").
    Validation is exact membership in LABELS.values(): any deviation,
    character for character, is rejected so we never write a corrupted label
    to the audit log. MANGLED_LABEL_MARKERS is kept only as an extra
    diagnostic hint in the error message.
    """
    canonical = set(LABELS.values())
    for entry in entries:
        if "label" not in entry:
            continue
        label = entry["label"]
        if label in canonical:
            continue
        lowered = label.lower() if isinstance(label, str) else ""
        hit = next((m for m in MANGLED_LABEL_MARKERS if m in lowered), None)
        marker_hint = f" (matches mangled marker '{hit}')" if hit else ""
        raise ValueError(
            "Refusing to write audit log: label is not a canonical value"
            f"{marker_hint}: {label!r}"
        )


def find_submission_entry(content_id):
    """Return (index, entry) for the submission with this content_id.

    Only 'submission' entries are matched, so an appeal event sharing the
    same content_id is never mistaken for the original submission. Returns
    (None, None) when no matching submission exists.
    """
    entries = read_audit_log()
    for index, entry in enumerate(entries):
        if entry.get("event_type") == "submission" and entry.get("content_id") == content_id:
            return index, entry
    return None, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    creator_id = data.get("creator_id")

    # Validate required fields.
    if not creator_id or not str(creator_id).strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    # Multi-modal stretch support: /submit accepts an optional content_type so
    # the same detection pipeline can score plain text or image metadata. Text
    # submissions behave exactly as before; image_metadata submissions provide
    # a "metadata" object that we flatten into readable text for the detector.
    content_type = data.get("content_type", "text")

    if content_type == "text":
        text = data.get("text")
        if not text or not str(text).strip():
            return jsonify({"error": "Field 'text' is required."}), 400
    elif content_type == "image_metadata":
        metadata = data.get("metadata")
        if not isinstance(metadata, dict) or not metadata:
            return jsonify({
                "error": "Field 'metadata' is required for image_metadata submissions."
            }), 400
        text = metadata_to_text(metadata)
    else:
        return jsonify({
            "error": "Unsupported content_type. Use 'text' or 'image_metadata'."
        }), 400

    # Run all three detection signals.
    llm_score, explanation = get_llm_score(text)
    stylometric_score = get_stylometric_score(text)
    repetition_score = get_repetition_score(text)

    # Blend the signals into the final verdict.
    attribution, confidence, label, combined_score = classify(
        llm_score, stylometric_score, repetition_score
    )

    # The label must be the exact canonical string for this attribution. We
    # reuse classify()'s returned label verbatim everywhere below and never
    # reconstruct or re-space it.
    assert label == LABELS[attribution]

    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build the audit log entry as a named dict so we can assert on the exact
    # object we hand to the logger.
    audit_entry = {
        "event_type": "submission",
        "timestamp": timestamp,
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": round(llm_score, 2),
        "stylometric_score": stylometric_score,
        "repetition_score": repetition_score,
        "combined_score": combined_score,
        "status": "classified",
        "label": label,
    }

    # Final canonical-label checks immediately before persisting. The label on
    # the dict we are about to write must match its attribution exactly and be
    # a member of the canonical label set, character for character.
    assert audit_entry["label"] == LABELS[attribution]
    assert audit_entry["label"] in set(LABELS.values())

    # Write the audit log entry.
    append_audit_entry(audit_entry)

    # Build and return the API response.
    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
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


@app.route("/appeal", methods=["POST"])
def appeal():
    """Let a creator contest the classification of a prior submission.

    Marks the original submission entry as 'under_review', records the
    appeal on that entry, and appends a separate 'appeal' audit event.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")
    creator_id = data.get("creator_id")

    # Validate required fields.
    if not content_id or not str(content_id).strip():
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not creator_reasoning or not str(creator_reasoning).strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    # Look up the original submission.
    index, entry = find_submission_entry(content_id)
    if entry is None:
        return jsonify({
            "error": f"No submission found with content_id '{content_id}'."
        }), 404

    timestamp = datetime.now(timezone.utc).isoformat()
    appeal_id = str(uuid.uuid4())

    # Update the original submission entry in place.
    #
    # We ONLY add/update the appeal-related fields below. The existing "label"
    # value is preserved exactly as written by /submit: it is never read,
    # rebuilt, re-spaced, or recomputed here, and classify() is never called
    # in this route.
    entries = read_audit_log()
    entries[index]["status"] = "under_review"
    entries[index]["appeal_submitted"] = True
    entries[index]["appeal_reasoning"] = creator_reasoning
    entries[index]["appeal_timestamp"] = timestamp

    # Append a separate audit event describing the appeal itself.
    appeal_entry = {
        "event_type": "appeal",
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_reasoning": creator_reasoning,
        "timestamp": timestamp,
        "status": "under_review",
    }
    if creator_id is not None and str(creator_id).strip():
        appeal_entry["creator_id"] = creator_id
    entries.append(appeal_entry)

    # update_audit_log() runs the shared assert_labels_intact() guard before
    # writing, so corrupted labels can never be persisted here.
    update_audit_log(entries)

    return jsonify({
        "appeal_received": True,
        "content_id": content_id,
        "status": "under_review",
        "appeal_id": appeal_id,
    })


# Stretch feature: Provenance Certificate.
#
# Read-only endpoint that builds a shareable provenance certificate for a
# previously classified submission. It only reads the audit log and never
# calls the LLM or re-runs any detection signal.
@app.route("/certificate/<content_id>", methods=["GET"])
def certificate(content_id):
    """Return a provenance certificate for a classified submission."""
    _, entry = find_submission_entry(content_id)
    if entry is None:
        return jsonify({"error": "No submission found for this content_id."}), 404

    return jsonify({
        "certificate_id": "cert-" + content_id,
        "content_id": content_id,
        "creator_id": entry.get("creator_id"),
        "attribution": entry.get("attribution"),
        "confidence": entry.get("confidence"),
        "label": entry.get("label"),
        "timestamp": entry.get("timestamp"),
        "audit_status": entry["status"],
        "appeal_submitted": entry.get("appeal_submitted", False),
        "signal_scores": {
            "llm_score": entry.get("llm_score"),
            "stylometric_score": entry.get("stylometric_score"),
            "repetition_score": entry.get("repetition_score"),
            "combined_score": entry.get("combined_score"),
        },
    })


# Stretch feature: Analytics Dashboard.
#
# Read-only endpoint that summarizes the audit log into aggregate metrics
# (submission/appeal totals, detection breakdown, appeal rate, and average
# scores). It only reads the audit log and never calls the LLM.
@app.route("/analytics", methods=["GET"])
def analytics():
    """Return aggregate detection metrics computed from the audit log."""
    entries = read_audit_log()

    submissions = [e for e in entries if e.get("event_type") == "submission"]
    appeals = [e for e in entries if e.get("event_type") == "appeal"]

    total_submissions = len(submissions)
    total_appeals = len(appeals)

    # Detection breakdown across submissions only.
    detection_counts = {"likely_ai": 0, "likely_human": 0, "uncertain": 0}
    for entry in submissions:
        attribution = entry.get("attribution")
        if attribution in detection_counts:
            detection_counts[attribution] += 1

    if total_submissions:
        appeal_rate = round(total_appeals / total_submissions, 2)
        average_confidence = round(
            sum(e.get("confidence", 0) for e in submissions) / total_submissions, 2
        )
        average_combined_score = round(
            sum(e.get("combined_score", 0) for e in submissions) / total_submissions, 2
        )
    else:
        appeal_rate = 0
        average_confidence = 0
        average_combined_score = 0

    return jsonify({
        "total_submissions": total_submissions,
        "total_appeals": total_appeals,
        "detection_counts": detection_counts,
        "appeal_rate": appeal_rate,
        "average_confidence": average_confidence,
        "average_combined_score": average_combined_score,
    })


if __name__ == "__main__":
    validate_labels()
    app.run(debug=True, port=5001)
