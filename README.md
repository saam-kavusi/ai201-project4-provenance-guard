# Provenance Guard

## Overview

Provenance Guard is a Flask API that analyzes submitted content and returns an
attribution decision: **likely_human**, **uncertain**, or **likely_ai**. Rather
than relying on a single detector, it combines multiple detection signals into a
weighted ensemble, then maps the blended score to a verdict.

Alongside the core attribution, Provenance Guard provides plain-language
transparency labels that explain each decision, an append-only audit log of every
submission and appeal, an appeals workflow so creators can contest a result, and
rate limiting to protect the service. It also includes a set of stretch features:
ensemble detection, provenance certificates, an analytics endpoint, and
multi-modal image-metadata submissions.

Provenance Guard does not claim perfect AI detection. It is intentionally
conservative to reduce the risk of falsely accusing a human creator.

## Features

**Required features**
- `POST /submit` for content submission
- Multi-signal detection (LLM, stylometric, repetition)
- Confidence scoring for each decision
- Plain-language transparency labels
- Appeals workflow with `POST /appeal`
- Audit log with `GET /log`
- Rate limiting on `/submit`

**Stretch features**
- Ensemble detection using 3 weighted signals
- Provenance certificate endpoint (`GET /certificate/<content_id>`)
- Analytics endpoint (`GET /analytics`)
- Multi-modal `image_metadata` support on `/submit`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root containing your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Run the server:

```bash
python3 app.py
```

The server runs on:

```
http://localhost:5001
```

## API Endpoints

### 1. POST /submit

Submits content for analysis and returns an attribution decision. Supports two
content types: plain `text` and `image_metadata`.

**Text submission**

```bash
curl -s -X POST http://localhost:5001/submit \
  -H "Content-Type: application/json" \
  -d '{"creator_id":"alice","text":"This is my own writing.","content_type":"text"}' | python3 -m json.tool
```

**Image metadata submission**

```bash
curl -s -X POST http://localhost:5001/submit \
  -H "Content-Type: application/json" \
  -d '{"creator_id":"image_test","content_type":"image_metadata","metadata":{"prompt":"A futuristic city skyline","caption":"Generated concept art","tool":"AI image generator","description":"Polished synthetic architecture."}}' | python3 -m json.tool
```

**Example response**

```json
{
    "content_id": "f1c2a3b4-...",
    "creator_id": "alice",
    "content_type": "text",
    "attribution": "likely_human",
    "confidence": 0.78,
    "label": "Provenance Guard found strong signs that this content was likely written by a human.",
    "signals": {
        "llm_score": 0.20,
        "stylometric_score": 0.30,
        "repetition_score": 0.0,
        "explanation": "Short, plain first-person statement with no formulaic phrasing."
    },
    "status": "classified"
}
```

**Returned fields**
- `content_id` — unique ID assigned to this submission (used for appeals and certificates).
- `creator_id` — the creator who submitted the content.
- `content_type` — `text` or `image_metadata`.
- `attribution` — `likely_human`, `uncertain`, or `likely_ai`.
- `confidence` — how confident the decision is, from 0.0 to 1.0.
- `label` — a plain-language transparency label explaining the result.
- `signals` — the individual signal scores plus the LLM `explanation`.
- `status` — submission status (`classified` on a new submission).

### 2. GET /log

Returns the audit log entries recorded so far, including both submission events
and appeal events.

```bash
curl -s http://localhost:5001/log | python3 -m json.tool
```

### 3. POST /appeal

Lets a creator contest a prior classification.

```bash
curl -s -X POST http://localhost:5001/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id":"PASTE_CONTENT_ID","creator_id":"alice","creator_reasoning":"I wrote this by hand and can provide drafts."}' | python3 -m json.tool
```

The appeal updates the original submission's status to `under_review` and appends
a separate appeal event to the audit log. The creator's reasoning is preserved on
both the original submission entry and the appeal event.

**Example response**

```json
{
    "appeal_received": true,
    "content_id": "f1c2a3b4-...",
    "status": "under_review",
    "appeal_id": "9a8b7c6d-..."
}
```

### 4. GET /certificate/<content_id>

Returns a provenance certificate for a previously classified submission. This is a
read-only endpoint — it never re-runs detection or calls the LLM.

```bash
curl -s http://localhost:5001/certificate/PASTE_CONTENT_ID | python3 -m json.tool
```

The certificate includes `certificate_id`, `content_id`, `creator_id`,
`attribution`, `confidence`, `label`, `timestamp`, `audit_status`,
`appeal_submitted`, and `signal_scores` (the individual `llm_score`,
`stylometric_score`, `repetition_score`, and `combined_score`).

### 5. GET /analytics

Returns aggregate metrics computed from the audit log.

```bash
curl -s http://localhost:5001/analytics | python3 -m json.tool
```

The response includes `total_submissions`, `total_appeals`, `detection_counts`
(a breakdown by attribution), `appeal_rate`, `average_confidence`, and
`average_combined_score`.

## Detection Signals

Provenance Guard uses three independent signals:

1. **llm_score** — An LLM (Groq `llama-3.3-70b-versatile`) estimates how likely the
   content looks AI-generated, returning a score from 0.0 to 1.0. It captures
   nuanced phrasing and tone that simple heuristics miss, but it can be uncertain
   on short text and can be fooled by lightly edited AI writing.

2. **stylometric_score** — Measures writing structure from pure-Python statistics:
   sentence-length consistency (uniform lengths look AI-like), vocabulary diversity
   (type-token ratio), and punctuation variety. It captures structural uniformity
   well, but it can mislabel naturally consistent human writing and needs enough
   text to be meaningful.

3. **repetition_score** — Detects repeated or generic phrasing: stock AI-style
   transition phrases, repeated sentence openings, and repeated three-word phrases.
   It captures formulaic filler effectively, but it can miss AI text that avoids
   common filler and may over-flag legitimately repetitive content.

No single signal is trusted on its own; they are blended into an ensemble.

## Ensemble Formula

```
combined_score = (0.50 * llm_score) + (0.30 * stylometric_score) + (0.20 * repetition_score)
```

**Conflict resolution.** The LLM signal carries the highest weight (0.50),
stylometry is second (0.30), and repetition is third (0.20). Because the weights
are blended, a single signal cannot fully determine the final result — a strong
reading from one signal can be tempered by the others, which reduces the impact of
any single false positive.

## Attribution Thresholds

The `combined_score` (interpreted as AI-likelihood) maps to a verdict:

| combined_score | attribution   |
|----------------|---------------|
| 0.00 – 0.39    | likely_human  |
| 0.40 – 0.64    | uncertain     |
| 0.65 – 1.00    | likely_ai     |

## Transparency Labels

Every decision returns one of three exact labels:

- **likely_ai:** `Provenance Guard found strong signs that this content may have been AI-generated.`
- **likely_human:** `Provenance Guard found strong signs that this content was likely written by a human.`
- **uncertain:** `Provenance Guard could not confidently determine whether this content was human-written or AI-generated.`

These labels differ for high- and low-confidence results because they map to
different attribution outcomes and confidence scores: the `likely_ai` and
`likely_human` labels are returned only when the score lands in a confident band,
while the `uncertain` label is returned in the middle band where confidence is
moderate. The wording is validated against canonical strings at startup and before
every audit-log write, so labels cannot drift or become corrupted.

## Confidence Scoring

Confidence reflects how strongly the combined score supports the chosen verdict:

- **likely_human** confidence uses `1 - combined_score` — the lower the
  AI-likelihood, the higher the confidence it is human.
- **likely_ai** confidence uses `combined_score` — the higher the AI-likelihood,
  the higher the confidence it is AI.
- **uncertain** confidence is moderate and reflects distance from the center of the
  uncertain range, so an uncertain result never reads as a high-confidence verdict.

In testing, different inputs produced noticeably different confidence levels (for
example, a plainly human note scored much higher human-confidence than a polished,
formulaic paragraph).

## Appeals Workflow

1. A creator submits an appeal with the `content_id` and `creator_reasoning`.
2. The system marks the original submission's status as `under_review`.
3. The system appends a separate appeal event to the audit log.
4. The creator's reasoning is preserved on both the original submission entry and
   the appeal event.

The appeal route never recomputes the classification or rewrites the original
label — it only records the contest and flips the status for human review.

## Rate Limiting

`POST /submit` is limited to **10 per minute** and **100 per day** (per client
address). The `/log`, `/appeal`, `/certificate`, and `/analytics` endpoints are not
throttled.

**Test evidence.** Repeated submissions returned HTTP `429 Too Many Requests` once
the 10-per-minute limit was exceeded.

**Why.** Rate limiting prevents spam and abuse and controls API/LLM costs (each
submission may call the LLM), while still allowing normal creator usage.

## Audit Log

Each submission and appeal is recorded in `audit_log.json`. Submission entries
include:

- `timestamp`
- `content_id`
- `creator_id`
- `attribution`
- `confidence`
- individual signal scores (`llm_score`, `stylometric_score`, `repetition_score`)
- `combined_score`
- `label`
- `status`
- appeal information (`appeal_submitted`, `appeal_reasoning`, `appeal_timestamp`) when applicable

During testing the audit log accumulated multiple entries, including at least one
appeal event recorded as a separate `appeal` event alongside the original
submission.

## Stretch Features

1. **Ensemble Detection** — completed; three weighted signals are blended into a
   single `combined_score`.
2. **Provenance Certificate** — `GET /certificate/<content_id>` returns a
   shareable certificate built from the audit log.
3. **Analytics Dashboard** — `GET /analytics` returns aggregate detection metrics.
4. **Multi-modal Support** — `POST /submit` accepts `content_type` of
   `image_metadata`, flattening the metadata into text for the detector.

## Known Limitations

- The detector is intentionally conservative and can classify polished, AI-like, or
  lightly edited AI text as `uncertain` or even `likely_human`. This reduces false
  accusations against human creators but means some AI-generated content may not be
  flagged.
- `image_metadata` support analyzes the **metadata text** (prompt, caption, tool,
  description, etc.), not the raw image pixels. It reasons about how the image was
  described, not the image content itself.
- The LLM signal depends on an external API and a single model; results can vary and
  the service requires network access and a valid API key.

## AI Tool Usage

1. I used AI assistance to design the Flask endpoint structure and to revise the
   appeal/audit workflow. I reviewed and tested the generated code, paying
   particular attention to the audit log behavior and the label validation logic.
2. I used AI assistance to improve the multi-signal detection design and to plan the
   stretch features. I revised the thresholds and deliberately kept a conservative
   policy to reduce harmful false AI accusations.

I also tested every endpoint manually with `curl` and adjusted the implementation
based on the actual test output.

## Final Demo / Test Evidence

- `/submit` returned structured attribution JSON.
- `/appeal` returned `appeal_received: true` and `status: under_review`.
- `/log` showed both submission and appeal entries.
- `/submit` rate limit returned HTTP 429 after exceeding the limit.
- `/certificate/<content_id>` returned a provenance certificate.
- `/analytics` returned detection counts, appeal rate, and average confidence.
- `image_metadata` submission returned `content_type: image_metadata` with signal scores.
