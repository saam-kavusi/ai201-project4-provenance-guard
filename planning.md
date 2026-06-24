## Milestone 4 Implementation Update

For Milestone 4, I implemented a three-signal attribution pipeline:

1. LLM classification score
2. Stylometric score
3. Repetition / generic phrase score

The final AI-likelihood score is calculated with the planned ensemble formula:

combined_score = (0.50 * llm_score) + (0.30 * stylometric_score) + (0.20 * repetition_score)

The system then maps the combined score into one of three attribution outcomes:

* 0.00–0.39: likely_human
* 0.40–0.64: uncertain
* 0.65–1.00: likely_ai

I tested four deliberately chosen inputs: one AI-like sample, one clearly human-written sample, one formal human borderline sample, and one lightly edited AI-style borderline sample. The AI-like sample scored noticeably higher than the clearly human-written sample, while the borderline examples stayed closer to the human/uncertain boundary.

The AI-like test returned an uncertain attribution rather than a high-confidence AI label because the system uses a conservative threshold before labeling content as likely AI. I kept this behavior intentionally because false AI accusations can harm creators. The important checkpoint behavior was still satisfied: the signals produced different scores for different writing styles, the combined score was logged, and the audit log recorded each individual signal.

## Stretch Feature Plan

With Milestone 5 finished and committed, the following stretch features were planned. They are ordered by readiness and dependency.

### Implementation Update

All four stretch features have been completed and are in the code:

1. **Ensemble Detection** — implemented with `llm_score`, `stylometric_score`, `repetition_score`, and the weighted `combined_score` formula.
2. **Provenance Certificate** — implemented as `GET /certificate/<content_id>`, returning `certificate_id`, `content_id`, `creator_id`, `attribution`, `confidence`, `label`, `timestamp`, `audit_status`, `appeal_submitted`, and `signal_scores`.
3. **Analytics Dashboard** — implemented as `GET /analytics`, returning `total_submissions`, `total_appeals`, `detection_counts`, `appeal_rate`, `average_confidence`, and `average_combined_score`.
4. **Multi-modal Support** — implemented by extending `POST /submit` with `content_type`. It supports `"text"` by default and `"image_metadata"` using a metadata object that is converted into text for the existing detector.

The original plan for each feature is kept below for reference.

### 1. Ensemble Detection (already implemented)

The ensemble detector is already in place and uses three independent signals:

* `llm_score` — LLM classification score
* `stylometric_score` — stylometric analysis score
* `repetition_score` — repetition / generic phrase score

The signals are combined with a weighted formula:

combined_score = (0.50 * llm_score) + (0.30 * stylometric_score) + (0.20 * repetition_score)

**Conflict resolution.** When the three signals disagree, the weighted ensemble resolves the conflict rather than requiring a unanimous vote. Each signal contributes proportionally to its weight, so the LLM score dominates (highest weight, 0.50), the stylometric score is the second strongest influence (0.30), and the repetition score breaks remaining ties as the lightest contributor (0.20). This ordering reflects how much we trust each signal: the LLM classifier sees the most semantic context, stylometry captures structural writing patterns, and repetition is a useful but noisier surface signal. A single signal therefore cannot force a high-confidence label on its own, which keeps the conservative behavior described above.

### 2. Provenance Certificate

Add a certificate object and a corresponding endpoint that returns a structured, shareable record of an attribution decision. The certificate will include:

* `content_id` — identifier for the analyzed content
* `attribution` — the outcome label (likely_human / uncertain / likely_ai)
* `confidence` — confidence score for the decision
* `label` — human-readable label
* `timestamp` — when the certificate was issued
* `signal_scores` — the individual llm/stylometric/repetition scores
* `audit_status` — whether the result has been audited or appealed

This gives downstream consumers a single object that captures both the verdict and the evidence behind it.

### 3. Analytics Dashboard

Add a simple `/analytics` endpoint that summarizes activity across all analyzed content. It will report:

* detection counts (totals per attribution outcome)
* appeal rate (share of results that were appealed)
* average confidence across decisions

This is intended as a lightweight monitoring view rather than a full dashboard.

### 4. Multi-modal Support

The system was extended to accept a second content type beyond text. `POST /submit` now takes a `content_type` field that supports `"text"` by default and `"image_metadata"`, where a metadata object is converted into text so the existing signal pipeline can be reused on it, while leaving room to add image-specific signals later. This was the lowest-priority stretch item and was completed after the others were stable.
