## Milestone 4 Implementation Update

For Milestone 4, I implemented a three-signal attribution pipeline:

1. LLM classification score
2. Stylometric score
3. Repetition / generic phrase score

The final AI-likelihood score is calculated with the planned ensemble formula:

combined_score = (0.50 * llm_score) + (0.30 * stylometric_score) + (0.20 * repetition_score)

The system then maps the combined score into one of three attribution outcomes:

* 0.00–0.39: likely_human
* 0.40–0.74: uncertain
* 0.75–1.00: likely_ai

I tested four deliberately chosen inputs: one AI-like sample, one clearly human-written sample, one formal human borderline sample, and one lightly edited AI-style borderline sample. The AI-like sample scored noticeably higher than the clearly human-written sample, while the borderline examples stayed closer to the human/uncertain boundary.

The AI-like test returned an uncertain attribution rather than a high-confidence AI label because the system uses a conservative threshold before labeling content as likely AI. I kept this behavior intentionally because false AI accusations can harm creators. The important checkpoint behavior was still satisfied: the signals produced different scores for different writing styles, the combined score was logged, and the audit log recorded each individual signal.
