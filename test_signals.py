"""
Quick local sanity check for the pure-Python detection signals.

This does NOT call the Groq API or start the server, so it needs no API key.
It only exercises Signal 2 (stylometric) and Signal 3 (repetition) plus the
ensemble math, using a stubbed llm_score.

Run it with:
    python test_signals.py
"""

from app import classify, get_repetition_score, get_stylometric_score

SAMPLES = {
    "clearly_ai": (
        "Artificial intelligence is a transformative technology in our "
        "rapidly evolving world. It is important to note that AI plays a "
        "crucial role across industries. Furthermore, it continues to grow. "
        "Furthermore, it continues to expand. In conclusion, AI is important."
    ),
    "clearly_human": (
        "I burned the toast again this morning. Honestly? I don't even like "
        "toast that much — I just keep making it out of habit, which is "
        "ridiculous. My cat watched the whole disaster from the windowsill, "
        "judging me, and then demanded breakfast like nothing happened!"
    ),
    "formal_human": (
        "The committee reviewed the quarterly budget on Tuesday. Several "
        "members raised concerns about the marketing allocation. After a long "
        "discussion, we agreed to revisit the figures next month, once the "
        "regional sales data arrives."
    ),
}


def main():
    for name, text in SAMPLES.items():
        stylometric = get_stylometric_score(text)
        repetition = get_repetition_score(text)
        # Stub the LLM score so we can see the ensemble without an API call.
        stub_llm = 0.85 if name == "clearly_ai" else 0.2
        attribution, confidence, _, combined = classify(
            stub_llm, stylometric, repetition
        )
        print(f"--- {name} ---")
        print(f"  stylometric_score: {stylometric}")
        print(f"  repetition_score:  {repetition}")
        print(f"  (stub) llm_score:  {stub_llm}")
        print(f"  combined_score:    {combined}")
        print(f"  attribution:       {attribution}")
        print(f"  confidence:        {confidence}")
        print()


if __name__ == "__main__":
    main()
