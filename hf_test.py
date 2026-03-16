from transformers import pipeline
print("Loading summarizer (may download model; wait a bit)...")
summarizer = pipeline("summarization", model="t5-small")
out = summarizer(
    "Machine learning is the study of algorithms that learn from data.",
    max_length=60, min_length=10, truncation=True
)
print("Summary:", out[0]["summary_text"])
