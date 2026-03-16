# app_single.py
from flask import Flask, request, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Use t5-small for demo (CPU). For better quality use OpenAI later.
summarizer = pipeline("summarization", model="t5-small", device=-1)

HTML = """
<!doctype html>
<title>Notes Summarizer - Single Agent</title>
<h2>Paste notes and click Summarize</h2>
<form method="post">
  <textarea name="text" rows=14 cols=88 placeholder="Paste long notes here...">{{ request_form }}</textarea><br><br>
  <button type="submit">Summarize</button>
</form>
{% if summary %}
<hr>
<h3>Summary:</h3>
<pre style="white-space:pre-wrap">{{ summary }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET","POST"])
def index():
    summary = None
    request_form = ""
    if request.method == "POST":
        text = request.form.get("text","").strip()
        request_form = text
        if text:
            # For long text we truncate per model limits (demo). We'll improve chunking later.
            try:
                out = summarizer(text, max_length=120, min_length=30, truncation=True)
                summary = out[0]["summary_text"]
            except Exception as e:
                summary = f"Error running model: {e}"
    return render_template_string(HTML, summary=summary, request_form=request_form)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
