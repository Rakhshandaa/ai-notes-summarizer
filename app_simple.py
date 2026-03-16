from flask import Flask, request, render_template_string
from transformers import pipeline
from PyPDF2 import PdfReader
import os

app = Flask(__name__)
app.secret_key = "dev"

# Load local summarizer
summarizer = pipeline("summarization", model="t5-small")

HTML = """
<h2>Simple Notes Summarizer (No API, No Errors)</h2>
<form method="post" enctype="multipart/form-data">
  <textarea name="text" rows="12" cols="90">{{ text }}</textarea><br><br>
  Upload PDF/TXT: <input type="file" name="file"><br><br>
  <button type="submit">Summarize</button>
</form>

{% if summary %}
<hr>
<h3>Summary:</h3>
<pre style="white-space:pre-wrap">{{ summary }}</pre>
{% endif %}
"""

def extract_file_text(file):
    filename = file.filename.lower()
    fp = "uploaded." + filename.split(".")[-1]
    file.save(fp)

    if filename.endswith(".pdf"):
        reader = PdfReader(fp)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return text

    elif filename.endswith(".txt"):
        return open(fp, "r", encoding="utf-8", errors="ignore").read()

    return ""

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = ""

    if request.method == "POST":
        if request.files.get("file"):
            text = extract_file_text(request.files["file"])
        else:
            text = request.form.get("text", "")

        if text:
            summary = summarizer(text[:2000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]

    return render_template_string(HTML, text=text, summary=summary)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
