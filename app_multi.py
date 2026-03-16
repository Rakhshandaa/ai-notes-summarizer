# app_multi.py — Final Version (Groq backend + download + API)

from dotenv import load_dotenv
load_dotenv()

import os
import time
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, render_template_string, flash, send_file
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# Try importing Groq client
try:
    from groq import Groq
except:
    Groq = None

# -------------------------
# Config
# -------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize client
groq_client = None
if GROQ_KEY and Groq:
    try:
        groq_client = Groq(api_key=GROQ_KEY)
        print("Groq client initialized")
    except Exception as e:
        print("Groq client error:", e)

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)
app.secret_key = "dev-key"

# -------------------------
# UI Template
# -------------------------
HTML = """
<!doctype html>
<html>
<head>
<title>Notes Summarizer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{background:#f8f9fa;padding:30px}
.card{margin-top:20px}
pre{background:white;padding:15px;border-radius:6px}
</style>
</head>
<body>

<div class="container">

<h3>AI Notes Summarizer</h3>
<p class="text-muted">Model: {{model}}</p>

<form method="post" enctype="multipart/form-data">

<textarea name="text" class="form-control" rows="10" placeholder="Paste notes here">{{text_input}}</textarea>

<br>

<input type="file" name="file" class="form-control">

<br>

<button class="btn btn-primary">Summarize</button>

</form>

{% with messages = get_flashed_messages() %}
{% if messages %}
<div class="alert alert-danger mt-3">
{% for m in messages %}
<div>{{m}}</div>
{% endfor %}
</div>
{% endif %}
{% endwith %}

{% if summary %}
<div class="card">
<div class="card-body">

<div class="d-flex justify-content-between">
<h5>Final Summary</h5>

<form action="/download" method="post">
<input type="hidden" name="summary" value="{{summary}}">
<button class="btn btn-sm btn-outline-secondary">Download</button>
</form>

</div>

<hr>

<pre>{{summary}}</pre>

</div>
</div>
{% endif %}

{% if debug %}
<div class="card">
<div class="card-body">
<h5>Chunk Summaries</h5>
<hr>
<pre>{{debug}}</pre>
</div>
</div>
{% endif %}

</div>
</body>
</html>
"""

# -------------------------
# Text chunking
# -------------------------
def chunk_text(text, chunk_size=3000, overlap=200):

    chunks = []
    i = 0

    while i < len(text):

        end = min(i + chunk_size, len(text))
        chunk = text[i:end]

        chunks.append(chunk)

        i += chunk_size - overlap

    return chunks


# -------------------------
# Groq call
# -------------------------
def groq_call(prompt, max_tokens=300):

    if groq_client:

        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role":"system","content":"You summarize text clearly."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            return f"[Groq error: {e}]"

    else:

        headers={
            "Authorization":f"Bearer {GROQ_KEY}",
            "Content-Type":"application/json"
        }

        body={
            "model":GROQ_MODEL,
            "messages":[
                {"role":"system","content":"You summarize text clearly."},
                {"role":"user","content":prompt}
            ],
            "temperature":0.2,
            "max_tokens":max_tokens
        }

        try:

            r=requests.post(GROQ_URL,headers=headers,json=body,timeout=120)

            if r.status_code!=200:
                return f"[Groq REST error {r.status_code}]"

            data=r.json()

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            return f"[Groq REST error: {e}]"


# -------------------------
# Agents
# -------------------------
def summarizer_agent(chunk):

    prompt=f"Summarize clearly:\n\n{chunk}"

    result=groq_call(prompt)

    print("CHUNK RESULT:",result[:120])

    return result


def refiner_agent(summaries):

    combined="\n".join(summaries)

    prompt=f"Combine these summaries into a final summary:\n\n{combined}"

    return groq_call(prompt)


def orchestrator_multi_agent(text):

    chunks=chunk_text(text)

    with ThreadPoolExecutor(max_workers=4) as ex:

        summaries=list(ex.map(summarizer_agent,chunks))

    final_summary=refiner_agent(summaries)

    return final_summary,summaries


# -------------------------
# File extraction
# -------------------------
def extract_text(fp,filename):

    if filename.endswith(".txt"):

        with open(fp,"r",encoding="utf-8",errors="ignore") as f:
            return f.read()

    if filename.endswith(".pdf"):

        reader=PdfReader(fp)

        text=""

        for page in reader.pages:
            text+=page.extract_text() or ""

        return text

    return ""


# -------------------------
# Main page
# -------------------------
@app.route("/",methods=["GET","POST"])
def index():

    summary=""
    debug=""
    text_input=""

    if request.method=="POST":

        file=request.files.get("file")

        if file and file.filename:

            filename=secure_filename(file.filename)

            fp=os.path.join(UPLOAD_FOLDER,filename)

            file.save(fp)

            text_input=extract_text(fp,filename)

        else:

            text_input=request.form.get("text","").strip()

        if text_input:

            start=time.time()

            summary,chunks=orchestrator_multi_agent(text_input)

            debug="\n\n---\n\n".join(chunks)

            print("TOTAL TIME:",time.time()-start)

        else:

            flash("Paste text or upload a file")

    return render_template_string(
        HTML,
        summary=summary,
        debug=debug,
        text_input=text_input,
        model=GROQ_MODEL
    )


# -------------------------
# Download summary
# -------------------------
@app.route("/download",methods=["POST"])
def download():

    summary=request.form.get("summary","")

    file=BytesIO()

    file.write(summary.encode())

    file.seek(0)

    return send_file(
        file,
        as_attachment=True,
        download_name="summary.txt",
        mimetype="text/plain"
    )


# -------------------------
# API endpoint (React)
# -------------------------
@app.route("/api/summarize",methods=["POST"])
def api_summarize():

    data=request.get_json()

    if not data or "text" not in data:
        return {"error":"Missing text"},400

    summary,chunks=orchestrator_multi_agent(data["text"])

    return {
        "summary":summary,
        "chunks":chunks
    }


# -------------------------
# Run
# -------------------------
if __name__=="__main__":

    print("Starting Notes Summarizer")

    print("Model:",GROQ_MODEL)

    app.run(debug=True,port=5000)