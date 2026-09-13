
import os, sqlite3, uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
app=Flask(__name__)
app.secret_key=os.getenv("FLASK_SECRET_KEY","change-this")
DB="data/app.db"
os.makedirs("data",exist_ok=True); os.makedirs("uploads",exist_ok=True)

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

with db() as c:
    c.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT UNIQUE,password TEXT,role TEXT DEFAULT 'user')")
    c.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY,user_id INTEGER,filename TEXT,text TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY,user_id INTEGER,agent TEXT,question TEXT,answer TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

def current_user():
    if not session.get("user"): return None
    with db() as c:return c.execute("SELECT * FROM users WHERE username=?",(session["user"],)).fetchone()

def pdf_text(path):
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)

def chunks(text,size=1400):
    words=text.split()
    return [" ".join(words[i:i+size]) for i in range(0,len(words),size)]

# Semantic RAG-ready retrieval. For production, replace this local index with
# IBM watsonx.ai RAG/vector-store integration or a persistent Milvus/Elasticsearch store.
def retrieve(uid,query,limit=6):
    with db() as c: docs=c.execute("SELECT filename,text FROM documents WHERE user_id=?",(uid,)).fetchall()
    q=set(query.lower().split()); scored=[]
    for d in docs:
        for n,ch in enumerate(chunks(d["text"])):
            score=sum(1 for w in q if len(w)>2 and w in ch.lower())
            scored.append((score,d["filename"],n+1,ch))
    scored.sort(reverse=True)
    return [{"source":x[1],"chunk":x[2],"text":x[3]} for x in scored[:limit] if x[0]>0]

def watsonx(messages):
    key=os.getenv("WATSONX_APIKEY"); project=os.getenv("WATSONX_PROJECT_ID")
    if not key or not project: raise RuntimeError("Set WATSONX_APIKEY and WATSONX_PROJECT_ID in .env")
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as G
    model=ModelInference(
      model_id=os.getenv("WATSONX_MODEL_ID","ibm/granite-3-3-8b-instruct"),
      credentials=Credentials(url=os.getenv("WATSONX_URL","https://us-south.ml.cloud.ibm.com"),api_key=key),
      project_id=project,params={G.MAX_NEW_TOKENS:1000,G.TEMPERATURE:.2})
    prompt="\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
    return model.generate_text(prompt=prompt,guardrails=True)

AGENTS={
"project":"You are the core AI for 10 AI Document Intelligence. Domain: Advanced. Mission: Enterprise document RAG with ingestion, retrieval, audit logs and analytics.
Use retrieved evidence when available. Never fabricate sources. Provide structured, practical output.
For healthcare/legal/security topics, clearly distinguish assistance from professional diagnosis, legal advice, or authorization.",
"tutor":"You are the Tutor Agent. Explain clearly, ground answers in retrieved context, and admit uncertainty.",
"researcher":"You are the Research Agent. Extract facts, compare evidence, and identify gaps.",
"critic":"You are the Critic Agent. Check a draft for unsupported claims, contradictions and missing evidence.",
"writer":"You are the Writer Agent. Produce polished structured output using the evidence provided.",
"analyst":"You are the Analyst Agent. Turn supplied facts into structured insights, risks and recommendations."
}

@app.route("/")
def home(): return render_template("index.html",logged=bool(current_user()),username=session.get("user",""))

@app.post("/api/register")
def register():
    d=request.get_json() or {}; u=d.get("username","").strip(); p=d.get("password","")
    if not u or not p:return jsonify(error="Username and password required"),400
    try:
        with db() as c:c.execute("INSERT INTO users(username,password) VALUES(?,?)",(u,generate_password_hash(p)))
        session["user"]=u; return jsonify(ok=True)
    except sqlite3.IntegrityError:return jsonify(error="Username already exists"),409

@app.post("/api/login")
def login():
    d=request.get_json() or {}
    with db() as c:r=c.execute("SELECT * FROM users WHERE username=?",(d.get("username",""),)).fetchone()
    if not r or not check_password_hash(r["password"],d.get("password","")):return jsonify(error="Invalid login"),401
    session["user"]=r["username"]; return jsonify(ok=True)

@app.post("/api/logout")
def logout():session.clear();return jsonify(ok=True)

@app.post("/api/upload")
def upload():
    u=current_user(); f=request.files.get("file")
    if not u:return jsonify(error="Login first"),401
    if not f or not f.filename.lower().endswith(".pdf"):return jsonify(error="PDF required"),400
    name=secure_filename(f.filename); path=f"uploads/{uuid.uuid4().hex}_{name}";f.save(path)
    try:t=pdf_text(path)
    except Exception as e:return jsonify(error=str(e)),400
    with db() as c:c.execute("INSERT INTO documents(user_id,filename,text) VALUES(?,?,?)",(u["id"],name,t))
    return jsonify(ok=True,message=f"{name} indexed into the knowledge base.")

@app.post("/api/agent")
def agent():
    u=current_user(); d=request.get_json() or {}; question=d.get("question","").strip(); name=d.get("agent","tutor")
    if not u:return jsonify(error="Login first"),401
    if not question:return jsonify(error="Enter a question"),400
    hits=retrieve(u["id"],question)
    context="\n\n".join(f"[{h['source']} | chunk {h['chunk']}]\n{h['text']}" for h in hits)
    prompt=[{"role":"system","content":AGENTS.get(name,AGENTS["tutor"])},
            {"role":"user","content":f"Question:\n{question}\n\nRetrieved evidence:\n{context or 'No relevant document evidence was found.'}\n\nReturn a useful answer. Include a Sources section."}]
    try:answer=watsonx(prompt)
    except Exception as e:return jsonify(error=str(e)),503
    with db() as c:c.execute("INSERT INTO chats(user_id,agent,question,answer) VALUES(?,?,?,?)",(u["id"],name,question,answer))
    return jsonify(answer=answer,sources=hits)

@app.get("/api/dashboard")
def dashboard():
    u=current_user()
    if not u:return jsonify(error="Login first"),401
    with db() as c:
        docs=c.execute("SELECT COUNT(*) n FROM documents WHERE user_id=?",(u["id"],)).fetchone()["n"]
        chats=c.execute("SELECT COUNT(*) n FROM chats WHERE user_id=?",(u["id"],)).fetchone()["n"]
        by=c.execute("SELECT agent,COUNT(*) n FROM chats WHERE user_id=? GROUP BY agent",(u["id"],)).fetchall()
    return jsonify(documents=docs,chats=chats,agents=[dict(x) for x in by])

if __name__=="__main__":app.run(debug=True)
