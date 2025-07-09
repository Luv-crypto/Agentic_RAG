
import faulthandler
faulthandler.enable()      # prints traceback even inside threads

import os, uuid, datetime, markdown2
from pathlib import Path
from typing import List, Tuple
import threading, secrets
import sqlite3, hashlib, secrets
import re
from config import ALL_DOMAINS
from agentic_rag_agent import get_agent
from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory ,redirect, url_for,abort
    )

# ---------- your RAG core (imported) ---------------------------
from dotenv import load_dotenv

# Load .env into process environment
load_dotenv()


# ---------- constants ------------------------------------------
ROOT               = Path(__file__).parent.resolve()
SESSION_COOKIE_KEY = "sid"

# ---------- Flask -------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

# ---------- in-memory session store ------------------------------
# { session_id : [ {role, html, ts}, ... ] }
CHAT_LOGS = {}
INGEST_TASKS = {}


# ---------------- helper -----------------------------------------j
# ---------------- helper -----------------------------------------
def _chat_key(req) -> str:
    """
    Return the key under which this user's chat history is stored.
    Uses uid cookie; if somehow anonymous, falls back to a per-browser UUID.
    """
    uid = _current_uid(req)
    if uid is not None:                     # logged-in user
        return f"user_{uid}"                # e.g. "user_5"

    # ------- anonymous (should not happen after login guard) -------
    anon = req.cookies.get("sid")
    if not anon:
        anon = str(uuid.uuid4())
    return f"anon_{anon}"



        # ← only the autonomous agent
# -----------------------------------------------------------------------

# pre-compute every domain’s image / table directory
IMAGE_DIRS = {cfg.object_store_dirs["image"].resolve()
              for cfg in ALL_DOMAINS.values()}
TABLE_DIRS = {cfg.object_store_dirs["table"].resolve()
              for cfg in ALL_DOMAINS.values()}

MEDIA_TOKEN_RE = re.compile(r"<<(img|tbl):([0-9A-Fa-f]{8}|[0-9A-Fa-f\-]{32,36})>>")

def _run_rag(prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Pure agent wrapper — no direct smart_query.

    Returns
    -------
    html_answer : safe HTML string
    media_list  : [("img", url), ("tbl", url), …]  (may be empty)
    """
    uid = _current_uid(request)
    agent = get_agent(uid)                   # ← user-scoped agent
    answer_md = agent.run(prompt)
    html_answer = markdown2.markdown(
        answer_md, extras=["fenced-code-blocks", "tables"]
    )


    media_show: List[Tuple[str, str]] = []
    seen_files: set[str] = set()

    # Collect every unique image/table token → URL
    for kind, token in MEDIA_TOKEN_RE.findall(answer_md):
        kind = kind.lower()
        prefix = token[:8].lower()  # 8-char UUID prefix

        if kind == "img":
            # search each domain folder for a matching PNG
            for img_dir in IMAGE_DIRS:
                for cand in img_dir.glob(f"{prefix}*"):
                    fname = cand.name
                    if fname not in seen_files:
                        seen_files.add(fname)
                        media_show.append(("img", f"/media/image/{fname}"))
                    break
                # continue to next token, don't break out completely

        elif kind == "tbl":
            # search each domain folder for a matching MD
            for tbl_dir in TABLE_DIRS:
                for cand in tbl_dir.glob(f"{prefix}*"):
                    fname = cand.name
                    if fname not in seen_files:
                        seen_files.add(fname)
                        media_show.append(("tbl", f"/media/table/{fname}"))
                    break
                # continue to next token

    # ───────────────────────────────────────────────────────────────
    # Now return the HTML + full media_show list (multiple entries allowed)
    # ───────────────────────────────────────────────────────────────
    return html_answer, media_show




# ---------------- routes -----------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True)
    user_msg = (data or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    key = _chat_key(request)

# 1) store user message
    # 1) store user msg
    uid = _current_uid(request)
    db = _get_db()
    db.execute("INSERT INTO chats (user_id,role,html,ts) VALUES (?,?,?,?)",
            (uid, "user", markdown2.markdown(user_msg), datetime.datetime.utcnow().isoformat(timespec="seconds")   # Fix A
))
    db.commit()



    # 2) call RAG
    try:
        answer_html, media = _run_rag(user_msg)
    except Exception as e:
        answer_html = f"<p style='color:red'>Server error: {e}</p>"
        media = []

    # 3) store assistant answer
    # 3) build <img> / <iframe> tags once so history can replay them
    media_html = []
    for kind, url in media:
        if kind == "img":
            media_html.append(f'<img src="{url}" class="inline-img">')
        else:  # tables
            media_html.append(f'<iframe src="{url}" class="tbl-frame"></iframe>')

    db.execute("INSERT INTO chats (user_id,role,html,ts) VALUES (?,?,?,?)",
           (uid, "assistant", answer_html + "".join(media_html),
           datetime.datetime.utcnow().isoformat(timespec="seconds") ))
    db.commit()


    return jsonify({
        "answer_html": answer_html,
        "media": media
    })





# expose figures / tables ------------------------------------------------
@app.route("/media/image/<path:filename>")
def media_image(filename):
    """Look for the image in any domain’s images/ folder."""
    for fp in Path("object_store").rglob(f"images/{filename}"):
        if fp.exists():
            return send_from_directory(fp.parent, fp.name)
    return abort(404, description="image not found")


@app.route("/media/table/<path:filename>")
def media_table(filename):
    """Look for the markdown table in any domain’s tables/ folder."""
    for fp in Path("object_store").rglob(f"tables/{filename}"):
        if fp.exists():
            md_text = fp.read_text(encoding="utf-8")
            html = markdown2.markdown(md_text, extras=["tables"])
            css  = (
                "<style>"
                "table{border-collapse:collapse;width:100%;}"
                "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}"
                "</style>"
            )
            return f"<html><head>{css}</head><body>{html}</body></html>"
    return abort(404, description="table not found")




# recent history (optional sidebar) --------------------------------------
@app.route("/history")
def history():
    uid = _current_uid(request)
    if uid is None:
        return jsonify([])
    rows = _get_db().execute(
        "SELECT role,html,ts FROM chats WHERE user_id=? ORDER BY id", (uid,)
    ).fetchall()
    return jsonify([{"role":r[0],"html":r[1],"ts":r[2]} for r in rows])



# ───────────────────────────────────────────────────────────────
# 0)  Imports & database helpers
# ───────────────────────────────────────────────────────────────

DB_PATH = ROOT / "users.db"


def _get_db():
    db = sqlite3.connect(DB_PATH)
    # ── ensure users table exists ─────────────────────────────────────
    db.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE,
        pw_hash   TEXT
      )
    """)
    # ── ensure chats table exists ─────────────────────────────────────
    db.execute("""
      CREATE TABLE IF NOT EXISTS chats (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER,
        role      TEXT,    -- 'user' or 'assistant'
        html      TEXT,
        ts        TEXT
      )
    """)
    return db

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ───────────────────────────────────────────────────────────────
# 1)  User-session helpers
# ───────────────────────────────────────────────────────────────
USER_COOKIE = "uid"

def _current_uid(req) -> int | None:
    try:
        return int(req.cookies.get(USER_COOKIE))
    except (TypeError, ValueError):
        return None
    

def _login_resp(uid: int, resp):
    resp.set_cookie(USER_COOKIE, str(uid), max_age=60*60*24*30, httponly=True)
    return resp

# ───────────────────────────────────────────────────────────────
# 2)  Auth routes
# ───────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = request.form["user"].strip()
        pw   = request.form["pw"]
        db   = _get_db()
        try:
            db.execute("INSERT INTO users (username,pw_hash) VALUES (?,?)",
                       (user, _hash_pw(pw)))
            db.commit()
        except sqlite3.IntegrityError:
            return "Username taken", 400
        uid = db.execute("SELECT id FROM users WHERE username=?", (user,)).fetchone()[0]
        resp = redirect(url_for("index"))          # 302 → /
        return _login_resp(uid, resp)
    return render_template("login.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["user"].strip()
        pw   = request.form["pw"]
        db   = _get_db()
        row  = db.execute("SELECT id,pw_hash FROM users WHERE username=?", (user,)).fetchone()
        if not row or _hash_pw(pw) != row[1]:
            return "Bad credentials", 401
        resp = redirect(url_for("index"))          # 302 → /
        return _login_resp(row[0], resp)
        
    return render_template("login.html", mode="login")

@app.route("/logout")
def logout():
    resp = redirect(url_for("login"))
    resp.delete_cookie(USER_COOKIE)
    resp.delete_cookie(SESSION_COOKIE_KEY)
    return resp

from flask import redirect, url_for

# ───────────────────────────────────────────────────────────────
#  GLOBAL LOGIN REQUIRED (except a few routes)
# ───────────────────────────────────────────────────────────────
PUBLIC_PATHS = {"/login", "/register", "/static/", "/media/"}

@app.before_request
def force_login():
    # Let Flask serve static files & auth pages without a login
    path = request.path
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return

    # Otherwise require a valid user cookie
    if _current_uid(request) is None:
        return redirect(url_for("login"))


# ───────────────────────────────────────────────────────────────
# 3)  PDF-upload route
# ───────────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────────
# Upload  →  Ingest  →  Cancel
# ───────────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """GET  → show upload page
       POST → save PDF, return JSON {file_path} (no ingestion yet)"""
    uid = _current_uid(request)
    if uid is None:
        return "Login first", 401

    if request.method == "GET":
        return render_template("upload.html")

    # POST branch
    f = request.files.get("pdf")
    if not (f and f.filename.lower().endswith(".pdf")):
        return "PDF only", 400

    user_dir = UPLOAD_DIR / f"user_{uid}"
    user_dir.mkdir(exist_ok=True)

    save_path = user_dir / f"{secrets.token_hex(8)}_{f.filename}"
    f.save(save_path)
    return jsonify({"file_path": str(save_path)}), 200


@app.route("/ingest", methods=["POST"])
def ingest():
    """Kick off ingestion in a background thread, return task_id."""
    uid = _current_uid(request)
    data = request.get_json(force=True)
    path = Path(data.get("file_path", ""))
    if not path.exists():
        return "file not found", 400

    stop_flag = threading.Event()
    task_id   = secrets.token_hex(8)

    def worker():
        from rag_scipdf_core import ingest_documents
        try:
            ingest_documents(str(path), stop_event=stop_flag, user_id=uid)
            INGEST_TASKS[task_id]["status"] = "complete"
        except Exception as e:
            INGEST_TASKS[task_id]["status"] = f"failed: {e}"
        finally:
            # leave dict entry until front-end fetches status once more
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    INGEST_TASKS[task_id] = {"thread": t, "stop": stop_flag,"status":"running"}
    return jsonify({"task_id": task_id}), 202


@app.route("/ingest/cancel/<task_id>", methods=["POST"])
def cancel_ingest(task_id):
    task = INGEST_TASKS.get(task_id)
    if not task: return "task not found", 404
    task["stop"].set()
    task["status"] = "cancelled"
    return "cancelled", 200

@app.route("/ingest/status/<task_id>")
def ingest_status(task_id):
    task = INGEST_TASKS.get(task_id)
    if not task:
        return jsonify({"status": "unknown"}), 404
    return jsonify({"status": task.get("status", "running")})






# ───────── main ─────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000,debug=True)   # ← add this


