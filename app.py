import io
import re
import ssl
import smtplib
import hashlib
from email.message import EmailMessage
import pandas as pd
import streamlit as st

# ---------- DEPENDENCY CHECKS ----------
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_LIB = True
except ImportError:
    HAS_GEMINI_LIB = False
try:
    from supabase import create_client
    HAS_SUPABASE_LIB = True
except ImportError:
    HAS_SUPABASE_LIB = False

GEMINI_MODEL = "gemini-2.5-flash"
HAS_GEMINI_KEY = HAS_GEMINI_LIB and "GEMINI_API_KEY" in st.secrets
USE_DB = HAS_SUPABASE_LIB and "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets
HAS_EMAIL = "SMTP_USER" in st.secrets and "SMTP_APP_PASSWORD" in st.secrets
STATUSES = ["new", "shortlisted", "interview", "not selected"]

st.set_page_config(page_title="AI Match Bridge", page_icon="🧠", layout="wide")

# =====================================================================
# STORAGE LAYER (Supabase if configured, otherwise in-memory demo mode)
# =====================================================================
DEFAULT_JOBS = [
    {"title": "Python Automation Specialist",
     "description": "Write Python scripts that automate repetitive tasks, process spreadsheets and manage databases. Build small backend tools and maintain them.",
     "skills": ["python", "automation", "excel", "database", "sql", "scripting", "git"], "threshold": 60},
    {"title": "IT Support & Network Technician",
     "description": "Troubleshoot hardware and software problems, configure operating systems, set up WiFi and routers, and support staff at the help desk.",
     "skills": ["hardware", "troubleshooting", "wifi", "router", "windows", "networking", "helpdesk"], "threshold": 60},
]


@st.cache_resource
def sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def _mem():
    if "mem" not in st.session_state:
        jobs = [dict(j, id=i + 1) for i, j in enumerate(DEFAULT_JOBS)]
        st.session_state["mem"] = {"jobs": jobs, "apps": [], "next_job": len(jobs) + 1, "next_app": 1}
    return st.session_state["mem"]


def get_jobs():
    if USE_DB:
        return sb().table("jobs").select("*").order("created_at", desc=True).execute().data
    return list(_mem()["jobs"])


def add_job(title, description, skills, threshold, employer_email=""):
    row = {"title": title, "description": description, "skills": skills, "threshold": threshold}
    if employer_email:
        row["employer_email"] = employer_email
    if USE_DB:
        sb().table("jobs").insert(row).execute()
    else:
        m = _mem()
        m["jobs"].insert(0, dict(row, id=m["next_job"]))
        m["next_job"] += 1


def delete_job(job_id):
    if USE_DB:
        sb().table("jobs").delete().eq("id", job_id).execute()
    else:
        m = _mem()
        m["jobs"] = [j for j in m["jobs"] if j["id"] != job_id]
        m["apps"] = [a for a in m["apps"] if a["job_id"] != job_id]


def save_application(job_id, name, contact, score, found, missing):
    row = {"job_id": job_id, "name": name, "contact": contact, "score": score,
           "found": found, "missing": missing}
    if USE_DB:
        sb().table("applications").upsert(row, on_conflict="job_id,contact").execute()
    else:
        m = _mem()
        m["apps"] = [a for a in m["apps"] if not (a["job_id"] == job_id and a["contact"] == contact)]
        m["apps"].append(dict(row, id=m["next_app"], status="new"))
        m["next_app"] += 1


def get_applications(job_id):
    if USE_DB:
        return (sb().table("applications").select("*").eq("job_id", job_id)
                .order("score", desc=True).execute().data)
    return sorted([a for a in _mem()["apps"] if a["job_id"] == job_id],
                  key=lambda a: a["score"], reverse=True)


def get_all_applications():
    if USE_DB:
        return sb().table("applications").select("*").order("score", desc=True).execute().data
    return sorted(_mem()["apps"], key=lambda a: a["score"], reverse=True)


def update_status(app_id, status):
    if USE_DB:
        sb().table("applications").update({"status": status}).eq("id", app_id).execute()
    else:
        for a in _mem()["apps"]:
            if a["id"] == app_id:
                a["status"] = status


def send_employer_email(to_email, job, name, contact, score, found, missing):
    """Returns (ok, message). The application is already saved before this runs."""
    if not HAS_EMAIL:
        return False, "Email is not configured (SMTP_USER / SMTP_APP_PASSWORD missing)."
    if not to_email:
        return False, "No employer email set for this job."
    msg = EmailMessage()
    msg["Subject"] = f"New candidate for {job['title']}: {name} ({score}%)"
    msg["From"] = st.secrets["SMTP_USER"]
    msg["To"] = to_email
    status = "Meets" if score >= job["threshold"] else "Below"
    msg.set_content(
        f"A new candidate sent their result for: {job['title']}\n\n"
        f"Name: {name}\n"
        f"Contact: {contact}\n"
        f"Match score: {score}% ({status} the {job['threshold']}% threshold)\n"
        f"Skills found: {', '.join(found) or 'none'}\n"
        f"Skills not found: {', '.join(missing) or 'none'}\n\n"
        "Open the Employer dashboard, then Ranked candidates, to review everyone.\n"
        "Scores are AI guidance only. Please review each candidate yourself.")
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(st.secrets["SMTP_USER"], st.secrets["SMTP_APP_PASSWORD"])
            server.send_message(msg)
        return True, "Employer notified by email."
    except Exception as e:
        return False, f"Email failed: {e}"


# =====================================================================
# AI HELPERS
# =====================================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, size=150, overlap=30):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)] or [text]


def score_match(resume_text, job):
    model = load_model()
    job_vec = model.encode(job["description"], convert_to_tensor=True)
    chunk_vecs = model.encode(chunk_text(resume_text), convert_to_tensor=True)
    best = float(util.cos_sim(chunk_vecs, job_vec).flatten().max())
    semantic = min(max((best - 0.15) / (0.55 - 0.15), 0.0), 1.0)  # tune on real resumes
    low = resume_text.lower()
    found = [s for s in job["skills"] if re.search(rf"\b{re.escape(s)}", low)]
    missing = [s for s in job["skills"] if s not in found]
    cov = len(found) / max(len(job["skills"]), 1)
    return round((0.5 * semantic + 0.5 * cov) * 100), found, missing


@st.cache_data(show_spinner=False)
def tts_bytes(text):
    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    return buf.getvalue()


def speak(text):
    if HAS_GTTS:
        try:
            st.audio(tts_bytes(text), format="audio/mp3")
        except Exception:
            st.caption("Audio feedback unavailable (needs internet).")


@st.cache_data(show_spinner=False)
def transcribe_bytes(data):
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    part = types.Part.from_bytes(data=data, mime_type="audio/wav")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, "Transcribe this audio exactly. Return only the transcript."])
    return (resp.text or "").strip()


def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join((p.extract_text() or "") for p in reader.pages).strip()


# =====================================================================
# SEEKER PAGE
# =====================================================================
def seeker_page():
    st.title("🧠 AI Seeker-to-Giver Match Engine")
    st.write("The AI compares your experience with the employer's target profile and shows why.")

    jobs = get_jobs()
    if not jobs:
        st.info("No jobs have been posted yet. Check back soon.")
        return

    target = st.selectbox("Select the job position you are targeting:", [j["title"] for j in jobs])
    job = next(j for j in jobs if j["title"] == target)
    with st.expander("About this role"):
        st.write(job["description"])
        st.caption("Key skills: " + ", ".join(job["skills"]))

    method = st.radio("How would you like to submit your experience profile?",
                      ["📄 Upload PDF Resume", "🎙️ Record Voice Profile", "⌨️ Type or paste"])
    text = ""

    if method.startswith("📄"):
        up = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if up:
            if not HAS_PDF:
                st.error("PDF parser missing. Add PyPDF2 to requirements.txt.")
            else:
                try:
                    text = read_pdf(up)
                    if len(text) < 100:
                        st.warning("Very little text found. This may be a scanned PDF. Use 'Type or paste'.")
                        text = ""
                except Exception as e:
                    st.error(f"Could not read the PDF: {e}")
    elif method.startswith("🎙️"):
        rec = st.audio_input("Record and describe your skills and tools")
        if rec:
            st.audio(rec)
            if not HAS_GEMINI_KEY:
                st.error("Voice transcription needs GEMINI_API_KEY in Streamlit secrets.")
            else:
                data = rec.getvalue()
                try:
                    with st.spinner("Transcribing..."):
                        raw = transcribe_bytes(data)
                    text = st.text_area("Transcript (edit if needed):", raw, height=150,
                                        key="tr_" + hashlib.md5(data).hexdigest())
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
    else:
        text = st.text_area("Describe your skills, tools and experience:", height=200).strip()

    if not text:
        return
    if len(text.split()) < 8:
        st.info("Add a few more details (at least 8 words) so the AI can evaluate.")
        return
    if not HAS_TRANSFORMERS:
        st.error("sentence-transformers is not installed.")
        return

    st.write("---")
    st.markdown("### 🧠 AI Match Result")
    with st.spinner("Analysing..."):
        score, found, missing = score_match(text, job)
    thr = job["threshold"]
    st.metric(f"Match score for {target}", f"{score}%",
              "Meets employer threshold" if score >= thr else f"Below {thr}% threshold")
    st.progress(score / 100)
    if found:
        st.success("✅ Skills found: " + ", ".join(found))
    if missing:
        st.error("❌ Skills not found: " + ", ".join(missing))
    if score >= thr:
        st.success("Strong match. Your profile fits this role's requirements.")
        speak("Strong match. Your profile fits this role.")
    else:
        st.warning("Below the threshold. Add missing skills to your resume if you have them.")
        speak("Below the threshold. See the missing skills on screen.")

    st.markdown("#### Other roles that may fit you")
    others = sorted(((score_match(text, j)[0], j["title"]) for j in jobs if j["id"] != job["id"]),
                    reverse=True)[:3]
    for s, t in others:
        st.write(f"**{t}**: {s}%")

    st.write("---")
    st.markdown("#### 📨 Send my result to the employer (optional)")
    with st.form("apply_form"):
        name = st.text_input("Full name")
        contact = st.text_input("Phone or email")
        consent = st.checkbox("I agree to share my name, contact, score and skills with the employer. "
                              "My resume text is not stored.")
        if st.form_submit_button("Send"):
            if not (name.strip() and contact.strip()):
                st.error("Enter your name and contact.")
            elif not consent:
                st.error("Please tick the consent box.")
            else:
                save_application(job["id"], name.strip(), contact.strip(), score, found, missing)
                st.success("Sent. The employer can now see your result.")
                to = job.get("employer_email") or st.secrets.get("NOTIFY_EMAIL", "")
                ok, info = send_employer_email(to, job, name.strip(), contact.strip(), score, found, missing)
                if ok:
                    st.caption("The employer was also notified by email.")
                else:
                    st.caption("Your result is saved. (Email notice not sent.)")
                    st.session_state["last_email_error"] = info


# =====================================================================
# EMPLOYER PAGE
# =====================================================================
def employer_page():
    st.title("🏢 Employer Dashboard")

    if "EMPLOYER_PASSWORD" not in st.secrets:
        st.error("Set EMPLOYER_PASSWORD in Streamlit secrets to enable this page.")
        return
    if not st.session_state.get("emp_ok"):
        with st.form("login"):
            pw = st.text_input("Employer passcode", type="password")
            if st.form_submit_button("Enter"):
                if pw == st.secrets["EMPLOYER_PASSWORD"]:
                    st.session_state["emp_ok"] = True
                    st.rerun()
                else:
                    st.error("Wrong passcode.")
        return

    if not USE_DB:
        st.warning("Demo mode: data is kept in memory and disappears on refresh. "
                   "Add SUPABASE_URL and SUPABASE_KEY to secrets for permanent storage.")

    tab_new, tab_jobs, tab_all, tab_rank = st.tabs(
        ["➕ Create job", "📋 My jobs", "👥 All candidates", "🏆 Ranked candidates"])

    with tab_new:
        with st.form("job_form", clear_on_submit=True):
            title = st.text_input("Job title")
            desc = st.text_area("Job description (what the person will do)", height=140)
            skills_raw = st.text_input("Required skills, separated by commas",
                                       placeholder="python, excel, sql, communication")
            emp_email = st.text_input("Email to notify when a candidate applies (optional)",
                                      placeholder="hr@company.com")
            thr = st.slider("Match threshold (%)", 40, 90, 60, 5)
            if st.form_submit_button("Create job"):
                skills = list(dict.fromkeys(s.strip().lower() for s in skills_raw.split(",") if s.strip()))
                if not title.strip() or len(desc.split()) < 8 or len(skills) < 2:
                    st.error("Enter a title, a description (8+ words) and at least 2 skills.")
                else:
                    if emp_email.strip() and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", emp_email.strip()):
                        st.error("That email address does not look valid.")
                    else:
                        add_job(title.strip(), desc.strip(), skills, thr, emp_email.strip())
                        st.success(f"Created: {title.strip()}")

    jobs = get_jobs()

    with tab_jobs:
        if not jobs:
            st.info("No jobs yet.")
        for j in jobs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{j['title']}**  \nThreshold {j['threshold']}% | Skills: {', '.join(j['skills'])}"
                        + (f" | Notify: {j['employer_email']}" if j.get("employer_email") else ""))
            if c2.button("Delete", key=f"del_{j['id']}"):
                delete_job(j["id"])
                st.rerun()

    with tab_all:
        all_apps = get_all_applications()
        if not all_apps:
            st.info("No candidates yet. They appear here after they send their result.")
        else:
            titles = {j["id"]: j["title"] for j in jobs}
            adf = pd.DataFrame(all_apps)
            adf["job"] = adf["job_id"].map(titles).fillna("(deleted job)")
            adf["skills found"] = adf["found"].apply(lambda x: ", ".join(x or []))
            adf["skills missing"] = adf["missing"].apply(lambda x: ", ".join(x or []))
            adf["applied"] = pd.to_datetime(adf.get("created_at"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M") \
                if "created_at" in adf else ""

            f1, f2, f3 = st.columns(3)
            job_pick = f1.multiselect("Job", sorted(adf["job"].unique()))
            status_pick = f2.multiselect("Status", STATUSES)
            min_s = f3.slider("Minimum score", 0, 100, 0, 5, key="all_min")
            q = st.text_input("Search name or contact")

            v = adf
            if job_pick:
                v = v[v["job"].isin(job_pick)]
            if status_pick:
                v = v[v["status"].isin(status_pick)]
            v = v[v["score"] >= min_s]
            if q.strip():
                ql = q.strip().lower()
                v = v[v["name"].str.lower().str.contains(ql) | v["contact"].str.lower().str.contains(ql)]
            v = v.sort_values("score", ascending=False)

            m1, m2, m3 = st.columns(3)
            m1.metric("Candidates shown", len(v))
            m2.metric("Total candidates", len(adf))
            m3.metric("Average score", f"{v['score'].mean():.0f}%" if len(v) else "-")

            cols = ["name", "contact", "job", "score", "status", "skills found", "skills missing", "applied"]
            cols = [c for c in cols if c in v.columns]
            st.dataframe(v[cols], hide_index=True, use_container_width=True,
                         column_config={"score": st.column_config.ProgressColumn(
                             "score", min_value=0, max_value=100, format="%d%%")})
            st.download_button("Download all as CSV", v[cols].to_csv(index=False).encode(),
                               file_name="all_candidates.csv", mime="text/csv")
            st.caption("To change a candidate's status, use the Ranked candidates tab.")

    with tab_rank:
        if not jobs:
            st.info("Create a job first.")
            return
        title = st.selectbox("Job", [j["title"] for j in jobs], key="rank_job")
        job = next(j for j in jobs if j["title"] == title)
        apps = get_applications(job["id"])
        if not apps:
            st.info("No candidates yet. Candidates appear here after they send their result.")
            return

        min_score = st.slider("Show candidates with score at least", 0, 100, 0, 5)
        df = pd.DataFrame(apps)
        df = df[df["score"] >= min_score].sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
        df["skills found"] = df["found"].apply(lambda x: ", ".join(x or []))
        df["skills missing"] = df["missing"].apply(lambda x: ", ".join(x or []))
        view = df[["id", "rank", "name", "contact", "score", "skills found", "skills missing", "status"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", len(df))
        c2.metric("Meet threshold", int((df["score"] >= job["threshold"]).sum()))
        c3.metric("Top score", f"{int(df['score'].max())}%" if len(df) else "-")

        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, key=f"ed_{job['id']}",
            disabled=[c for c in view.columns if c != "status"],
            column_config={
                "id": None,
                "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%d%%"),
                "status": st.column_config.SelectboxColumn("status", options=STATUSES),
            })

        b1, b2 = st.columns(2)
        if b1.button("Save status changes"):
            changed = edited[edited["status"].values != view["status"].values]
            for _, r in changed.iterrows():
                update_status(r["id"], r["status"])
            st.success(f"Updated {len(changed)} candidate(s).")
            st.rerun()
        b2.download_button("Download CSV", view.drop(columns=["id"]).to_csv(index=False).encode(),
                           file_name=f"{job['title'].replace(' ', '_')}_candidates.csv",
                           mime="text/csv")
        st.caption("Scores are guidance from an AI model. Review every candidate before deciding.")


# =====================================================================
# NAVIGATION
# =====================================================================
with st.sidebar:
    st.markdown("### 🧠 AI Match Bridge")
    role = st.radio("I am a", ["Job seeker", "Employer"])
    with st.expander("⚙️ System health"):
        st.write(f"PDF parser: {'🟢' if HAS_PDF else '🔴'}")
        st.write(f"Audio (gTTS): {'🟢' if HAS_GTTS else '🔴'}")
        st.write(f"Embedder: {'🟢' if HAS_TRANSFORMERS else '🔴'}")
        st.write(f"Voice transcription: {'🟢' if HAS_GEMINI_KEY else '🔴'}")
        st.write(f"Database: {'🟢 Supabase' if USE_DB else '🟡 Demo (memory)'}")
        st.write(f"Email alerts: {'🟢' if HAS_EMAIL else '🔴 Needs SMTP secrets'}")
        if st.session_state.get("last_email_error"):
            st.caption("Last email error: " + st.session_state["last_email_error"])

if role == "Job seeker":
    seeker_page()
else:
    employer_page()