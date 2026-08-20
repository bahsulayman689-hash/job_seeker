import streamlit as st
import io
import re

# --- COMPONENT ENGINES CHECK ---
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
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    HAS_TRANSFORMERS = False

st.set_page_config(page_title="AI Match Bridge", page_icon="🧠", layout="centered")

st.title("🧠 AI Seeker-to-Giver Match Engine")
st.write("The AI evaluates your input directly against the employer's target profiles.")
st.logo("https://streamlit.io/images/brand/streamlit-mark-color.png")

# --- THE GIVER CRITERIA (5 DISTINCT JOB LISTINGS) ---
GIVER_JOBS = [
    {
        "id": 1, 
        "title": "Python Automation Specialist", 
        "keywords": "python coding scripting automation databases excel spreadsheets programming developer backend mechanical task"
    },
    {
        "id": 2, 
        "title": "Social Media Content Creator", 
        "keywords": "canva tiktok video editing instagram graphics creative management photography capcut content creation"
    },
    {
        "id": 3,
        "title": "IT Support & Network Technician",
        "keywords": "hardware troubleshooting software wifi routing operating systems desktop support helpdesk technical configuration"
    },
    {
        "id": 4,
        "title": "Customer Service & Help Assistant",
        "keywords": "communication call center client satisfaction listening soft skills email support problem solving hospitality"
    },
    {
        "id": 5,
        "title": "Administrative & Data Operations Intern",
        "keywords": "typing filing documentation scheduling Microsoft office processing records organization data entry clerk"
    }
]

# Set the strict mathematical threshold for matching acceptance
STRICT_THRESHOLD = 70

# --- THE SEEKER SELECTION FRAME ---
st.subheader("🧑‍🎓 Candidate Evaluation Pipeline")

# Dynamically generate the dropdown list from our expanded database
job_titles = [job["title"] for job in GIVER_JOBS]
target_position = st.selectbox("Select the job position you are targeting:", job_titles)

# Find and isolate the exact active keywords for the selected job position
selected_job = next(item for item in GIVER_JOBS if item["title"] == target_position)

# Choose input type: File or Voice
input_method = st.radio("How would you like to submit your experience profile?", ["📄 Upload PDF Resume", "🎙️ Record Voice Profile"])

extracted_text = ""

# Option 1: PDF Document Upload
if input_method == "📄 Upload PDF Resume":
    uploaded_file = st.file_uploader("Upload your resume file (PDF format)", type=["pdf"])
    if uploaded_file is not None:
        if HAS_PDF:
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() or ""
                st.info("📄 Resume content successfully pulled by PyPDF2.")
            except Exception as e:
                st.error(f"Error parsing PDF payload: {e}")
        else:
            extracted_text = "Placeholder sample background text."
            st.warning("Running sandbox text profile match.")

# Option 2: Voice Input Profile
elif input_method == "🎙️ Record Voice Profile":
    seeker_audio = st.audio_input("Click record and describe your skills and tools out loud")
    if seeker_audio is not None:
        st.audio(seeker_audio, format="audio/wav")
        
        # Smart dynamic mock speech-to-text response based on user selection
        if target_position == "Python Automation Specialist":
            extracted_text = "I build simple automated bots using python and manage backend database entries."
        elif target_position == "Social Media Content Creator":
            extracted_text = "I edit videos for TikTok, make graphics on Canva, and design social content."
        elif target_position == "IT Support & Network Technician":
            extracted_text = "I fix computers, troubleshoot hardware issues, configure operating systems, and set up office wifi."
        elif target_position == "Customer Service & Help Assistant":
            extracted_text = "I love talking to clients, answering customer emails with patience, and solving user complaints."
        else:
            extracted_text = "I am great at file management, typing fast, using Microsoft office tools, and keeping data organized."
            
        st.success(f"🎙️ AI Speech-to-Text Transcribed: \"{extracted_text}\"")

# --- THE AI BRIDGE EVALUATION LAYER ---
if extracted_text:
    st.write("---")
    st.markdown("### 🧠 AI Bridge Evaluation Outcome")
    
    if HAS_TRANSFORMERS:
        with st.spinner("AI is calculating vector embeddings..."):
            model = load_model()
            
            # Vectorize both profiles to find alignment
            job_vector = model.encode(selected_job["keywords"], convert_to_tensor=True)
            seeker_vector = model.encode(extracted_text, convert_to_tensor=True)
            
            # Compute similarity score 
            similarity = util.cos_sim(seeker_vector, job_vector).item()
            match_score = int(max(10, min(similarity * 100 + 40, 99))) # Normalized demo curve
            
            st.metric(label=f"Alignment Score for: {target_position}", value=f"{match_score}%")
            
            # 📊 ACCEPTED OR REJECTED DECISION MATRIX WITH ALTERNATIVE ENGINE
            if match_score >= STRICT_THRESHOLD:
                st.balloons()
                st.success(f"✅ **STATUS: ACCEPTED**\n\nYour profile matches the Giver's requirements perfectly. You have been passed through the bridge to the next hiring stage.")
                
                if HAS_GTTS:
                    tts = gTTS(text="Congratulations! You have been accepted for this position.", lang='en')
                    buffer = io.BytesIO()
                    tts.write_to_fp(buffer)
                    buffer.seek(0)
                    st.audio(buffer, format="audio/mp3")
            else:
                st.error(f"❌ **STATUS: REJECTED**\n\nYour profile score ({match_score}%) fell below the required hiring benchmark ({STRICT_THRESHOLD}%).")
                
                # 🚀 BEST ALTERNATIVE FIT ENGINE RUNNING IN THE BACKGROUND
                best_alt_title = ""
                best_alt_score = 0
                
                for job in GIVER_JOBS:
                    # Skip the job the user already applied for and failed
                    if job["title"] == target_position:
                        continue
                    
                    alt_job_vector = model.encode(job["keywords"], convert_to_tensor=True)
                    alt_similarity = util.cos_sim(seeker_vector, alt_job_vector).item()
                    alt_score = int(max(10, min(alt_similarity * 100 + 40, 99)))
                    
                    if alt_score > best_alt_score:
                        best_alt_score = alt_score
                        best_alt_title = job["title"]
                
                # Check if the alternative match is high enough to suggest
                if best_alt_score >= STRICT_THRESHOLD:
                    st.info(f"✨ **AI Alternative Route Discovery:**\n\nWhile you didn't qualify for this specific role, your background has a high neural compatibility rating (**{best_alt_score}%**) for **{best_alt_title}**. We recommend redirecting your application there!")
                    
                    if HAS_GTTS:
                        tts_text = f"Application reviewed. We found a better match for you as a {best_alt_title}."
                        tts = gTTS(text=tts_text, lang='en')
                        buffer = io.BytesIO()
                        tts.write_to_fp(buffer)
                        buffer.seek(0)
                        st.audio(buffer, format="audio/mp3")
                else:
                    if HAS_GTTS:
                        tts = gTTS(text="Application reviewed. Criteria not met.", lang='en')
                        buffer = io.BytesIO()
                        tts.write_to_fp(buffer)
                        buffer.seek(0)
                        st.audio(buffer, format="audio/mp3")
    else:
        st.error("Sentence-transformers library missing. Run pip installation.")
