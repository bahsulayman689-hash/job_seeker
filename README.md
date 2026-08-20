# 🧠 AI Seeker-to-Giver Match Engine (Hackathon Prototype)

An AI-powered, frictionless career matchmaking network designed to solve youth unemployment. This application completely eliminates traditional recruitment waiting times by acting as an automated intelligence layer between **Job Seekers** and **Job Givers**. 

The system accepts raw inputs—via **PDF Resume upload** or **Live Voice Profiles**—transforms them into high-dimensional vector embeddings, and delivers an instantaneous, objective **Accepted** or **Rejected** hiring decision with an automated sound cue loop. If a seeker falls short of the target role, a background algorithm dynamically re-routes them to their **Best Alternative Fit**.

---

## 🌐 System Architecture Flow

- **Seeker Interfaces:** 📄 Upload PDF Resume OR 🎙️ Record Voice Note
- **Giver Criteria:** 📋 Target Profiles & Keyword Benchmarks (Adjustable Sidebar)
- **The AI Bridge:** PyPDF2 Text Pulling + Vector Embedding Map
- **Cognitive Acceptance:** Instantly passed to Giver + Celebratory balloons + Plays vocal audio confirmation
- **Deviation Engine:** Evaluates background jobs + Suggests Best Alternative Fit + Plays spoken feedback path

---

## 🚀 Core Features

* **Multimodal Input Support:** Youth job seekers can apply by uploading their traditional resume files or by speaking about their casual life experiences directly into their microphone.
* **Explainable Neural Match Matrix:** Powered by `sentence-transformers`, the backend mathematically computes cosine similarities to measure the true contextual alignment of experiences over basic keyword counts.
* **Instant Verification Loop:** Generates dynamic audio tracks on the fly using `gTTS` to provide immediate spoken confirmation of application status.
* **Zero-Waste Talent Re-routing:** When a user is rejected for a specific role, a background looping sub-routine scans the employer's database to suggest an alternative open position that matches their skill set.
* **Dynamic Control Sidebar:** Includes an interactive slider that allows administrators and employers to adjust match thresholds in real time to filter talent pools.

---

## 💻 Technical Stack & Ecosystem

* **Frontend Layout Engine:** Streamlit
* **Document Parser Wrapper:** PyPDF2
* **Acoustic Text-to-Speech Engine:** gTTS (Google Text-to-Speech)
* **High-Dimensional Neural Model:** Sentence-Transformers (`all-MiniLM-L6-v2`)

---

## 🛠️ Step-by-Step Installation

Follow these quick commands to build and test the software ecosystem locally on your terminal:

### 1. Clone or Create Project Folder
```bash
mkdir ai-youth-job-bridge
cd ai-youth-job-bridge
```

### 2. Set Up Dependencies
Create a `requirements.txt` file and populate it with the required libraries:
```text
streamlit>=1.35.0
PyPDF2>=3.0.0
gTTS>=2.5.0
sentence-transformers>=2.7.0
torch>=2.0.0
```

Install the full framework bundle:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application Prototype
Save the main Python solution file as `app.py` in your working directory and execute:
```bash
streamlit run app.py
```

---

## 🏆 Presentation Demo Guide for the Pitch

To deliver a winning presentation to the judging panel, use this step-by-step walkthrough:

1. **Demonstrate Rejection first:** Select *Python Automation Specialist*, slide the sidebar threshold to **70%**, and upload a mismatched file or profile. Show the **❌ STATUS: REJECTED** state. Highlight how traditional HR filters act as dead ends for young job seekers.
2. **Reveal the Intelligent Alternative:** Point out the **AI Alternative Route Discovery** module beneath the rejection alert. Show how the engine automatically matched the profile to an open *IT Support* or *Administrative* role instead.
3. **Demonstrate Acceptance:** Lower the threshold slider or update your input text to include high-relevance automation keywords. Trigger the **✅ STATUS: ACCEPTED** state to launch the on-screen balloons and audio feedback!
