# ============================================================
# SMARTHIRE AI
# Resume-to-Job Matching & Career Guidance Engine
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #172554, #2563eb);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.job-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128, 128, 128, 0.35);
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    margin-bottom: 15px;
}

.job-card h3,
.job-card p,
.job-card b {
    color: var(--text-color) !important;
}

.score {
    font-size: 28px;
    font-weight: bold;
}

.skill {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 15px;
    background-color: #fee2e2;
    color: #991b1b;
    font-size: 14px;
}

.success-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #dcfce7;
    color: #166534;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "jobs_clean.csv"

CLASSIFIER_PATH = ROOT / "models" / "resume_classifier.pkl"
RESUME_TFIDF_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

JOB_TFIDF_PATH = ROOT / "models" / "job_tfidf_vectorizer.pkl"
JOB_MATRIX_PATH = ROOT / "models" / "job_tfidf_matrix.pkl"


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_jobs():

    jobs = pd.read_csv(DATA_PATH)

    jobs["job_text"] = jobs["job_text"].fillna("")

    return jobs


@st.cache_resource
def load_models():

    classifier = joblib.load(CLASSIFIER_PATH)
    resume_tfidf = joblib.load(RESUME_TFIDF_PATH)

    job_tfidf = joblib.load(JOB_TFIDF_PATH)
    job_matrix = joblib.load(JOB_MATRIX_PATH)

    return classifier, resume_tfidf, job_tfidf, job_matrix


jobs_df = load_jobs()

classifier, resume_tfidf, job_tfidf, job_matrix = load_models()


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    # TXT
    if file_name.endswith(".txt"):

        return uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    # PDF
    elif file_name.endswith(".pdf"):

        try:

            from PyPDF2 import PdfReader

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            return text

        except Exception:

            st.error(
                "PDF reading requires PyPDF2. "
                "Install it using: pip install PyPDF2"
            )

            return ""

    # DOCX
    elif file_name.endswith(".docx"):

        try:

            from docx import Document

            document = Document(uploaded_file)

            text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

            return text

        except Exception:

            st.error(
                "DOCX reading requires python-docx. "
                "Install it using: pip install python-docx"
            )

            return ""

    return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SKILLS
# ============================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "c",
    "sql",
    "machine learning",
    "artificial intelligence",
    "data science",
    "deep learning",
    "natural language processing",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "scikit-learn",
    "streamlit",
    "git",
    "javascript",
    "html",
    "css",
    "nlp",
    "excel",
    "tableau",
    "power bi",
    "spark",
    "hadoop",
    "linux"
]


def extract_skills(text):

    text = text.lower()

    found_skills = set()

    for skill in SKILLS:

        if skill in text:
            found_skills.add(skill)

    return found_skills


# ============================================================
# SKILL OVERLAP
# ============================================================

def calculate_skill_overlap(
    resume_skills,
    job_text
):

    job_skills = extract_skills(job_text)

    if len(job_skills) == 0:
        return 0

    matching_skills = resume_skills.intersection(
        job_skills
    )

    return (
        len(matching_skills)
        /
        len(job_skills)
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>💼 SmartHire AI</h1>

<p>
Resume-to-Job Matching & Career Guidance Engine
</p>

<p>
Upload your resume and discover the jobs, skills and career
opportunities that match your profile.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ SmartHire")

    st.write(
        "AI-powered resume analysis and job matching."
    )

    st.divider()

    st.write("📊 Dataset")

    st.metric(
        "Available Jobs",
        f"{len(jobs_df):,}"
    )

    st.divider()

    st.write("🔍 System Modules")

    st.write("✅ Resume Classification")
    st.write("✅ Job Recommendation")
    st.write("✅ Fit Score")
    st.write("✅ Skill Gap Analysis")


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📄 Upload Your Resume</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload TXT, PDF or DOCX",
    type=["txt", "pdf", "docx"]
)


# ============================================================
# MAIN PROCESS
# ============================================================

if uploaded_file is not None:

    resume_text = extract_resume_text(
        uploaded_file
    )

    if resume_text.strip() == "":

        st.error(
            "Could not extract text from the resume."
        )

        st.stop()


    # --------------------------------------------------------
    # CLEAN RESUME
    # --------------------------------------------------------

    cleaned_resume = clean_text(
        resume_text
    )


    # ========================================================
    # RESUME CLASSIFICATION
    # ========================================================

    prediction_vector = resume_tfidf.transform(
        [cleaned_resume]
    )

    predicted_category = classifier.predict(
        prediction_vector
    )[0]


    # ========================================================
    # RESUME SKILLS
    # ========================================================

    resume_skills = extract_skills(
        cleaned_resume
    )


    # ========================================================
    # JOB RECOMMENDATION
    # ========================================================

    resume_job_vector = job_tfidf.transform(
        [cleaned_resume]
    )

    similarity_scores = cosine_similarity(
        resume_job_vector,
        job_matrix
    ).flatten()


    jobs_result = jobs_df.copy()

    jobs_result["match_score"] = (
        similarity_scores * 100
    )


    # ========================================================
    # SKILL OVERLAP
    # ========================================================

    jobs_result["skill_overlap"] = jobs_result[
        "job_text"
    ].apply(
        lambda x:
        calculate_skill_overlap(
            resume_skills,
            x
        )
    )


    # ========================================================
    # FIT SCORE
    # ========================================================

    jobs_result["fit_score"] = (

        0.6 * jobs_result["match_score"]

        +

        0.4 *
        jobs_result["skill_overlap"] * 100
    )


    # ========================================================
    # TOP JOBS
    # ========================================================

    top_jobs = jobs_result.sort_values(
        "fit_score",
        ascending=False
    ).head(5)


    # ========================================================
    # DASHBOARD SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Resume Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"### 🎓 Predicted Domain\n\n"
            f"**{predicted_category}**"
        )

    with col2:

        st.success(
            f"### 🧠 Skills Detected\n\n"
            f"**{len(resume_skills)} skills**"
        )

    with col3:

        st.warning(
            f"### 💼 Jobs Analyzed\n\n"
            f"**{len(jobs_df):,} jobs**"
        )


    # ========================================================
    # DETECTED SKILLS
    # ========================================================

    st.markdown(
        '<div class="section-title">🛠️ Detected Skills</div>',
        unsafe_allow_html=True
    )

    if resume_skills:

        skills_text = " • ".join(
            sorted(resume_skills)
        )

        st.success(
            skills_text
        )

    else:

        st.warning(
            "No predefined skills detected."
        )


    # ========================================================
    # JOB RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">🚀 Top Job Recommendations</div>',
        unsafe_allow_html=True
    )


    for index, (_, job) in enumerate(
        top_jobs.iterrows(),
        start=1
    ):

        job_title = job.get(
            "jobtitle",
            "Job Opportunity"
        )

        company = job.get(
            "company",
            "Company not specified"
        )

        location = job.get(
            "joblocation_address",
            "Location not specified"
        )

        fit_score = job["fit_score"]

        match_score = job["match_score"]

        skill_overlap = (
            job["skill_overlap"] * 100
        )


        # ----------------------------------------------------
        # FIT LEVEL
        # ----------------------------------------------------

        if fit_score >= 70:

            fit_level = "Excellent Fit 🌟"

        elif fit_score >= 50:

            fit_level = "Good Fit 👍"

        elif fit_score >= 30:

            fit_level = "Moderate Fit 🟡"

        else:

            fit_level = "Low Fit 🔴"


        # ----------------------------------------------------
        # SKILL GAP
        # ----------------------------------------------------

        job_skills = extract_skills(
            job["job_text"]
        )

        missing_skills = sorted(
            job_skills - resume_skills
        )


        # ----------------------------------------------------
        # JOB CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="job-card">

            <h3>
            {index}. {job_title}
            </h3>

            <p>
            🏢 <b>{company}</b>
            &nbsp;&nbsp; | &nbsp;&nbsp;
            📍 {location}
            </p>

            <p>
            <b>🎯 Fit:</b> {fit_level}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Overall Fit",
                f"{fit_score:.1f}%"
            )

        with col2:

            st.metric(
                "Resume-Job Match",
                f"{match_score:.1f}%"
            )

        with col3:

            st.metric(
                "Skill Match",
                f"{skill_overlap:.1f}%"
            )


        # ----------------------------------------------------
        # SKILL GAP
        # ----------------------------------------------------

        if missing_skills:

            with st.expander(
                "🔎 View Skill Gap"
            ):

                st.write(
                    "Skills recommended for this job:"
                )

                for skill in missing_skills:

                    st.markdown(
                        f'<span class="skill">{skill}</span>',
                        unsafe_allow_html=True
                    )

        else:

            st.success(
                "🎉 Your detected skills cover the "
                "skills identified in this job."
            )


        st.divider()


    # ========================================================
    # CAREER SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 SmartHire Career Insight</div>',
        unsafe_allow_html=True
    )


    best_job = top_jobs.iloc[0]

    best_title = best_job.get(
        "jobtitle",
        "your recommended role"
    )

    best_score = best_job["fit_score"]


    st.markdown(
        f"""
        <div class="success-box">

        <h3>🌟 Your strongest opportunity</h3>

        <p>
        Based on your resume skills and job-description similarity,
        <b>{best_title}</b> has the highest calculated fit score
        among the analyzed opportunities.
        </p>

        <p>
        <b>Fit Score: {best_score:.1f}%</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown("""
    <div style="
        text-align:center;
        padding:60px 20px;
        border-radius:18px;
        border:2px dashed #cbd5e1;
        margin-top:25px;
    ">

    <h2>👋 Welcome to SmartHire AI</h2>

    <p style="font-size:18px;">
    Upload your resume above to start your AI-powered
    career analysis.
    </p>

    <p>
    🔹 Discover your job domain<br>
    🔹 Find matching job opportunities<br>
    🔹 Calculate job-fit scores<br>
    🔹 Identify missing skills
    </p>

    </div>
    """, unsafe_allow_html=True)