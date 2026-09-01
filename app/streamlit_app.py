# ============================================================
# SMARTHIRE AI
# Resume Classification + Job Recommendation + Skill Gap
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import re
import os

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resume_classifier.pkl"
)

RESUME_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

JOB_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "job_tfidf_vectorizer.pkl"
)

JOB_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "models",
    "job_tfidf_matrix.pkl"
)

JOBS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "jobs_clean.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    resume_vectorizer = joblib.load(
        RESUME_TFIDF_PATH
    )

    job_vectorizer = joblib.load(
        JOB_TFIDF_PATH
    )

    job_matrix = joblib.load(
        JOB_MATRIX_PATH
    )

    return (
        model,
        resume_vectorizer,
        job_vectorizer,
        job_matrix
    )


# ============================================================
# LOAD JOB DATA
# ============================================================

@st.cache_data
def load_jobs():

    jobs = pd.read_csv(
        JOBS_PATH
    )

    jobs = jobs.fillna("")

    return jobs


model, resume_vectorizer, job_vectorizer, job_matrix = load_models()

jobs_df = load_jobs()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SKILL LIST
# ============================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "c",
    "javascript",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit learn",
    "scikit-learn",
    "tensorflow",
    "keras",
    "pytorch",
    "matplotlib",
    "seaborn",
    "streamlit",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "linux",
    "spark",
    "hadoop",
    "excel",
    "power bi",
    "tableau",
    "statistics",
    "flask",
    "django",
    "rest api",
    "mongodb",
    "mysql",
    "postgresql",
    "r",
    "scala",
    "html",
    "css"
]


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        skill_clean = skill.lower()

        if skill_clean in text:

            found_skills.append(
                skill
            )

    return sorted(
        list(set(found_skills))
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "💼 SmartHire AI"
)

st.subheader(
    "AI-Powered Resume Classification, Job Recommendation & Skill Gap Analysis"
)

st.write(
    """
    SmartHire analyzes a candidate's resume using
    Natural Language Processing and Machine Learning,
    predicts suitable job categories, recommends relevant
    job opportunities, and identifies skill gaps.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🤖 SmartHire AI"
)

st.sidebar.markdown(
    "### 🚀 Features"
)

st.sidebar.write(
    "✅ Resume Classification"
)

st.sidebar.write(
    "✅ Job Recommendation"
)

st.sidebar.write(
    "✅ Skill Matching"
)

st.sidebar.write(
    "✅ Skill Gap Analysis"
)

st.sidebar.write(
    "✅ Job Similarity"
)

st.sidebar.write(
    "✅ K-Means Job Clustering"
)

st.sidebar.divider()

st.sidebar.metric(
    "Classifier Accuracy",
    "68.81%"
)

st.sidebar.metric(
    "Available Jobs",
    f"{len(jobs_df):,}"
)

st.sidebar.info(
    "Final Model: Random Forest"
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header(
    "📄 Upload Your Resume"
)

uploaded_file = st.file_uploader(
    "Upload your resume as a TXT file",
    type=["txt"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded successfully: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # READ RESUME
    # --------------------------------------------------------

    resume_text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    if not resume_text.strip():

        st.error(
            "The uploaded resume is empty."
        )

        st.stop()


    # --------------------------------------------------------
    # RESUME PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "📋 View Resume Text"
    ):

        st.text(
            resume_text[:5000]
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=False
    )


    if analyze:

        # ====================================================
        # TEXT PREPROCESSING
        # ====================================================

        cleaned_resume = clean_text(
            resume_text
        )


        # ====================================================
        # RESUME CLASSIFICATION
        # ====================================================

        resume_vector = resume_vectorizer.transform(
            [cleaned_resume]
        )

        prediction = model.predict(
            resume_vector
        )

        predicted_category = prediction[0]


        # ====================================================
        # CLASSIFICATION SECTION
        # ====================================================

        st.divider()

        st.header(
            "🎯 Resume Classification"
        )

        st.success(
            f"Predicted Job Category: **{predicted_category}**"
        )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                resume_vector
            )[0]

            classes = model.classes_

            results = sorted(
                zip(classes, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            top_predictions = results[:3]

            st.subheader(
                "📊 Top 3 Model Predictions"
            )

            col1, col2, col3 = st.columns(3)

            columns = [
                col1,
                col2,
                col3
            ]

            medals = [
                "🥇",
                "🥈",
                "🥉"
            ]

            for i, (
                category,
                probability
            ) in enumerate(
                top_predictions
            ):

                with columns[i]:

                    st.metric(
                        f"{medals[i]} {category}",
                        f"{probability * 100:.2f}%"
                    )

            st.caption(
                "Note: Model confidence represents the probability "
                "assigned to each class for this particular resume. "
                "It is different from the overall test accuracy."
            )


        # ====================================================
        # SKILL EXTRACTION
        # ====================================================

        resume_skills = extract_skills(
            resume_text
        )


        # ====================================================
        # JOB RECOMMENDATION
        # ====================================================

        st.divider()

        st.header(
            "💼 Recommended Jobs"
        )

        st.write(
            "Top job opportunities ranked according to "
            "resume-job similarity."
        )


        # ====================================================
        # RESUME → JOB VECTOR
        # ====================================================

        job_resume_vector = job_vectorizer.transform(
            [cleaned_resume]
        )


        # ====================================================
        # COSINE SIMILARITY
        # ====================================================

        similarities = cosine_similarity(
            job_resume_vector,
            job_matrix
        )[0]


        # ====================================================
        # ADD MATCH SCORE
        # ====================================================

        jobs_result = jobs_df.copy()

        jobs_result[
            "match_score"
        ] = similarities * 100


        # ====================================================
        # SORT
        # ====================================================

        jobs_result = jobs_result.sort_values(
            by="match_score",
            ascending=False
        )


        # ====================================================
        # TOP 5 JOBS
        # ====================================================

        top_jobs = jobs_result.head(
            5
        )


        # ====================================================
        # DISPLAY JOBS
        # ====================================================

        for rank, (
            index,
            job
        ) in enumerate(
            top_jobs.iterrows(),
            start=1
        ):

            st.markdown(
                f"### {rank}. 💼 {job['jobtitle']}"
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.write(
                    f"🏢 **Company:** "
                    f"{job['company']}"
                )

                st.write(
                    f"📍 **Location:** "
                    f"{job['joblocation_address']}"
                )

                st.write(
                    f"💼 **Experience:** "
                    f"{job['experience']}"
                )

            with col2:

                st.metric(
                    "🎯 Match Score",
                    f"{job['match_score']:.2f}%"
                )


            # ------------------------------------------------
            # JOB SKILLS
            # ------------------------------------------------

            job_text = ""

            if "job_text" in job:

                job_text = str(
                    job["job_text"]
                )

            else:

                job_text = (
                    str(job.get("jobtitle", ""))
                    + " "
                    + str(job.get("skills", ""))
                    + " "
                    + str(job.get("jobdescription", ""))
                )


            job_skills = extract_skills(
                job_text
            )


            # ------------------------------------------------
            # SKILL MATCH
            # ------------------------------------------------

            if len(job_skills) > 0:

                matching_skills = sorted(
                    set(resume_skills)
                    .intersection(
                        set(job_skills)
                    )
                )

                missing_skills = sorted(
                    set(job_skills)
                    - set(resume_skills)
                )

                skill_match = (
                    len(matching_skills)
                    /
                    len(job_skills)
                ) * 100

            else:

                matching_skills = []

                missing_skills = []

                skill_match = 0


            # ------------------------------------------------
            # SKILL MATCH DISPLAY
            # ------------------------------------------------

            st.subheader(
                "🧩 Skill Analysis"
            )

            st.metric(
                "Skill Match",
                f"{skill_match:.2f}%"
            )


            skill_col1, skill_col2 = st.columns(
                2
            )


            with skill_col1:

                st.markdown(
                    "### ✅ Matching Skills"
                )

                if matching_skills:

                    st.write(
                        ", ".join(
                            matching_skills
                        )
                    )

                else:

                    st.write(
                        "No matching skills detected."
                    )


            with skill_col2:

                st.markdown(
                    "### ❌ Missing Skills"
                )

                if missing_skills:

                    st.write(
                        ", ".join(
                            missing_skills
                        )
                    )

                else:

                    st.write(
                        "No major missing skills detected."
                    )


            # ------------------------------------------------
            # JOB DESCRIPTION
            # ------------------------------------------------

            with st.expander(
                "📖 View Job Description"
            ):

                description = str(
                    job.get(
                        "jobdescription",
                        ""
                    )
                )

                if description:

                    st.write(
                        description[:3000]
                    )

                else:

                    st.write(
                        "Job description not available."
                    )


            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "SmartHire AI | NLP + Machine Learning | "
    "Resume Intelligence → Job Discovery → Skill Growth"
)