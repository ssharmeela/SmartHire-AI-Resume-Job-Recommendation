# 🚀 SmartHire AI — Resume-to-Job Matching & Career Guidance Engine

> **An AI/ML-powered career assistant that transforms a resume into job recommendations, skill-gap insights, job-domain classification, and an explainable fit score.**

SmartHire AI is a machine-learning based recruitment and career guidance system designed to reduce the gap between **what a candidate knows** and **what the job market requires**.

Instead of simply predicting a job category, SmartHire answers four practical questions:

**📌 What type of role does my resume match?**  
**📌 Which jobs are most relevant to me?**  
**📌 Which skills am I missing?**  
**📌 How well does my profile match the recommended job?**

---

## 🎯 Problem Statement

Job seekers often face three major problems:

- They don't know which job roles best match their profile.
- Finding relevant jobs among thousands of listings is difficult.
- Job descriptions contain skills that candidates may not have.

At the same time, recruiters have to process large numbers of resumes manually.

### 💡 Our Solution

SmartHire AI uses classical Machine Learning and NLP techniques to convert unstructured resume and job-description text into meaningful career recommendations.

The system combines:

```text
Resume
   ↓
Text Processing
   ↓
Machine Learning + NLP
   ↓
Job Classification
   ↓
Job Recommendation
   ↓
Skill Gap Analysis
   ↓
Fit Score
   ↓
Career Guidance

---

## 🧠 Key Features

### 1️⃣ Resume Classification
Predicts the job category that best matches the candidate's resume.

**TF-IDF → Random Forest**

**Best Accuracy: 72.84%**

### 2️⃣ Job Recommendation
Compares the resume with 21,851 cleaned job listings and recommends the Top 5 most relevant jobs.

**TF-IDF → Cosine Similarity → Top 5 Jobs**

### 3️⃣ Skill Gap Analysis
Compares the candidate's skills with the skills required by recommended jobs.

**Output:**
- ✅ Matching Skills
- ❌ Missing Skills
- 📊 Skill Match Percentage

### 4️⃣ Fit Score
Combines text similarity and skill overlap to provide an explainable job-fit score.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Random Forest
- Cosine Similarity
- K-Means Clustering
- Streamlit
- Joblib
- Git & GitHub

---

## 📊 Project Results

| Component | Result |
|---|---|
| Resume Dataset | 2,484 resumes |
| Resume Categories | 24 |
| Job Dataset | 21,851 cleaned jobs |
| Best Classifier | Random Forest |
| Classification Accuracy | **72.84%** |
| Recommended Jobs | Top 5 |
| Skill Analysis | Matching + Missing Skills |

---

## 🖥️ Application

The project is deployed as a Streamlit application.

### User Flow

```text
Upload Resume
      ↓
Extract & Clean Text
      ↓
Classify Resume
      ↓
Detect Skills
      ↓
Recommend Top 5 Jobs
      ↓
Calculate Skill Match
      ↓
Identify Missing Skills
      ↓
Calculate Fit Score
      ↓
Career Guidance
