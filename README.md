# 🚀 SmartHire — Resume to Job Matching and Career Guidance Engine

### 🌐 Live Demo

👉 [Launch SmartHire AI](https://smarthire-ai-resume-job-recommendation-oyhfzlyeq3cqmpvfynaxda.streamlit.app/)

---

## 📌 About The Project

SmartHire is an end-to-end AI-powered platform that analyzes a candidate's resume and connects it with suitable job opportunities.

It uses Natural Language Processing (NLP) and Machine Learning to classify resumes, recommend relevant jobs, identify missing skills, and calculate a resume-to-job fit score.

---

## 🎯 Problem Statement

Recruiters often receive a large number of resumes, making it difficult to quickly identify suitable candidates for different job roles.

At the same time, candidates may struggle to understand which jobs match their profile and which skills they are missing.

SmartHire addresses this problem by analyzing a resume and providing:

- 📄 Resume job-domain classification
- 🔎 Relevant job recommendations
- 🎯 Resume-to-job fit score
- 🧩 Skill gap analysis
- 💡 Career guidance based on the candidate's profile

---

## ✨ Key Features

- 📄 **Resume Classification** — Predicts the most relevant job domain from the uploaded resume.
- 🔎 **Job Recommendation** — Recommends the Top 5 jobs based on resume-job similarity.
- 🎯 **Fit Score** — Calculates how well the resume matches each recommended job.
- 🧩 **Skill Gap Analysis** — Identifies matching skills and highlights skills missing for the recommended job.
- 📊 **Multiple ML Models** — Compares Logistic Regression, Linear SVM, Decision Tree, and Random Forest.
- 📁 **Resume Upload** — Supports TXT, PDF, and DOCX resume files.
- 💻 **Interactive Streamlit App** — Provides an easy-to-use interface for real-time resume analysis.
- 🚀 **Live Deployment** — The complete application is deployed and accessible online.

---

## 🔄 Project Workflow

```text
                📄 Resume Upload
                       │
                       ▼
              🧹 Text Preprocessing
                       │
                       ▼
                 🔢 TF-IDF
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      🤖 Resume Classifier    🔎 Job Matching
             │                   │
             ▼                   ▼
      🎯 Job Domain        📊 Cosine Similarity
                                 │
                                 ▼
                         ⭐ Top 5 Job Recommendations
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             🧩 Skill Gap                 🎯 Fit Score
             Analysis
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                       💡 Career Guidance
                                 │
                                 ▼
                         🌐 Streamlit App
```

## 🛠️ Tech Stack

### Programming & Development
- 🐍 **Python**
- 📓 **Jupyter Notebook / Google Colab**
- 💻 **VS Code**

### Machine Learning
- 🤖 **Scikit-learn**
- 🌲 Random Forest
- 📈 Logistic Regression
- ⚡ Linear SVM
- 🌳 Decision Tree

### NLP
- 🔤 **NLTK**
- 🔢 **TF-IDF Vectorization**
- 📐 **Cosine Similarity**
- 🧹 Text Preprocessing

### Data Processing & Visualization
- 🐼 **Pandas**
- 🔢 **NumPy**
- 📊 **Matplotlib**
- 📉 **Seaborn**

### Deployment
- 🚀 **Streamlit**
- 📦 **Joblib**

---
## 📂 Dataset Used

SmartHire uses two datasets:

- 📄 **Resume Dataset** — Used to train the resume classification model across multiple job roles.
- 💼 **Naukri Job Dataset** — Contains job titles, descriptions, skills, experience, company, and location.

### 🔄 Data Processing

The data is cleaned and transformed using **NLP preprocessing, TF-IDF, and Cosine Similarity** for classification and job matching.

---
## 🤖 Models Used & Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 63.38% |
| Decision Tree | 64.19% |
| Linear SVM | 70.22% |
| **Random Forest** | **72.84%** |

### 🏆 Best Model

**Random Forest (200 trees)** achieved the highest accuracy of **72.84%** using TF-IDF features.

This is a challenging multi-class classification problem because different job roles often have overlapping skills and terminology. The result is a practical baseline for classical NLP + Machine Learning resume classification.

---
## 🧩 Skill Gap Analysis

SmartHire compares the candidate's skills with the skills required by recommended jobs.

It shows:
- ✅ Matching skills
- ❌ Missing skills
- 📊 Skill match percentage

This helps candidates understand what skills they need to improve for their target roles.

---
## 🎯 Resume–Job Fit Score

SmartHire calculates a fit score using:

- 🔤 Resume–job text similarity
- 🧩 Skill overlap

The combined score helps rank recommended jobs and classify them as **Excellent Fit, Good Fit, Moderate Fit, or Low Fit**.

---
## 👩‍💻 About Me

**Sharmeela S**

B.Tech Computer Science Engineering | AI/ML Enthusiast

Passionate about building practical AI/ML solutions using Python, Machine Learning, NLP, and recommendation systems. Currently focused on developing real-world AI applications and strengthening my skills in Generative AI and Agentic AI.
