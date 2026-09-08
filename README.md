 🚀 SmartHire AI
### Resume-to-Job Matching & Career Guidance System

SmartHire AI is a Machine Learning and NLP based career assistance system that helps candidates find suitable jobs based on their resume.

Instead of only predicting a job category, SmartHire provides:
✅ Resume Classification
✅ Job Recommendations
✅ Skill Gap Analysis
✅ Job Fit Score

---

## 🎯 Problem

Job seekers often face difficulty in:

• Finding jobs that match their skills
• Understanding which career role suits them
• Identifying missing skills
• Choosing the right jobs from thousands of listings

SmartHire AI solves these problems using Machine Learning and NLP.

---

## 💡 How SmartHire Works
--
                 USER RESUME
                      ↓
              Extract Resume Text
                      ↓
                Text Cleaning
                      ↓
          ┌───────────┼───────────┐
          ↓           ↓           ↓
     TF-IDF       Skill         Clean
     Features    Extraction      Text
          ↓           ↓           ↓
   Random Forest      │     Job Matching
   Classification     │           ↓
          ↓           │    Cosine Similarity
    Job Category      │           ↓
                      │      Top 5 Jobs
                      │           ↓
                      └────→ Skill Gap
                                  ↓
                              Fit Score
                                  ↓
                         Career Guidance
