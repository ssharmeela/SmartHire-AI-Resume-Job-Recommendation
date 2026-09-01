# SmartHire — Complete Classical ML Project

This package is built around the user's actual Resume.csv dataset. The dataset contains ID, Resume_str, Resume_html and Category; the classifier uses Resume_str to predict Category.

## Project criteria covered
- Resume preprocessing / EDA
- TF-IDF NLP representation
- Multi-class resume category classification
- Logistic Regression, Linear SVM, Decision Tree, Random Forest comparison
- GridSearchCV tuning
- Accuracy, precision, recall, F1 and confusion matrix
- Content-based job recommendation using TF-IDF + cosine similarity
- Top-N ranking
- Skill-gap analysis using a predefined skill vocabulary
- Job clustering + silhouette score + 2D visualization
- Streamlit web demo

The official brief says the core passing scope is resume classifier + job recommender + skill-gap report; clustering is also required in the Week 2 milestone, while fit predictor and topic modeling are optional. See the supplied brief.

## FIRST TIME SETUP — Windows / VS Code
Open the project folder in VS Code, then Terminal → New Terminal.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for this session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Train everything
```powershell
python src/build_demo_jobs.py
python src/train_classifier.py
python -c "import sys; sys.path.insert(0,'src'); import clustering; clustering.run()"
```

## Run the website
```powershell
streamlit run app/streamlit_app.py
```

## IMPORTANT JOB DATA NOTE
The included jobs_clean.csv is a small DEMO corpus so the application can run immediately. For final submission, replace it with a real public Naukri/LinkedIn job dataset and normalize it to these columns:
`title, company, location, skills, description, experience, text`.
Do not claim the demo rows are real job postings.

## Dataset
`data/raw/Resume.csv` is the actual uploaded dataset supplied for this project.
